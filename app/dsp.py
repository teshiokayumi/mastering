"""音声処理の本体。画面やサーバーには依存しない。

入力ファイルは読み取るだけで、書き込みは常に呼び出し側が指定した出力先に行う。
"""
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import fft as sp_fft
from scipy import ndimage, signal

CD_RATE = 44100
CEILING_DBTP = -1.0
# リミッター後の誤差と書き出し時のディザを見込んだ余裕
_LIMITER_MARGIN_DB = 0.1
_MIN_LUFS = -70.0
_MAX_GAIN_DB = 60.0

# 4倍では高域のピークを最大0.5dBほど読み逃すため8倍にする
_TP_FACTOR = 8
_TP_MARGIN = 256
_TP_CHUNK = 1 << 15


class AudioError(Exception):
    """利用者にそのまま見せられるメッセージを持つエラー"""


@dataclass
class MasterOptions:
    hpf: bool = True
    hpf_freq: float = 30.0
    tone: bool = True
    low_db: float = 0.0
    mid_db: float = 0.0
    high_db: float = 0.0
    glue: bool = True
    glue_db: float = 2.0
    release: bool = True
    target_lufs: float = -12.0
    format: str = "cd"


@dataclass
class ProcessResult:
    source: dict
    output: dict
    glue_db: float = 0.0
    reduction_db: float = 0.0
    warnings: list = field(default_factory=list)
    waveform: list = field(default_factory=list)


def _db_to_lin(db):
    return 10.0 ** (db / 20.0)


def _lin_to_db(value):
    return 20.0 * np.log10(max(value, 1e-12))


# ---------------------------------------------------------------- 読み書き

def read_audio(path):
    try:
        data, rate = sf.read(str(path), dtype="float64", always_2d=True)
    except (sf.LibsndfileError, RuntimeError, ValueError) as exc:
        raise AudioError("音声ファイルとして読み込めませんでした。ファイルが壊れている可能性があります。") from exc
    if data.size == 0:
        raise AudioError("音声データが入っていません。")
    if data.shape[1] > 2:
        raise AudioError("モノラルまたはステレオのファイルのみ処理できます。")
    return np.nan_to_num(data, posinf=0.0, neginf=0.0), rate


def _source_subtype(path):
    return sf.info(str(path)).subtype_info


def write_wav(path, x, rate, fmt):
    """TPDFディザをかけて整数PCMで書き出す。"""
    bits = 16 if fmt == "cd" else 24
    scale = float(2 ** (bits - 1))
    rng = np.random.default_rng()
    dither = rng.random(x.shape) - rng.random(x.shape)
    quantized = np.clip(np.floor(x * scale + dither + 0.5), -scale, scale - 1)
    if bits == 16:
        sf.write(str(path), quantized.astype(np.int16), rate, subtype="PCM_16", format="WAV")
    else:
        # libsndfile の int32 は左詰めで扱われる
        sf.write(str(path), quantized.astype(np.int32) << 8, rate, subtype="PCM_24", format="WAV")


def write_preview(path, x, rate):
    """試聴用WAV。処理前の音を変えずにブラウザで再生するためのもの。"""
    sf.write(str(path), np.clip(x, -1.0, 1.0), rate, subtype="PCM_24", format="WAV")


def resample(x, rate, new_rate):
    if rate == new_rate:
        return x
    divisor = gcd(rate, new_rate)
    up, down = new_rate // divisor, rate // divisor
    largest = max(up, down)
    fir = signal.firwin(2 * 128 * largest + 1, 0.97 / largest, window=("kaiser", 12.0))
    return signal.resample_poly(x, up, down, axis=0, window=fir)


# ---------------------------------------------------------------- 測定

def _k_weighting(rate):
    # ITU-R BS.1770 の2段フィルターを任意のサンプルレート向けに求める
    k = np.tan(np.pi * 1681.974450955533 / rate)
    q = 0.7071752369554196
    vh = 10.0 ** (3.999843853973347 / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = [
        (vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
        1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0,
    ]
    k = np.tan(np.pi * 38.13547087602444 / rate)
    q = 0.5003270373238773
    a0 = 1.0 + k / q + k * k
    highpass = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def integrated_lufs(x, rate):
    """ITU-R BS.1770-4 のゲーティング付きラウドネス。無音は -inf。"""
    weighted = signal.sosfilt(_k_weighting(rate), x, axis=0)
    energy = np.concatenate(([0.0], np.cumsum((weighted ** 2).sum(axis=1))))
    window = int(round(rate * 0.4))
    hop = int(round(rate * 0.1))
    if len(x) < window:
        blocks = np.array([energy[-1] / max(len(x), 1)])
    else:
        starts = np.arange(0, len(x) - window + 1, hop)
        blocks = (energy[starts + window] - energy[starts]) / window
    loudness = lambda value: -0.691 + 10.0 * np.log10(np.maximum(value, 1e-30))
    blocks = blocks[loudness(blocks) > _MIN_LUFS]
    if blocks.size == 0:
        return float("-inf")
    relative_gate = loudness(blocks.mean()) - 10.0
    gated = blocks[loudness(blocks) > relative_gate]
    return float(loudness(gated.mean()))


def _peak_envelope(x):
    """FFTでオーバーサンプリングして求めた、フレームごとのピーク（全チャンネル共通）。

    区切りの影響が出ないよう、前後に余白をつけた区間ごとに計算する。
    """
    n = len(x)
    envelope = np.abs(x).max(axis=1)

    def scan(start):
        end = min(n, start + _TP_CHUNK)
        a, b = max(0, start - _TP_MARGIN), min(n, end + _TP_MARGIN)
        block = np.pad(x[a:b], ((_TP_MARGIN - (start - a), _TP_MARGIN - (b - end)), (0, 0)))
        spectrum = sp_fft.rfft(block, axis=0)
        upsampled = sp_fft.irfft(spectrum, n=len(block) * _TP_FACTOR, axis=0) * _TP_FACTOR
        part = np.abs(upsampled[_TP_MARGIN * _TP_FACTOR:(_TP_MARGIN + end - start) * _TP_FACTOR])
        peaks = envelope[start:end]
        for channel in range(x.shape[1]):
            for phase in range(_TP_FACTOR):
                np.maximum(peaks, part[phase::_TP_FACTOR, channel], out=peaks)

    # 各スレッドは envelope の重ならない範囲だけを書き換える
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        list(pool.map(scan, range(0, n, _TP_CHUNK)))
    return envelope


def true_peak_db(x):
    return float(_lin_to_db(_peak_envelope(x).max()))


def waveform_peaks(x, bins=1200):
    """画面の波形表示用に、区間ごとのピークへ間引く。"""
    linked = np.abs(x).max(axis=1)
    edges = np.linspace(0, len(linked), min(bins, len(linked)) + 1).astype(int)[:-1]
    return np.round(np.maximum.reduceat(linked, edges), 4).tolist()


def probe(path):
    """読み込み時の解析。無音はここで受け付けない。"""
    x, rate = read_audio(path)
    lufs = integrated_lufs(x, rate)
    if not np.isfinite(lufs):
        raise AudioError("無音のファイルは処理できません。")
    return {
        "duration": len(x) / rate,
        "rate": rate,
        "channels": x.shape[1],
        "subtype": _source_subtype(path),
        "lufs": lufs,
        "tp": true_peak_db(x),
        "waveform": waveform_peaks(x),
    }


# ---------------------------------------------------------------- 各処理

def high_pass(x, rate, freq):
    sos = signal.butter(4, min(freq, rate * 0.45), "high", fs=rate, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def _biquad_peak(freq, q, gain_db, rate):
    a = 10.0 ** (gain_db / 40.0)
    w = 2.0 * np.pi * freq / rate
    alpha = np.sin(w) / (2.0 * q)
    a0 = 1.0 + alpha / a
    return [
        (1.0 + alpha * a) / a0, -2.0 * np.cos(w) / a0, (1.0 - alpha * a) / a0,
        1.0, -2.0 * np.cos(w) / a0, (1.0 - alpha / a) / a0,
    ]


def _biquad_shelf(freq, gain_db, rate, high):
    a = 10.0 ** (gain_db / 40.0)
    w = 2.0 * np.pi * freq / rate
    cos_w = np.cos(w)
    beta = 2.0 * np.sqrt(a) * np.sin(w) / (2.0 * 0.707)
    sign = -1.0 if high else 1.0
    a0 = (a + 1.0) + sign * (a - 1.0) * cos_w + beta
    return [
        a * ((a + 1.0) - sign * (a - 1.0) * cos_w + beta) / a0,
        2.0 * sign * a * ((a - 1.0) - sign * (a + 1.0) * cos_w) / a0,
        a * ((a + 1.0) - sign * (a - 1.0) * cos_w - beta) / a0,
        1.0,
        -2.0 * sign * ((a - 1.0) + sign * (a + 1.0) * cos_w) / a0,
        ((a + 1.0) + sign * (a - 1.0) * cos_w - beta) / a0,
    ]


def tone_shape(x, rate, low_db=0.0, mid_db=0.0, high_db=0.0, mud_cut_db=-1.0):
    """低域シェルフ、こもり除去、中域ピーク、高域シェルフ。"""
    sections = []
    if low_db:
        sections.append(_biquad_shelf(100.0, low_db, rate, high=False))
    if mud_cut_db:
        sections.append(_biquad_peak(250.0, 1.0, mud_cut_db, rate))
    if mid_db and 1500.0 < rate * 0.45:
        sections.append(_biquad_peak(1500.0, 0.8, mid_db, rate))
    if high_db and 8000.0 < rate * 0.45:
        sections.append(_biquad_shelf(8000.0, high_db, rate, high=True))
    return signal.sosfilt(np.array(sections), x, axis=0) if sections else x


def _reduction_curve(level_db, threshold_db, ratio, knee_db):
    over = level_db - threshold_db
    half = knee_db / 2.0
    knee = (over + half) ** 2 / (2.0 * knee_db)
    amount = np.where(over <= -half, 0.0, np.where(over >= half, over, knee))
    return amount * (1.0 - 1.0 / ratio)


def glue_compress(x, rate, amount_db, ratio=2.0, attack=0.03, release=0.2, knee_db=6.0):
    """曲のまとまりを出すコンプ。

    amount_db は音が鳴っている区間の平均リダクション量で、これに合うしきい値を自動で求める。
    制御信号は1ms単位で計算する。戻り値は (処理後の音, 実際の平均リダクション量)。
    """
    step = max(1, int(round(rate * 0.001)))
    blocks = len(x) // step
    if amount_db <= 0.0 or blocks < 2:
        return x, 0.0
    # 低音でポンピングしないよう、検出側だけ低域を除く
    sidechain = signal.sosfilt(signal.butter(2, 90.0, "high", fs=rate, output="sos"), x, axis=0)
    power = (sidechain[:blocks * step] ** 2).reshape(blocks, step, x.shape[1]).mean(axis=1).max(axis=1)
    level = 10.0 * np.log10(np.maximum(power, 1e-12))
    rise = np.exp(-0.001 * step / rate * 1000.0 / attack)
    fall = np.exp(-0.001 * step / rate * 1000.0 / release)
    envelope = np.empty(blocks)
    state = float(level[0])
    for index, value in enumerate(level.tolist()):
        coefficient = rise if value > state else fall
        state = coefficient * state + (1.0 - coefficient) * value
        envelope[index] = state

    active = envelope > np.percentile(envelope, 95) - 20.0
    high, low = float(envelope.max()) + knee_db, float(envelope.max()) - 80.0
    for _ in range(30):
        threshold = (high + low) / 2.0
        if _reduction_curve(envelope[active], threshold, ratio, knee_db).mean() > amount_db:
            low = threshold
        else:
            high = threshold
    reduction = _reduction_curve(envelope, (high + low) / 2.0, ratio, knee_db)
    centers = (np.arange(blocks) + 0.5) * step
    gain_db = np.interp(np.arange(len(x)), centers, amount_db - reduction)
    return x * _db_to_lin(gain_db)[:, None], float(reduction[active].mean())


def limit(x, rate, ceiling_db=CEILING_DBTP, lookahead=0.005, release=0.1, envelope=None):
    """ルックアヘッド・リミッター。上限を超える箇所がなければ音を変えない。

    envelope に x のピーク包絡を渡すと再計算を省ける。
    戻り値は (処理後の音, 最大リダクション量dB)。
    """
    ceiling = _db_to_lin(ceiling_db - _LIMITER_MARGIN_DB)
    if envelope is None:
        envelope = _peak_envelope(x)
    if envelope.max() <= ceiling:
        return x, 0.0
    required = np.minimum(1.0, ceiling / np.maximum(envelope, 1e-12))
    width = int(rate * lookahead) | 1
    held = ndimage.minimum_filter1d(required, size=width, mode="nearest")

    # 戻りの計算は32サンプル単位で行う。各節点は前後の区間の最小値なので、
    # 節点の間を直線で結んでも必要な減衰量を下回らない。
    step = 32
    count = int(np.ceil(len(x) / step))
    block_min = np.minimum.reduceat(held, np.arange(count) * step)
    nodes = np.minimum(np.concatenate(([1.0], block_min)), np.concatenate((block_min, [1.0])))
    coefficient = np.exp(-step / (rate * release))
    state = 1.0
    released = np.empty(count + 1)
    for index, value in enumerate(nodes.tolist()):
        state = min(value, coefficient * state + 1.0 - coefficient)
        released[index] = state
    gain = np.interp(np.arange(len(x)), np.arange(count + 1) * step, released)
    gain = ndimage.uniform_filter1d(gain, size=width, mode="nearest")
    return x * gain[:, None], float(-_lin_to_db(gain.min()))


def normalize_loudness(x, rate, target_lufs):
    """目標ラウドネスまで上げ下げし、はみ出すピークはリミッターで抑える。

    リミッターで音量が変わるため、最大3回まで調整をやり直す。
    戻り値は (処理後の音, 最大リダクション量dB, 到達したラウドネス)。
    """
    envelope = _peak_envelope(x)
    gain_db = target_lufs - integrated_lufs(x, rate)
    previous = None
    for attempt in range(4):
        gain_db = min(gain_db, _MAX_GAIN_DB)
        gain = _db_to_lin(gain_db)
        result, reduction = limit(x * gain, rate, envelope=envelope * gain)
        lufs = integrated_lufs(result, rate)
        error = target_lufs - lufs
        if abs(error) < 0.05 or attempt == 3:
            break
        # リミッターが深くかかるほどゲインの割に音量は増えないので、傾きを見て補正する
        slope = 1.0
        if previous is not None and gain_db != previous[0]:
            slope = float(np.clip((lufs - previous[1]) / (gain_db - previous[0]), 0.1, 1.0))
        previous = (gain_db, lufs)
        gain_db += error / slope
    return result, reduction, lufs


# ---------------------------------------------------------------- 全体の流れ

def _output_rate(rate, fmt):
    return CD_RATE if fmt == "cd" else rate


def _summary(path):
    x, rate = read_audio(path)
    return x, {"lufs": integrated_lufs(x, rate), "tp": true_peak_db(x), "rate": rate, "duration": len(x) / rate}


def _finish(y, rate, out_path, fmt, source, glue_db, reduction_db, warnings, target=None):
    write_wav(out_path, y, rate, fmt)
    written, output = _summary(out_path)
    output["bits"] = 16 if fmt == "cd" else 24
    if reduction_db > 6.0:
        warnings.append(f"リミッターが最大 {reduction_db:.1f} dB かかっています。音が潰れていないか確認してください。")
    if target is not None and abs(output["lufs"] - target) > 0.5:
        warnings.append(f"目標 {target:.1f} LUFS に届かず、{output['lufs']:.1f} LUFS になりました。")
    return ProcessResult(
        source=source, output=output, glue_db=glue_db, reduction_db=reduction_db,
        warnings=warnings, waveform=waveform_peaks(written),
    )


def _source_stats(x, rate, known):
    """処理前の測定値。読み込み時に測った値があればそれを使う。"""
    source = known or {"lufs": integrated_lufs(x, rate), "tp": true_peak_db(x)}
    if not np.isfinite(source["lufs"]):
        raise AudioError("無音のファイルは処理できません。")
    return source


def master_file(in_path, out_path, options, progress=lambda fraction, message: None, source=None):
    """機能1: 1曲をマスタリングして out_path に書き出す。"""
    x, rate = read_audio(in_path)
    source = _source_stats(x, rate, source)

    progress(0.05, "読み込みとサンプルレート変換")
    new_rate = _output_rate(rate, options.format)
    y = resample(x, rate, new_rate)
    glue_db = reduction_db = 0.0
    warnings = []

    if options.hpf:
        progress(0.2, "ハイパスフィルター")
        y = high_pass(y, new_rate, options.hpf_freq)
    if options.tone:
        progress(0.3, "トーン整理")
        y = tone_shape(y, new_rate, options.low_db, options.mid_db, options.high_db)
    if options.glue:
        progress(0.4, "グルー・コンプレッション")
        y, glue_db = glue_compress(y, new_rate, options.glue_db)

    if options.release:
        progress(0.55, "リリース用マスタリング")
        y, reduction_db, _ = normalize_loudness(y, new_rate, options.target_lufs)
    else:
        # 音圧調整をしない場合でも、EQなどの増幅で音割れしないようにする
        over = true_peak_db(y) - (CEILING_DBTP - _LIMITER_MARGIN_DB)
        if over > 0.0:
            y = y * _db_to_lin(-over)
            warnings.append(f"音割れを防ぐため、全体を {over:.1f} dB 下げました。")

    progress(0.85, "書き出しと再測定")
    return _finish(
        y, new_rate, out_path, options.format, source, glue_db, reduction_db, warnings,
        target=options.target_lufs if options.release else None,
    )


def align_target(lufs_values, mode, value=None):
    """機能2の基準ラウドネス。mode は 'average' か 'fixed'。"""
    return float(np.mean(lufs_values)) if mode == "average" else float(value)


def align_file(in_path, out_path, target_lufs, fmt="cd", progress=lambda fraction, message: None, source=None):
    """機能2: EQとコンプはかけず、音量だけを揃えて書き出す。"""
    x, rate = read_audio(in_path)
    source = _source_stats(x, rate, source)
    progress(0.1, "サンプルレート変換")
    new_rate = _output_rate(rate, fmt)
    y = resample(x, rate, new_rate)
    progress(0.3, "音量調整")
    y, reduction_db, _ = normalize_loudness(y, new_rate, target_lufs)
    progress(0.8, "書き出しと再測定")
    return _finish(y, new_rate, out_path, fmt, source, 0.0, reduction_db, [], target=target_lufs)


def output_name(original_name):
    """mastering_曲名.wav を作る。Windowsのファイル名に使えない文字は除く。"""
    stem = Path(str(original_name).replace("\\", "/")).stem
    cleaned = "".join(c for c in stem if c not in '<>:"/\\|?*' and ord(c) >= 32).strip(" .")
    return f"mastering_{cleaned or 'track'}.wav"
