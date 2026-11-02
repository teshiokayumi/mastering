"""合成音で音声処理を確認する。実行: python -m unittest discover -s tests"""
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "app"))
import dsp  # noqa: E402

RATE = 44100


def sine(freq, seconds=2.0, amp=0.25, rate=RATE, channels=2):
    t = np.arange(int(seconds * rate)) / rate
    wave = amp * np.sin(2 * np.pi * freq * t)
    return np.column_stack([wave] * channels)


def music_like(seconds=20.0, amp=0.1, rate=RATE, seed=1):
    """持続音に減衰するノイズの連打を重ねた音。ピークと平均の差が音楽に近い。"""
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    noise = rng.standard_normal((n, 2))
    period = int(rate * 0.5)
    decay = np.exp(-(np.arange(n) % period) / (rate * 0.08))
    tones = sum(np.sin(2 * np.pi * f * t) for f in (82, 220, 330, 880)) * 0.25
    return np.clip(amp * (noise * decay[:, None] + tones[:, None]), -1.0, 1.0)


def rms_db(x):
    # フィルターの立ち上がりを避けて後半だけを測る
    tail = x[len(x) // 2:]
    return 20 * np.log10(np.sqrt(np.mean(tail ** 2)))


class MeasureTest(unittest.TestCase):
    def test_997hz_full_scale_sine_reads_about_minus_3(self):
        # BS.1770: 0 dBFS の 997 Hz サイン波（ステレオ）はおよそ 0 LUFS、片チャンネルでは -3.01 LUFS
        self.assertAlmostEqual(dsp.integrated_lufs(sine(997, 5, 1.0, channels=1), RATE), -3.01, delta=0.1)
        self.assertAlmostEqual(dsp.integrated_lufs(sine(997, 5, 1.0), RATE), 0.0, delta=0.1)

    def test_silence_is_negative_infinity(self):
        self.assertEqual(dsp.integrated_lufs(np.zeros((RATE, 2)), RATE), float("-inf"))

    def test_true_peak_finds_inter_sample_peak(self):
        # fs/4 を45度ずらすと、サンプル値は 0.707 でも実際のピークは 1.0
        t = np.arange(RATE)
        wave = np.sin(np.pi / 2 * t + np.pi / 4)
        x = np.column_stack([wave, wave])
        self.assertLess(np.abs(x).max(), 0.71)
        self.assertAlmostEqual(dsp.true_peak_db(x), 0.0, delta=0.3)

    def test_true_peak_does_not_under_read_near_nyquist(self):
        self.assertGreater(dsp.true_peak_db(sine(20000, 1.0, 1.0)), -0.3)


class FilterTest(unittest.TestCase):
    def test_high_pass_cuts_lows_and_keeps_mids(self):
        self.assertLess(rms_db(dsp.high_pass(sine(10), RATE, 30)) - rms_db(sine(10)), -30)
        self.assertAlmostEqual(rms_db(dsp.high_pass(sine(1000), RATE, 30)), rms_db(sine(1000)), delta=0.1)

    def test_tone_bands_move_their_own_range(self):
        for freq, key in ((40, "low_db"), (1500, "mid_db"), (15000, "high_db")):
            shaped = dsp.tone_shape(sine(freq), RATE, mud_cut_db=0.0, **{key: 6.0})
            self.assertAlmostEqual(rms_db(shaped) - rms_db(sine(freq)), 6.0, delta=0.5, msg=key)
            shaped = dsp.tone_shape(sine(freq), RATE, mud_cut_db=0.0, **{key: -6.0})
            self.assertAlmostEqual(rms_db(shaped) - rms_db(sine(freq)), -6.0, delta=0.5, msg=key)

    def test_mud_cut_lowers_250hz_only(self):
        self.assertAlmostEqual(rms_db(dsp.tone_shape(sine(250), RATE)) - rms_db(sine(250)), -1.0, delta=0.1)
        self.assertAlmostEqual(rms_db(dsp.tone_shape(sine(8000), RATE)) - rms_db(sine(8000)), 0.0, delta=0.1)


class DynamicsTest(unittest.TestCase):
    def test_glue_reaches_requested_reduction(self):
        _, achieved = dsp.glue_compress(music_like(), RATE, 2.0)
        self.assertAlmostEqual(achieved, 2.0, delta=0.1)

    def test_glue_zero_is_bypass(self):
        x = music_like(2)
        y, achieved = dsp.glue_compress(x, RATE, 0.0)
        self.assertIs(y, x)
        self.assertEqual(achieved, 0.0)

    def test_limiter_holds_the_ceiling(self):
        y, reduction = dsp.limit(music_like(amp=0.5) * 4.0, RATE)
        self.assertLessEqual(dsp.true_peak_db(y), dsp.CEILING_DBTP)
        self.assertGreater(reduction, 0.0)

    def test_limiter_leaves_quiet_audio_untouched(self):
        x = sine(1000, amp=0.1)
        y, reduction = dsp.limit(x, RATE)
        self.assertIs(y, x)
        self.assertEqual(reduction, 0.0)

    def test_normalize_reaches_target(self):
        for target in (-14.0, -12.0, -10.0):
            y, _, lufs = dsp.normalize_loudness(music_like(), RATE, target)
            self.assertAlmostEqual(lufs, target, delta=0.1)
            self.assertLessEqual(dsp.true_peak_db(y), dsp.CEILING_DBTP)


class FileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, x, rate=RATE, **kwargs):
        path = self.dir / name
        sf.write(str(path), x, rate, **kwargs)
        return path

    def test_cd_output_is_44k_16bit_within_limits(self):
        source = self.write("in.wav", music_like(rate=48000), 48000, subtype="PCM_24")
        out = self.dir / "out.wav"
        result = dsp.master_file(source, out, dsp.MasterOptions())
        info = sf.info(str(out))
        self.assertEqual((info.samplerate, info.subtype, info.channels), (44100, "PCM_16", 2))
        self.assertAlmostEqual(result.output["lufs"], -12.0, delta=0.2)
        self.assertLessEqual(result.output["tp"], dsp.CEILING_DBTP)
        self.assertAlmostEqual(info.duration, 20.0, delta=0.01)

    def test_hires_output_keeps_rate_and_is_24bit(self):
        source = self.write("in.wav", music_like(rate=48000), 48000, subtype="PCM_24")
        out = self.dir / "out.wav"
        dsp.master_file(source, out, dsp.MasterOptions(format="hires"))
        info = sf.info(str(out))
        self.assertEqual((info.samplerate, info.subtype), (48000, "PCM_24"))

    def test_24bit_writer_keeps_level(self):
        x = sine(1000, amp=0.5)
        out = self.dir / "out.wav"
        dsp.write_wav(out, x, RATE, "hires")
        back, _ = sf.read(str(out), dtype="float64", always_2d=True)
        self.assertLess(np.abs(back - x).max(), 1e-5)

    def test_every_check_off_still_prevents_clipping(self):
        source = self.write("in.wav", music_like(amp=0.5), subtype="FLOAT")
        out = self.dir / "out.wav"
        options = dsp.MasterOptions(hpf=False, tone=False, glue=False, release=False)
        result = dsp.master_file(source, out, options)
        self.assertLessEqual(result.output["tp"], dsp.CEILING_DBTP)

    def test_aiff_and_mono_are_supported(self):
        source = self.write("in.aiff", music_like()[:, 0], format="AIFF", subtype="PCM_24")
        out = self.dir / "out.wav"
        dsp.master_file(source, out, dsp.MasterOptions())
        self.assertEqual(sf.info(str(out)).channels, 1)

    def test_silence_and_broken_files_are_rejected(self):
        with self.assertRaises(dsp.AudioError):
            dsp.probe(self.write("silent.wav", np.zeros((RATE, 2))))
        broken = self.dir / "broken.wav"
        broken.write_bytes(b"not audio at all")
        with self.assertRaises(dsp.AudioError):
            dsp.probe(broken)

    def test_six_tracks_align_and_sources_stay_identical(self):
        paths = [
            self.write(f"t{i}.wav", music_like(amp=amp, seed=i), subtype="PCM_24")
            for i, amp in enumerate((0.02, 0.05, 0.08, 0.12, 0.2, 0.3))
        ]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        target = dsp.align_target([dsp.probe(p)["lufs"] for p in paths], "average")
        loudness = []
        for i, path in enumerate(paths):
            result = dsp.align_file(path, self.dir / f"o{i}.wav", target)
            self.assertLessEqual(result.output["tp"], dsp.CEILING_DBTP)
            loudness.append(result.output["lufs"])
        for value in loudness:
            self.assertAlmostEqual(value, target, delta=0.5)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_output_name(self):
        self.assertEqual(dsp.output_name("夏の歌.aiff"), "mastering_夏の歌.wav")
        self.assertEqual(dsp.output_name('C:\\a\\b?c:d.wav'), "mastering_bcd.wav")
        self.assertEqual(dsp.output_name("....wav"), "mastering_track.wav")


if __name__ == "__main__":
    unittest.main()
