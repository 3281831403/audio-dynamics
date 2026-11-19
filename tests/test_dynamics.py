"""Red tests for the audio dynamics exercise.

Every test calls at least one public function from ``dsp.dynamics``.  The
stubs raise ``NotImplementedError``, so all 15 tests fail at collection-free
runtime until the module is implemented.
"""

import math

import pytest

from dsp.dynamics import (
    DspError,
    apply_biquad,
    apply_limiter,
    biquad_coeffs,
    compress,
    envelope,
    gain_computer,
    process_batch,
    smoothed_gain,
)

PARAMS = {
    "threshold_db": -20.0,
    "ratio": 4.0,
    "knee_db": 6.0,
    "attack_ms": 5.0,
    "release_ms": 50.0,
    "makeup_db": 0.0,
}


def _sine(freq, sample_rate, count):
    return [
        math.sin(2.0 * math.pi * freq * n / sample_rate) for n in range(count)
    ]


def test_defect_01_envelope_uses_distinct_attack_and_release():
    """A fast attack and slow release must not collapse to one coefficient."""
    samples = [1.0] + [0.0] * 100
    env = envelope(samples, attack_ms=0.1, release_ms=100.0, sample_rate=1000.0)

    assert len(env) == len(samples)
    assert env[0] == pytest.approx(1.0 - math.exp(-10.0), abs=1e-6)
    assert env[50] > 0.5


def test_defect_02_soft_knee_rounds_the_threshold_corner():
    """At the threshold the knee must already be compressing, not at 0 dB."""
    gain = gain_computer(
        [-20.0],
        threshold_db=-20.0,
        ratio=4.0,
        knee_db=12.0,
        makeup_db=0.0,
    )[0]

    assert gain == pytest.approx(-1.125, abs=1e-6)


def test_defect_03_unity_ratio_ignores_makeup_gain():
    """ratio=1 is a strict bypass; makeup must not change the level."""
    gains = gain_computer(
        [-30.0, -20.0, -10.0, 0.0],
        threshold_db=-20.0,
        ratio=1.0,
        knee_db=12.0,
        makeup_db=6.0,
    )

    assert gains == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_defect_04_limiter_looks_ahead_and_keeps_the_ceiling():
    """An impulse must be attenuated before it reaches the output."""
    samples = [0.0] * 5 + [1.0] + [0.0] * 5
    limited = apply_limiter(
        samples,
        ceiling_db=-6.0,
        lookahead_ms=2.0,
        sample_rate=1000.0,
    )

    ceiling = 10.0 ** (-6.0 / 20.0)
    peak = max(abs(value) for value in limited)
    peak_index = max(range(len(limited)), key=lambda i: abs(limited[i]))

    assert len(limited) == len(samples)
    assert peak <= ceiling + 1e-6
    assert peak > 0.3
    assert peak_index > 5


def test_defect_05_gain_smoothing_follows_the_requested_time():
    """A fixed smoothing coefficient cannot satisfy both time constants."""
    gains = [0.0] + [1.0] * 200
    fast = smoothed_gain(gains, time_ms=1.0, sample_rate=1000.0)
    slow = smoothed_gain(gains, time_ms=100.0, sample_rate=1000.0)

    assert fast[10] > 0.99
    assert slow[10] < 0.2


def test_defect_06_lowpass_coeffs_are_normalized_by_a0():
    """A low-pass must have unity gain at DC after coefficient normalization."""
    b0, b1, b2, a1, a2 = biquad_coeffs(
        "lowpass", 1000.0, 0.707, 0.0, 48000.0
    )
    dc_gain = (b0 + b1 + b2) / (1.0 + a1 + a2)

    assert dc_gain == pytest.approx(1.0, abs=1e-6)


def test_defect_07_recomputed_coeffs_preserve_block_phase():
    """Recomputing identical coefficients per block must not reset phase."""
    sample_rate = 48000.0
    signal = _sine(440.0, sample_rate, 128)
    c0 = biquad_coeffs("lowpass", 1800.0, 0.707, 0.0, sample_rate)

    apply_biquad([], c0)  # reset the filter instance
    whole = apply_biquad(signal, c0)

    apply_biquad([], c0)  # reset again for the block-by-block run
    blocked = []
    for start in range(0, len(signal), 16):
        current = biquad_coeffs("lowpass", 1800.0, 0.707, 0.0, sample_rate)
        assert current == c0
        blocked.extend(apply_biquad(signal[start : start + 16], current))

    assert blocked == pytest.approx(whole, abs=1e-6)


def test_defect_08_biquad_state_survives_a_block_boundary():
    """Splitting a signal must not create a transient at the block edge."""
    sample_rate = 48000.0
    signal = _sine(523.25, sample_rate, 96)
    coeffs = biquad_coeffs("highpass", 3000.0, 0.8, 0.0, sample_rate)

    apply_biquad([], coeffs)
    whole = apply_biquad(signal, coeffs)

    apply_biquad([], coeffs)
    split = 37
    blocked = apply_biquad(signal[:split], coeffs) + apply_biquad(
        signal[split:], coeffs
    )

    assert blocked == pytest.approx(whole, abs=1e-6)


def test_defect_09_batch_channels_do_not_share_compressor_state():
    """A loud channel must not change the gain applied to a quiet channel."""
    sample_rate = 8000.0
    loud = [0.8] * 64 + [0.0] * 64
    quiet = [0.01] * 64 + [0.0] * 64

    loud_solo = compress(loud, PARAMS, sample_rate)
    quiet_solo = compress(quiet, PARAMS, sample_rate)
    result = process_batch([loud, quiet], PARAMS, sample_rate)

    assert isinstance(result, dict)
    assert result["channels"][0] == pytest.approx(loud_solo, abs=1e-6)
    assert result["channels"][1] == pytest.approx(quiet_solo, abs=1e-6)


def test_defect_10_batch_non_finite_channel_rolls_back_atomically():
    """One NaN channel must raise instead of yielding a partial NaN output."""
    good = [0.1, -0.2, 0.3]
    bad = [0.0, float("nan"), 0.0]
    good_before = list(good)
    bad_before = list(bad)

    with pytest.raises(DspError):
        process_batch([good, bad], PARAMS, 48000.0)

    assert good == good_before
    assert bad[0] == bad_before[0]
    assert math.isnan(bad[1])
    assert bad[2] == bad_before[2]


def test_unity_ratio_and_zero_makeup_are_identity():
    """The full compressor chain must be transparent at ratio=1, makeup=0."""
    samples = [0.0, 0.1, -0.25, 0.5, -0.75, 0.0]
    params = dict(PARAMS, ratio=1.0, makeup_db=0.0)
    output = compress(samples, params, 48000.0)

    assert output == pytest.approx(samples, abs=1e-9)


def test_zero_knee_curve_is_continuous_at_threshold():
    """With no knee, the gain curve must not jump at the threshold."""
    gains = gain_computer(
        [-20.000001, -20.0, -19.999999],
        threshold_db=-20.0,
        ratio=4.0,
        knee_db=0.0,
        makeup_db=0.0,
    )

    assert max(gains) - min(gains) <= 1e-6


def test_empty_input_returns_empty_output():
    """The envelope of an empty signal is an empty list."""
    result = envelope([], attack_ms=1.0, release_ms=10.0, sample_rate=48000.0)
    assert result == []


def test_non_positive_sample_rate_raises_dsp_error():
    """Zero and negative sample rates are invalid for every time-based API."""
    for bad_rate in (0.0, -48000.0):
        with pytest.raises(DspError):
            envelope([0.5], attack_ms=1.0, release_ms=10.0, sample_rate=bad_rate)


def test_unknown_biquad_kind_raises_dsp_error():
    """Only the five documented filter kinds are accepted."""
    with pytest.raises(DspError):
        biquad_coeffs("bandpass", 1000.0, 0.707, 0.0, 48000.0)
