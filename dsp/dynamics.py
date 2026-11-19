"""Dynamics primitives for a small offline audio-processing chain.

Conventions used by every function in this module:

* Sample values are linear amplitudes; ``0.0`` is silence and ``1.0`` is
  full scale.  Levels are converted with ``20 * log10(value)`` (dBFS).
* All time constants are in milliseconds (``ms``); sample rates are in Hz
  and are always passed explicitly.
* Unless a stricter tolerance is stated, comparisons use an absolute
  tolerance of ``1e-6``.
* Empty sample lists produce empty result lists.
* Invalid parameters, a non-positive sample rate, or non-finite samples
  raise :class:`DspError`.

The functions are deliberately unimplemented: the exercise is to replace
each stub body with the contract-preserving code.
"""

import cmath
import math


class DspError(ValueError):
    """Raised for invalid DSP parameters, sample rates, or sample data."""


def _is_real_finite(value) -> bool:
    """True for finite real numbers (rejects NaN, infinity, bools, non-numbers)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_samples(samples, name: str = "samples") -> None:
    """Raise DspError unless every sample is a finite real number."""
    for sample in samples:
        if not _is_real_finite(sample):
            raise DspError(f"non-finite value in {name}")


def _check_rate(sample_rate) -> None:
    if not _is_real_finite(sample_rate) or sample_rate <= 0:
        raise DspError("sample_rate must be a positive finite number")


def envelope(
    samples: list[float],
    attack_ms: float,
    release_ms: float,
    sample_rate: float,
) -> list[float]:
    """Return the one-pole peak envelope of ``samples``.

    The detector target is ``abs(sample)``.  Let ``a = 1 / (attack_ms / 1000
    * sample_rate)`` and ``r = 1 / (release_ms / 1000 * sample_rate)``.  The
    one-pole coefficients are ``exp(-a)`` and ``exp(-r)`` respectively.
    The envelope starts at ``0.0``.  For each sample, if the target is
    greater than the current envelope, update with the attack coefficient;
    otherwise update with the release coefficient:

    ``env = coeff * env + (1 - coeff) * target``

    This makes attack and release independent: a short attack with a long
    release rises quickly and decays slowly.  A time constant of ``0`` is
    instantaneous (its coefficient is ``0``); negative time constants are
    invalid.  Time constants are in ms.  The result has the same length as
    ``samples`` and is finite for finite input.  Absolute tolerance is
    ``1e-6``.

    Raises:
        DspError: if ``sample_rate <= 0``, a time constant is negative, or
            a sample is not finite.
    """
    _check_rate(sample_rate)
    if not _is_real_finite(attack_ms) or attack_ms < 0:
        raise DspError("attack_ms must be a non-negative finite number")
    if not _is_real_finite(release_ms) or release_ms < 0:
        raise DspError("release_ms must be a non-negative finite number")
    _check_samples(samples)

    attack_coeff = 0.0 if attack_ms == 0.0 else math.exp(
        -1.0 / (attack_ms / 1000.0 * sample_rate)
    )
    release_coeff = 0.0 if release_ms == 0.0 else math.exp(
        -1.0 / (release_ms / 1000.0 * sample_rate)
    )

    result = []
    env = 0.0
    for sample in samples:
        target = abs(sample)
        if target > env:
            coeff = attack_coeff
        else:
            coeff = release_coeff
        env = coeff * env + (1.0 - coeff) * target
        result.append(env)
    return result


def gain_computer(
    env_db: list[float],
    threshold_db: float,
    ratio: float,
    knee_db: float,
    makeup_db: float,
) -> list[float]:
    """Return the soft-knee compressor gain in dB for each envelope level.

    Inputs and outputs are in dB.  ``knee_db`` is the full knee width in dB
    (not a half-width) and must be non-negative.  For ``x = env_db -
    threshold_db`` and ``k = knee_db`` the curve is:

    * ``x <= -k/2``: ``0`` dB;
    * ``-k/2 < x < k/2``:
      ``-((x + k/2) ** 2) / (2 * k) * (1 - 1 / ratio)`` dB;
    * ``x >= k/2``: ``-x * (1 - 1 / ratio)`` dB.

    With ``knee_db == 0`` the middle branch disappears and the two remaining
    branches meet continuously at the threshold.  ``makeup_db`` is added to
    the gain for ``ratio > 1``.  ``ratio == 1`` is a strict unity bypass:
    every returned gain is ``0.0`` dB and makeup is not applied, so a
    direct signal is never level-changed.  No soft-knee corner may remain.

    Absolute tolerance is ``1e-6``.

    Raises:
        DspError: for ``ratio < 1``, ``knee_db < 0``, non-finite input, or
            non-finite parameters.
    """
    if not _is_real_finite(threshold_db) or not _is_real_finite(ratio):
        raise DspError("threshold_db and ratio must be finite numbers")
    if not _is_real_finite(knee_db) or not _is_real_finite(makeup_db):
        raise DspError("knee_db and makeup_db must be finite numbers")
    if ratio < 1.0:
        raise DspError("ratio must be >= 1")
    if knee_db < 0.0:
        raise DspError("knee_db must be >= 0")
    _check_samples(env_db, "env_db")

    # ratio == 1 is a strict unity bypass: makeup is never applied.
    if ratio == 1.0:
        return [0.0 for _ in env_db]

    half_knee = knee_db / 2.0
    slope = 1.0 - 1.0 / ratio
    result = []
    for level_db in env_db:
        x = level_db - threshold_db
        if knee_db > 0.0 and -half_knee < x < half_knee:
            gain = -(((x + half_knee) ** 2) / (2.0 * knee_db)) * slope
        elif x >= half_knee:
            gain = -x * slope
        else:
            gain = 0.0
        if ratio > 1.0:
            gain += makeup_db
        result.append(gain)
    return result


def compress(
    samples: list[float],
    params: dict,
    sample_rate: float,
) -> list[float]:
    """Compress ``samples`` with the envelope/gain-computer pair.

    ``params`` contains ``threshold_db``, ``ratio``, ``attack_ms``,
    ``release_ms``, optional ``knee_db`` (default ``0.0``), and optional
    ``makeup_db`` (default ``0.0``).  The processing chain is:

    1. compute the peak envelope with the supplied attack/release times;
    2. convert it to dB with ``20 * log10(max(env, 1e-12))``;
    3. compute per-sample gain with :func:`gain_computer`;
    4. multiply the input by the linear gain.

    The output has the same length as the input and contains no NaN for
    finite input.  A ratio of ``1`` with zero makeup is an identity
    operation within ``1e-9``.  Time constants are in ms.  Other
    comparisons use an absolute tolerance of ``1e-6``.

    Raises:
        DspError: for invalid parameters, ``sample_rate <= 0``, or a
            non-finite sample.
    """
    _check_rate(sample_rate)
    _check_samples(samples)

    threshold_db = params["threshold_db"]
    ratio = params["ratio"]
    attack_ms = params["attack_ms"]
    release_ms = params["release_ms"]
    knee_db = params.get("knee_db", 0.0)
    makeup_db = params.get("makeup_db", 0.0)

    # Delegate parameter validation to the primitives.
    env_linear = envelope(samples, attack_ms, release_ms, sample_rate)
    env_db = [20.0 * math.log10(max(value, 1e-12)) for value in env_linear]
    gains_db = gain_computer(env_db, threshold_db, ratio, knee_db, makeup_db)

    output = []
    for sample, gain_db in zip(samples, gains_db):
        linear_gain = 10.0 ** (gain_db / 20.0)
        output.append(sample * linear_gain)
    return output


def apply_limiter(
    samples: list[float],
    ceiling_db: float,
    lookahead_ms: float,
    sample_rate: float,
) -> list[float]:
    """Apply a look-ahead peak limiter.

    ``ceiling_db`` is the maximum output level in dBFS.  Let ``C =
    10 ** (ceiling_db / 20)`` and ``D = round(lookahead_ms / 1000 *
    sample_rate)``.  The signal is delayed by ``D`` samples, and the gain
    at each delayed sample is derived from the maximum absolute input in
    its ``D``-sample look-ahead window (including the current sample), so a
    transient is attenuated *before* it reaches the output.  With ``D == 0``
    the limiter acts instantaneously without a delay.  Every returned
    absolute value satisfies ``abs(y) <= C + 1e-9``; the output length
    equals the input length.

    Time is in ms.  Other comparisons use an absolute tolerance of
    ``1e-6``.

    Raises:
        DspError: for ``lookahead_ms < 0``, ``sample_rate <= 0``,
            non-finite parameters, or a non-finite sample.
    """
    _check_rate(sample_rate)
    if not _is_real_finite(ceiling_db):
        raise DspError("ceiling_db must be a finite number")
    if not _is_real_finite(lookahead_ms) or lookahead_ms < 0:
        raise DspError("lookahead_ms must be a non-negative finite number")
    _check_samples(samples)

    ceiling = 10.0 ** (ceiling_db / 20.0)
    delay = round(lookahead_ms / 1000.0 * sample_rate)
    count = len(samples)
    output = [0.0] * count

    for n in range(count):
        # The sample leaving the delay line, or silence before it starts.
        source_index = n - delay
        if source_index < 0:
            continue
        source = samples[source_index]
        # Maximum absolute input in the D-sample look-ahead window,
        # including the (delayed) current sample.
        window_end = min(source_index + delay, count - 1)
        window_peak = max(
            abs(samples[i]) for i in range(source_index, window_end + 1)
        )
        if window_peak > ceiling:
            gain = ceiling / window_peak
        else:
            gain = 1.0
        output[n] = source * gain
    return output


def smoothed_gain(
    gains: list[float],
    time_ms: float,
    sample_rate: float,
) -> list[float]:
    """Smooth a gain sequence with a one-pole filter.

    ``gains`` are linear gains and ``time_ms`` is the time constant.  Let
    ``coeff = exp(-1 / (time_ms / 1000 * sample_rate))`` (and ``coeff = 0``
    when ``time_ms == 0``).  The first output is exactly ``gains[0]``;
    thereafter:

    ``y[n] = coeff * y[n-1] + (1 - coeff) * gains[n]``

    Therefore a step in ``gains`` produces an exponential approach rather
    than an instantaneous jump, and a larger ``time_ms`` produces a slower
    approach.  Time is in ms.  Absolute tolerance is ``1e-6``.

    Raises:
        DspError: for ``time_ms < 0``, ``sample_rate <= 0``, or a
            non-finite gain.
    """
    _check_rate(sample_rate)
    if not _is_real_finite(time_ms) or time_ms < 0:
        raise DspError("time_ms must be a non-negative finite number")
    _check_samples(gains, "gains")

    if not gains:
        return []

    coeff = 0.0 if time_ms == 0.0 else math.exp(
        -1.0 / (time_ms / 1000.0 * sample_rate)
    )

    result = [gains[0]]
    previous = gains[0]
    for gain in gains[1:]:
        previous = coeff * previous + (1.0 - coeff) * gain
        result.append(previous)
    return result


def biquad_coeffs(
    kind: str,
    freq: float,
    q: float,
    gain_db: float,
    sample_rate: float,
) -> tuple:
    """Return normalized biquad coefficients ``(b0, b1, b2, a1, a2)``.

    ``kind`` is one of ``"peak"``, ``"lowshelf"``, ``"highshelf"``,
    ``"lowpass"``, or ``"highpass"``.  ``freq`` is in Hz, ``q`` is the
    quality factor (or shelf slope control), ``gain_db`` is used by
    ``peak`` and both shelf kinds, and ``sample_rate`` is in Hz.  The RBJ
    audio-EQ-cookbook forms are used.

    The returned values are divided by ``a0``.  In particular, ``a1`` and
    ``a2`` belong to the denominator ``1 + a1 z^-1 + a2 z^-2``; the
    coefficients must not be left un-normalized.  A normalized low-pass
    therefore has unity gain at DC:
    ``(b0 + b1 + b2) / (1 + a1 + a2) == 1`` within ``1e-6``.

    Raises:
        DspError: for an unknown ``kind``, ``freq`` outside
            ``(0, sample_rate / 2)``, ``q <= 0``, or ``sample_rate <= 0``.
    """
    _check_rate(sample_rate)
    if not _is_real_finite(freq) or not (0.0 < freq < sample_rate / 2.0):
        raise DspError("freq must satisfy 0 < freq < sample_rate / 2")
    if not _is_real_finite(q) or q <= 0:
        raise DspError("q must be a positive finite number")
    if not _is_real_finite(gain_db):
        raise DspError("gain_db must be a finite number")

    omega = 2.0 * math.pi * freq / sample_rate
    cos_w = math.cos(omega)
    sin_w = math.sin(omega)
    alpha = sin_w / (2.0 * q)

    if kind == "lowpass":
        b0 = (1.0 - cos_w) / 2.0
        b1 = 1.0 - cos_w
        b2 = (1.0 - cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
    elif kind == "highpass":
        b0 = (1.0 + cos_w) / 2.0
        b1 = -(1.0 + cos_w)
        b2 = (1.0 + cos_w) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha
    elif kind == "peak":
        amplitude = 10.0 ** (gain_db / 40.0)
        b0 = 1.0 + alpha * amplitude
        b1 = -2.0 * cos_w
        b2 = 1.0 - alpha * amplitude
        a0 = 1.0 + alpha / amplitude
        a1 = -2.0 * cos_w
        a2 = 1.0 - alpha / amplitude
    elif kind == "lowshelf":
        amplitude = 10.0 ** (gain_db / 40.0)
        cosh_term = (amplitude + 1.0 / amplitude) * (1.0 / q - 1.0)
        beta = 2.0 * math.sqrt(amplitude) * math.sqrt(
            max(0.0, (cosh_term + 2.0) / 2.0)
        )
        amp_plus = amplitude + 1.0
        amp_minus = amplitude - 1.0
        b0 = amplitude * (amp_plus - amp_minus * cos_w + beta * sin_w)
        b1 = 2.0 * amplitude * (amp_minus - amp_plus * cos_w)
        b2 = amplitude * (amp_plus - amp_minus * cos_w - beta * sin_w)
        a0 = amp_plus + amp_minus * cos_w + beta * sin_w
        a1 = -2.0 * (amp_minus + amp_plus * cos_w)
        a2 = amp_plus + amp_minus * cos_w - beta * sin_w
    elif kind == "highshelf":
        amplitude = 10.0 ** (gain_db / 40.0)
        cosh_term = (amplitude + 1.0 / amplitude) * (1.0 / q - 1.0)
        beta = 2.0 * math.sqrt(amplitude) * math.sqrt(
            max(0.0, (cosh_term + 2.0) / 2.0)
        )
        amp_plus = amplitude + 1.0
        amp_minus = amplitude - 1.0
        b0 = amplitude * (amp_plus + amp_minus * cos_w + beta * sin_w)
        b1 = -2.0 * amplitude * (amp_minus + amp_plus * cos_w)
        b2 = amplitude * (amp_plus + amp_minus * cos_w - beta * sin_w)
        a0 = amp_plus - amp_minus * cos_w + beta * sin_w
        a1 = 2.0 * (amp_minus - amp_plus * cos_w)
        a2 = amp_plus - amp_minus * cos_w - beta * sin_w
    else:
        raise DspError(f"unknown biquad kind: {kind!r}")

    # Normalize every coefficient by a0 so the denominator's constant term is 1.
    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def apply_biquad(samples: list[float], coeffs: tuple) -> list[float]:
    """Filter ``samples`` with a transposed direct-form II biquad.

    ``coeffs`` is ``(b0, b1, b2, a1, a2)`` as returned by
    :func:`biquad_coeffs`.  For each input ``x``::

        y  = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y

    Filter state is kept per coefficient tuple between calls, so splitting
    one signal into consecutive blocks and processing each block with the
    same coefficients yields the same result as processing the whole signal
    in one call; the phase is continuous at block boundaries.  Passing an
    empty ``samples`` list resets the stored state for those coefficients
    (and returns ``[]``).  This reset is the supported way to start an
    independent filter run with reused coefficient values.

    Absolute tolerance is ``1e-6``.

    Raises:
        DspError: for malformed coefficient tuples or non-finite samples.
    """
    # One independent state vector per coefficient tuple, so recomputing
    # identical coefficients per block reuses the same filter instance while
    # different filters (and different channels) never share state.
    if not hasattr(apply_biquad, "_states"):
        apply_biquad._states = {}
    states = apply_biquad._states

    if not isinstance(coeffs, tuple) or len(coeffs) != 5:
        raise DspError("coeffs must be a (b0, b1, b2, a1, a2) tuple")
    if not all(_is_real_finite(value) for value in coeffs):
        raise DspError("coefficients must be finite numbers")
    _check_samples(samples)

    b0, b1, b2, a1, a2 = coeffs

    # An empty block resets the stored state for these coefficients.
    if not samples:
        states.pop(coeffs, None)
        return []

    z1, z2 = states.get(coeffs, (0.0, 0.0))
    result = []
    for x in samples:
        y = b0 * x + z1
        z1 = b1 * x - a1 * y + z2
        z2 = b2 * x - a2 * y
        result.append(y)
    states[coeffs] = (z1, z2)
    return result


def process_batch(
    channels: list[list[float]],
    params: dict,
    sample_rate: float,
) -> dict:
    """Compress every channel independently and return a result dict.

    ``channels`` is a list of sample lists.  Each channel is processed with
    :func:`compress` and the same ``params``/``sample_rate``; no envelope or
    filter state may be shared between channels, so a loud left channel
    cannot alter the gain of a quiet right channel.

    On success the return value is::

        {"channels": [list[float], ...], "sample_rate": float}

    Before producing any output, every sample in every channel is checked
    for finiteness.  If any channel contains NaN or infinity, the whole
    batch is rolled back atomically: :class:`DspError` is raised, no partial
    result is returned, and the input lists are not modified.  A batch must
    never silently skip a bad channel and leave NaN in its output.

    Other comparisons use an absolute tolerance of ``1e-6``.

    Raises:
        DspError: for any non-finite sample, invalid parameters, or
            ``sample_rate <= 0``.
    """
    _check_rate(sample_rate)
    if not isinstance(channels, list):
        raise DspError("channels must be a list of sample lists")

    # Validate every sample of every channel before any processing so the
    # batch is rolled back atomically: no output and untouched inputs.
    for channel in channels:
        if not isinstance(channel, list):
            raise DspError("every channel must be a sample list")
        _check_samples(channel)

    processed = [compress(channel, params, sample_rate) for channel in channels]
    return {"channels": processed, "sample_rate": sample_rate}
