import numpy as np
from cilia.datastructures.DataName import TANGENT_ANGLE, CONDITION_DESCRIPTION, MEAN_SHAPE
from cilia.transformers.Transformer import data_transformer


@data_transformer(
    inputs=(TANGENT_ANGLE,),
    outputs=(TANGENT_ANGLE,),
    algorithm="mean_gauge"
)
def mean_gauge_segments_t(segments):
    """
    Apply spatial mean gauge to each segment.
    segments: list of arrays (T_segment, Ns)
    """
    gauged_segments = []

    for seg in segments:
        psi = np.asarray(seg)
        psi_g = psi - psi.mean(axis=1, keepdims=True)
        gauged_segments.append(psi_g)

    return {
        TANGENT_ANGLE.key: np.array(gauged_segments, dtype=object)
    } 

@data_transformer(
    inputs=(TANGENT_ANGLE,),
    outputs=(TANGENT_ANGLE,),
    algorithm="base_gauge"
)
def base_gauge_segments_t(segments):
    """
    Apply spatial base gauge to each segment.
    segments: list of arrays (T_segment, Ns)
    """
    gauged_segments = []

    for seg in segments:
        psi = np.asarray(seg)
        psi_mg = psi - psi.mean(axis=1, keepdims=True)
        base_signal = psi_mg[:, 0]
        T = len(base_signal)
        if T < 100:
            gauged_segments.append(psi_mg-base_signal[:,None])
            continue

        ft = np.fft.rfft(base_signal)
        freqs = np.fft.rfftfreq(T)
        power = np.abs(ft) ** 2

        # Ignore zero frequency when finding peak
        peak_idx = np.argmax(power[1:]) + 1
        peak_freq = freqs[peak_idx]

        # keep frequencies up to 5x peak frequency
        cutoff = 5 * peak_freq
        mask = freqs <= cutoff
        ft_filtered = ft * mask

        base_clean = np.fft.irfft(ft_filtered, n=T)
        psi_bg = psi_mg - base_clean[:, None]
        gauged_segments.append(psi_bg)

    return {
        TANGENT_ANGLE.key: np.array(gauged_segments, dtype=object)
    }

@data_transformer(
    inputs=(TANGENT_ANGLE,),
    outputs=(TANGENT_ANGLE, MEAN_SHAPE),
    algorithm="base_gauge_sym_v2"
)
def base_gauge_sym_segments_t(segments):
    """
    Base gauge + subtract global mean shape.

    Steps:
    1) mean gauge (per segment)
    2) base gauge (per segment)
    3) compute global mean shape across all segments (time-averaged)
    4) subtract that shape from every segment
    """
    gauged_segments = []

    # ---------------------------
    # First pass: base gauge
    # ---------------------------
    for seg in segments:
        psi = np.asarray(seg)
        # import matplotlib.pyplot as plt
        # plt.imshow(psi.T, aspect="auto")
        # plt.show()
        psi_mg = psi - psi.mean(axis=1, keepdims=True)

        base_signal = psi_mg[:, 0]
        T = len(base_signal)

        if T < 100:
            psi_bg = psi_mg - base_signal[:, None]
            gauged_segments.append(psi_bg)
            continue

        ft = np.fft.rfft(base_signal)
        freqs = np.fft.rfftfreq(T)
        power = np.abs(ft) ** 2

        peak_idx = np.argmax(power[1:]) + 1
        peak_freq = freqs[peak_idx]

        cutoff = 5 * peak_freq
        mask = freqs <= cutoff
        ft_filtered = ft * mask

        base_clean = np.fft.irfft(ft_filtered, n=T)
        psi_bg = psi_mg - base_clean[:, None]

        
        gauged_segments.append(psi_bg)

    # ---------------------------
    # Second pass: compute global mean shape
    # ---------------------------
    shapes = []
    for psi_bg in gauged_segments:
        if psi_bg.ndim != 2 or psi_bg.shape[0] == 0:
            continue
        # temporal mean → shape (Ns,)
        shapes.append(np.mean(psi_bg, axis=0))

    if len(shapes) == 0:
        return {
            TANGENT_ANGLE.key: np.array(gauged_segments, dtype=object),
            MEAN_SHAPE.key: np.array([], dtype=float),
        }

    shapes = np.array(shapes)
    mean_shape = np.mean(shapes, axis=0)  # (Ns,)
    if mean_shape[-1] < 0:
        gauged_segments = [-psi_bg for psi_bg in gauged_segments]
        mean_shape = -mean_shape

    # ---------------------------
    # Third pass: subtract mean shape
    # ---------------------------
    final_segments = []
    for psi_bg in gauged_segments:
        psi_sym = psi_bg - mean_shape[None, :]
        # import matplotlib.pyplot as plt
        # plt.imshow(psi_sym.T, aspect="auto")
        # plt.show()

        final_segments.append(psi_sym)

    return {
        TANGENT_ANGLE.key: np.array(final_segments, dtype=object),
        MEAN_SHAPE.key: np.array(mean_shape, dtype=float),
    }

@data_transformer(
    inputs=(TANGENT_ANGLE,),
    outputs=(TANGENT_ANGLE, MEAN_SHAPE),
    algorithm="base_gauge_sym_flip_v2"
)
def base_gauge_sym_flip_t(segments):
    """
    Mean gauge, orient segments consistently, base gauge, subtract global mean shape.

    Short segments T < 100 are skipped.

    Orientation convention:
    - mean curvature should be positive, estimated by comparing mean shape at s=0.2 and s=0.7
    - travelling wave should go base -> tip, estimated from phase shift between s=0.4 and s=0.6
    """
    import matplotlib.pyplot as plt

    def debug_kymo(arr, title):
        plt.imshow(arr.T, aspect="auto")
        plt.title(title)
        plt.colorbar()
        plt.show()

    def dominant_phase(signal, peak_idx, band_half_width):
        ft = np.fft.rfft(signal)
        lo = max(1, peak_idx - band_half_width)
        hi = min(len(ft), peak_idx + band_half_width + 1)

        weights = np.abs(ft[lo:hi]) ** 2
        if np.sum(weights) == 0:
            return np.angle(ft[peak_idx])

        z = np.sum(ft[lo:hi] * weights)
        return np.angle(z)

    def wave_direction_is_base_to_tip(psi_mg):
        T, Ns = psi_mg.shape

        i1 = int(0.4 * (Ns - 1))
        i2 = int(0.6 * (Ns - 1))

        sig1 = psi_mg[:, i1]
        sig2 = psi_mg[:, i2]

        ft1 = np.fft.rfft(sig1)
        power = np.abs(ft1) ** 2

        if len(power) <= 1 or np.all(power[1:] == 0):
            return True

        peak_idx = np.argmax(power[1:]) + 1
        band_half_width = max(1, int(round(0.1 * peak_idx)))

        phase1 = dominant_phase(sig1, peak_idx, band_half_width)
        phase2 = dominant_phase(sig2, peak_idx, band_half_width)

        dphi = np.angle(np.exp(1j * (phase2 - phase1)))

        # With numpy FFT convention, a wave sin(omega t - k s)
        # has phase decreasing with s, hence dphi < 0 for base -> tip.
        return dphi < 0

    oriented_segments = []

    for seg in segments:
        psi = np.asarray(seg)
        #print(psi.shape)
        if psi.ndim != 2 or psi.shape[0] < 70:
            continue

        psi_mg = psi - psi.mean(axis=1, keepdims=True)
        # debug_kymo(psi_mg, "mean gauge")

        T, Ns = psi_mg.shape

        i_left = int(0.2 * (Ns - 1))
        i_right = int(0.7 * (Ns - 1))

        mean_shape = np.mean(psi_mg, axis=0)
        curvature_wrong = (mean_shape[i_right] - mean_shape[i_left]) < 0

        direction_forward = wave_direction_is_base_to_tip(psi_mg)
        
        psi_aligned = psi_mg.copy()

        if not direction_forward:
            psi_aligned = psi_aligned[:, ::-1]
        # debug_kymo(psi_aligned, "after direction")

        if curvature_wrong ^  (not direction_forward):
            psi_aligned = -psi_aligned

        # direction_after_sign = direction_forward if not curvature_wrong else not direction_forward

        # if not direction_after_sign:
        #     psi_aligned = psi_aligned[:, ::-1]

        # debug_kymo(psi_aligned, "after alignment")

        base_signal = psi_aligned[:, 0]
        ft = np.fft.rfft(base_signal)
        freqs = np.fft.rfftfreq(T)
        power = np.abs(ft) ** 2

        if len(power) <= 10 or np.all(power[1:] == 0):
            base_clean = base_signal
        else:
            peak_idx = np.argmax(power[4:]) + 1
            peak_freq = freqs[peak_idx]
            mask = freqs <= 5 * peak_freq
            base_clean = np.fft.irfft(ft * mask, n=T)
            # import matplotlib.pyplot as plt
            # print("peak_idx:", peak_idx, "peak_freq:", peak_freq, "cutoff:", 5 * peak_freq)
            # plt.figure()
            # plt.plot(freqs, power, label="power")
            # plt.axvline(freqs[peak_idx], linestyle="--", label="peak")
            # plt.axvspan(freqs[0], 5 * peak_freq, alpha=0.2, label="masked")
            # plt.legend()
            # plt.show()
            # plt.figure()
            # plt.plot(base_signal, label="base_signal")
            # plt.plot(base_clean, label="base_clean")
            # plt.legend()
            # plt.show()

        psi_bg = psi_aligned - base_clean[:, None]
        # debug_kymo(psi_bg, "base gauge")

        oriented_segments.append(psi_bg)

    if len(oriented_segments) == 0:
        return {
            TANGENT_ANGLE.key: [np.full((1, 1), np.nan, dtype=float)],
            MEAN_SHAPE.key: np.array([np.nan], dtype=float),
        }

    mean_shape_global = np.mean(
        [np.mean(seg, axis=0) for seg in oriented_segments],
        axis=0
    )

    final_segments = []

    for psi_bg in oriented_segments:
        psi_sym = psi_bg - mean_shape_global[None, :]
        # debug_kymo(psi_sym, "after global mean-shape subtraction")
        final_segments.append(psi_sym)

    return {
        TANGENT_ANGLE.key: np.array(final_segments, dtype=object),
        MEAN_SHAPE.key: np.array(mean_shape_global, dtype=float),
    }




@data_transformer(
    inputs=(TANGENT_ANGLE,),
    outputs=(MEAN_SHAPE,),
    algorithm="mean_shape"
)
def mean_shape_segments_t(segments):

    mean_shape_segments =[]

    for seg in segments:
        psi = np.asarray(seg)
       
        psi_mg = psi - psi.mean(axis=1, keepdims=True)
        psi_mg_avg = psi_mg.mean(axis=0) 
       
        
        mean_shape_segments.append(psi_mg_avg-psi_mg_avg[[0]])



    shapes = np.array(mean_shape_segments)
    mean_shape = np.mean(shapes, axis=0)  # (Ns,)
    # import matplotlib.pyplot as plt
    # plt.plot(mean_shape)
    # plt.show()
    # print(mean_shape)
    # raise

    return {
        MEAN_SHAPE.key: np.array(mean_shape, dtype=float)
    }
