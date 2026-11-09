#!/usr/bin/env python3
"""Compare experimental and simulated fundamental shear-angle modes.

Datasets selected by default
----------------------------
* Cass: structured/cass_2d_phasespace, Nmotor = infinity (encoded as -1),
  with mu_a = 1570 when that field is available.
* 2D: structured/cass_2d_extraction, largest finite Nmotor.
* 3D: structured/cass_3d_extraction, largest finite Nmotor.
* Experiment: structured/sharma, WT, ATP = 750 uM, KCl = 0 mM.

For a phase-conditioned waveform gamma(theta, s), the fundamental complex
Fourier coefficient is defined with the Cass/literature normalization

    gamma_tilde(s) = (1/N) sum_j gamma(theta_j, s) exp(-i theta_j),

so the reconstructed fundamental contribution is

    gamma_1(theta, s) = gamma_tilde(s) exp(i theta) + c.c.

Experimental realizations are first aligned to one another under global phase
rotations. Each model realization is phase-aligned to the experimental reference
so stochastic time origins do not cancel during averaging. Then, for each of the
three model ensemble means, one complex least-squares factor c = rho exp(i delta)
is applied. This jointly optimizes the model's global phase and amplitude scale
against the experimental ensemble mean under the Cass R^2 definition. The
experimental mode is never rescaled.

A single common display rotation is finally applied to experiment and models,
exactly following reproduce_WT_fundamental_mode.py: the point of maximum
experimental amplitude is placed on the negative real axis. This display-only
rotation leaves all Cass R^2 scores unchanged.

Two independently processed figures are saved with the same 2x2 layout:
Re(gamma_tilde), a(s), Im(gamma_tilde), and Phi(s). The first uses the stored
base gauge, while the second uses the mean gauge

    gamma(s,t) -> gamma(s,t) - <gamma(s,t)>_s.

The phase is base-side referenced and a dotted linear fit is shown. Cass-style
complex R^2 values and the fitted model phase/scale adjustments are printed and
saved to separate CSV files for each gauge, but the adjustments are not
annotated in the figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


DATASET_COLORS = {
    "Cass": "#808080",
    "2D": "#1f77b4",
    "3D": "#369b36",
    "Experiment": "#d62728",
}
EXPERIMENT_LABEL = "Experiment"


def _font_rc(base_size: float) -> dict[str, float]:
    """Matplotlib font settings shared by the main and debug figures."""
    return {
        "font.size": base_size,
        "axes.labelsize": base_size + 2,
        "xtick.labelsize": base_size,
        "ytick.labelsize": base_size,
        "legend.fontsize": max(base_size - 2, 8),
        "legend.title_fontsize": max(base_size - 1, 9),
    }


def _import_cilia(cilia_root: Path | None) -> None:
    if cilia_root is not None:
        root = cilia_root.expanduser().resolve()
        # Accept either the parent containing cilia/ or cilia/ itself.
        parent = root.parent if root.name == "cilia" else root
        sys.path.insert(0, str(parent))


@dataclass
class ModeRecord:
    experiment_id: str
    condition: dict[str, Any]
    s: np.ndarray
    z: np.ndarray


@dataclass
class ModeGroup:
    label: str
    records: list[ModeRecord]
    selected_nmotor: float | None = None


class ComparisonError(RuntimeError):
    pass


def _as_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_float(value: Any) -> float:
    value = _as_scalar(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _condition_value(condition: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in condition:
            return _as_scalar(condition[key])
    return None


def _resolve(dataset: Any, realization: Any, item: Any) -> Any:
    exp_path = os.path.join(dataset.path, realization.experiment_id)
    return item.resolve(dataset, exp_path)


def _unwrap_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        nested = value.get("metadata")
        if isinstance(nested, dict):
            return nested
        return value
    return {}


def _newest_item(dataset: Any, realization: Any, items: Sequence[Any]) -> Any | None:
    if not items:
        return None

    exp_path = Path(dataset.path) / realization.experiment_id

    def mtime(item: Any) -> float:
        path = exp_path / f"{item.id}.json"
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return max(items, key=mtime)


def _read_condition(dataset: Any, realization: Any, condition_key: str) -> dict[str, Any]:
    items = [
        item
        for item in realization.data_items.values()
        if item.data_name.key == condition_key
    ]
    item = _newest_item(dataset, realization, items)
    if item is None:
        return {}
    try:
        return _unwrap_dict(_resolve(dataset, realization, item))
    except Exception as exc:
        print(f"WARNING: cannot read condition for {realization.experiment_id}: {exc}")
        return {}


def _read_arclength(dataset: Any, realization: Any, arclength_key: str, ns: int) -> np.ndarray:
    items = [
        item
        for item in realization.data_items.values()
        if item.data_name.key == arclength_key
    ]
    item = _newest_item(dataset, realization, items)
    if item is None:
        return np.linspace(0.0, 1.0, ns)

    try:
        value = _resolve(dataset, realization, item)
        if isinstance(value, dict):
            # Defensive support for old default-loader wrappers.
            arrays = [v for v in value.values() if isinstance(v, np.ndarray)]
            value = arrays[0] if arrays else value.get("data", value)
        s = np.asarray(value, dtype=float).squeeze()
    except Exception as exc:
        print(f"WARNING: cannot read arclength for {realization.experiment_id}: {exc}")
        return np.linspace(0.0, 1.0, ns)

    if s.ndim != 1 or s.size != ns or not np.all(np.isfinite(s)):
        print(
            f"WARNING: invalid arclength for {realization.experiment_id}; "
            "using a normalized uniform grid"
        )
        return np.linspace(0.0, 1.0, ns)

    span = s[-1] - s[0]
    if not np.isfinite(span) or abs(span) < 1e-15:
        return np.linspace(0.0, 1.0, ns)
    return (s - s[0]) / span


def _first_harmonic(cycle: np.ndarray) -> np.ndarray:
    cycle = np.asarray(cycle, dtype=float)
    if cycle.ndim != 2:
        raise ValueError(f"periodic average must have shape (Nphase, Ns), got {cycle.shape}")
    n_phase = cycle.shape[0]
    if n_phase < 4:
        raise ValueError("periodic average has fewer than four phase samples")
    theta = 2.0 * np.pi * np.arange(n_phase, dtype=float) / n_phase
    return (1.0 / n_phase) * np.sum(cycle * np.exp(-1j * theta)[:, None], axis=0)


def _read_mode(
    dataset: Any,
    realization: Any,
    periodic_key: str,
    arclength_key: str,
    periodic_algorithm_prefix: str,
) -> ModeRecord | None:
    candidates = [
        item
        for item in realization.data_items.values()
        if item.data_name.key == periodic_key
        and str(item.algorithm).startswith(periodic_algorithm_prefix)
    ]
    item = _newest_item(dataset, realization, candidates)
    if item is None:
        return None

    try:
        payload = _resolve(dataset, realization, item)
        payload = _unwrap_dict(payload)
        cycle = payload.get("mean")
        if cycle is None:
            return None
        cycle = np.asarray(cycle, dtype=float)
        z = _first_harmonic(cycle)
        s = _read_arclength(dataset, realization, arclength_key, z.size)
        return ModeRecord(realization.experiment_id, {}, s, z)
    except Exception as exc:
        print(f"WARNING: cannot read periodic mode for {realization.experiment_id}: {exc}")
        return None


def _is_infinite_motor(value: Any) -> bool:
    x = _as_float(value)
    return bool(np.isinf(x) or np.isclose(x, -1.0))


def _finite_motor(value: Any) -> float:
    x = _as_float(value)
    if not np.isfinite(x) or x < 0:
        return float("nan")
    return x


def _genotype(sexp: Any) -> str:
    if sexp is None:
        return ""
    return str(_as_scalar(sexp)).strip().split("_", 1)[0].upper()


def _build_dataset(path: Path) -> Any:
    from cilia.datasets.DataSet import DataSet
    from cilia.datasets.Realization import Realization

    class FolderDataset(DataSet):
        def _initialize_realizations(self) -> None:
            root = Path(self.path)
            if not root.is_dir():
                self.realizations = []
                return
            self.realizations = [
                Realization(child.name)
                for child in sorted(root.iterdir())
                if child.is_dir()
            ]

    return FolderDataset(str(path))


def _collect_conditions(dataset: Any, condition_key: str) -> dict[str, dict[str, Any]]:
    return {
        realization.experiment_id: _read_condition(dataset, realization, condition_key)
        for realization in dataset
    }


def _collect_group(
    label: str,
    dataset_path: Path,
    condition_predicate: Callable[[dict[str, Any]], bool],
    periodic_algorithm_prefix: str,
    selected_nmotor: float | None = None,
) -> ModeGroup:
    from cilia.datastructures.DataName import ARCLENGTH, CONDITION_DESCRIPTION, PERIODIC_AVG

    if not dataset_path.is_dir():
        raise ComparisonError(f"Dataset directory does not exist: {dataset_path}")

    dataset = _build_dataset(dataset_path)
    conditions = _collect_conditions(dataset, CONDITION_DESCRIPTION.key)
    records: list[ModeRecord] = []
    missing_periodic: list[str] = []

    for realization in dataset:
        condition = conditions.get(realization.experiment_id, {})
        if not condition_predicate(condition):
            continue
        record = _read_mode(
            dataset,
            realization,
            PERIODIC_AVG.key,
            ARCLENGTH.key,
            periodic_algorithm_prefix,
        )
        if record is None:
            missing_periodic.append(realization.experiment_id)
            continue
        record.condition = condition
        records.append(record)
        realization.unload_data()

    if not records:
        detail = ""
        if missing_periodic:
            detail = (
                f" Matching realizations existed, but {len(missing_periodic)} lacked a stored "
                f"{PERIODIC_AVG.key} generated by {periodic_algorithm_prefix}*."
            )
        raise ComparisonError(f"No usable modes found for {label} in {dataset_path}.{detail}")

    if missing_periodic:
        print(
            f"WARNING: {label}: skipped {len(missing_periodic)} matching realizations "
            "without the expected periodic average"
        )

    print(f"{label}: selected {len(records)} realization(s)")
    return ModeGroup(label, records, selected_nmotor=selected_nmotor)


def _max_motor(dataset_path: Path) -> float:
    from cilia.datastructures.DataName import CONDITION_DESCRIPTION

    dataset = _build_dataset(dataset_path)
    conditions = _collect_conditions(dataset, CONDITION_DESCRIPTION.key)
    values = [
        _finite_motor(_condition_value(cond, "Nmotor", "N_motor"))
        for cond in conditions.values()
    ]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        raise ComparisonError(f"No finite Nmotor values found in {dataset_path}")
    return float(max(values))


def _interp_complex(s_old: np.ndarray, z_old: np.ndarray, s_new: np.ndarray) -> np.ndarray:
    order = np.argsort(s_old)
    s = np.asarray(s_old, dtype=float)[order]
    z = np.asarray(z_old, dtype=complex)[order]
    unique, idx = np.unique(s, return_index=True)
    z = z[idx]
    if unique.size < 2:
        raise ValueError("need at least two distinct arclength samples")
    real = np.interp(s_new, unique, z.real)
    imag = np.interp(s_new, unique, z.imag)
    return real + 1j * imag


def _overlap(reference: np.ndarray, candidate: np.ndarray, s: np.ndarray) -> complex:
    """Discrete complex overlap used by the Cass R^2 objective.

    ``s`` is retained in the signature because all callers already pass the
    common arclength grid, but Eq. (9) uses an unweighted discrete sum.
    """
    del s
    mask = (
        np.isfinite(reference.real)
        & np.isfinite(reference.imag)
        & np.isfinite(candidate.real)
        & np.isfinite(candidate.imag)
    )
    if np.count_nonzero(mask) < 2:
        return 0.0 + 0.0j
    return complex(np.sum(np.conj(reference[mask]) * candidate[mask]))


def _align_to(reference: np.ndarray, candidate: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply the unit-modulus phase rotation minimizing the Cass residual."""
    ov = _overlap(reference, candidate, s)
    phase_removed = float(np.angle(ov)) if abs(ov) > 0 else 0.0
    return candidate * np.exp(-1j * phase_removed), -phase_removed


def _optimal_complex_adjustment(
    experimental: np.ndarray,
    simulated: np.ndarray,
) -> tuple[complex, float, float]:
    """Return the complex scalar minimizing the Cass squared residual.

    For ``c = rho * exp(i delta)``, this solves

        min_c sum_i |experimental_i - c * simulated_i|^2.

    The closed-form solution is

        c = sum_i conj(simulated_i) * experimental_i / sum_i |simulated_i|^2.

    Returns ``(c, rho, delta)`` where ``rho >= 0`` and ``delta`` is the applied
    global phase shift in radians.
    """
    mask = (
        np.isfinite(experimental.real)
        & np.isfinite(experimental.imag)
        & np.isfinite(simulated.real)
        & np.isfinite(simulated.imag)
    )
    if np.count_nonzero(mask) == 0:
        return 1.0 + 0.0j, 1.0, 0.0

    sim = simulated[mask]
    exp = experimental[mask]
    denominator = float(np.sum(np.abs(sim) ** 2))
    if denominator <= 0.0 or not np.isfinite(denominator):
        return 1.0 + 0.0j, 1.0, 0.0

    coefficient = complex(np.sum(np.conj(sim) * exp) / denominator)
    scale = float(abs(coefficient))
    phase = float(np.angle(coefficient))
    return coefficient, scale, phase


def _build_experimental_reference(zs: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align experimental realizations under global U(1) rotations.

    The remaining common phase is deliberately left free here. It is fixed only
    after the ensemble mean is formed, using the literature display gauge.
    """
    reference = zs[0].copy()
    aligned = zs.copy()
    for _ in range(8):
        aligned = np.stack([_align_to(reference, z, s)[0] for z in zs])
        new_reference = np.mean(aligned, axis=0)
        denom = np.linalg.norm(reference)
        change = np.linalg.norm(new_reference - reference) / max(denom, 1e-15)
        reference = new_reference
        if change < 1e-10:
            break
    return reference, aligned


def _literature_display_rotation(gamma_exp: np.ndarray) -> tuple[complex, int, float]:
    """Place max-amplitude experimental point on the negative real axis.

    This reproduces
        exp(1j * (pi - angle(mode[jmax])))
    from reproduce_WT_fundamental_mode.py.
    """
    amplitude = np.abs(gamma_exp)
    if amplitude.size == 0 or not np.any(np.isfinite(amplitude)):
        return 1.0 + 0.0j, 0, 0.0
    jmax = int(np.nanargmax(amplitude))
    alpha = float(np.pi - np.angle(gamma_exp[jmax]))
    return np.exp(1j * alpha), jmax, alpha


def _phase_profile(gamma_tilde: np.ndarray, mask_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return amplitude and base-side-referenced unwrapped phase.

    This follows the attached WT reproduction script: phase is omitted until
    the amplitude first reaches a fixed fraction of its maximum, then the
    unwrapped phase at that first retained point is set to zero.
    """
    amplitude = np.abs(gamma_tilde)
    phase = np.full(amplitude.shape, np.nan, dtype=float)
    if amplitude.size == 0 or not np.any(np.isfinite(amplitude)):
        return amplitude, phase

    threshold = mask_fraction * np.nanmax(amplitude)
    candidates = np.flatnonzero(np.isfinite(amplitude) & (amplitude >= threshold))
    if candidates.size == 0:
        return amplitude, phase

    first = int(candidates[0])
    segment = np.unwrap(np.angle(gamma_tilde[first:]))
    phase[first:] = segment - segment[0]
    # Keep phase undefined wherever the mode amplitude is too small.
    phase[amplitude < threshold] = np.nan
    return amplitude, phase


def _cass_r2(experimental: np.ndarray, simulated: np.ndarray) -> float:
    """Cass et al. Eq. (9): 1 - sum|exp-sim|^2 / sum|exp|^2."""
    mask = (
        np.isfinite(experimental.real)
        & np.isfinite(experimental.imag)
        & np.isfinite(simulated.real)
        & np.isfinite(simulated.imag)
    )
    if np.count_nonzero(mask) == 0:
        return float("nan")
    denominator = float(np.sum(np.abs(experimental[mask]) ** 2))
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    numerator = float(np.sum(np.abs(experimental[mask] - simulated[mask]) ** 2))
    return 1.0 - numerator / denominator


def _apply_mean_gauge(zs: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Convert first-harmonic modes to the spatial-mean gauge.

    Because both the first temporal Fourier transform and the spatial average
    are linear, applying

        gamma(s,t) -> gamma(s,t) - <gamma(s,t)>_s

    to the phase-conditioned waveform is exactly equivalent to applying

        gamma_tilde(s) -> gamma_tilde(s) - <gamma_tilde(s)>_s

    to its first harmonic.  The average is the arclength average on the common
    plotting grid, evaluated independently for every realization.
    """
    zs = np.asarray(zs, dtype=complex)
    s = np.asarray(s, dtype=float)
    if zs.ndim != 2 or zs.shape[1] != s.size:
        raise ValueError("mean-gauge input must have shape (Nrealization, Ns)")

    span = float(s[-1] - s[0])
    if not np.isfinite(span) or abs(span) < 1e-15:
        spatial_mean = np.mean(zs, axis=1)
    else:
        # Both np.trapezoid (newer NumPy) and np.trapz (older NumPy) support
        # complex arrays and give the continuum-style arclength average.
        trapezoid = getattr(np, "trapezoid", np.trapz)
        spatial_mean = trapezoid(zs, x=s, axis=1) / span
    return zs - spatial_mean[:, None]


def _prepare_groups(
    groups: Sequence[ModeGroup],
    n_grid: int,
    phase_mask_fraction: float,
    gauge: str = "base",
):
    s = np.linspace(0.0, 1.0, n_grid)
    interpolated: dict[str, np.ndarray] = {}
    for group in groups:
        values = np.stack(
            [_interp_complex(record.s, record.z, s) for record in group.records]
        )
        if gauge == "mean":
            values = _apply_mean_gauge(values, s)
        elif gauge != "base":
            raise ValueError(f"unknown angle gauge: {gauge!r}")
        interpolated[group.label] = values

    print(f"\nPreparing {gauge}-gauge modes independently")

    # First align the experimental realizations to one another. This removes
    # arbitrary movie time origins but does not change experimental amplitudes.
    exp_zs = interpolated[EXPERIMENT_LABEL]
    exp_reference, exp_aligned = _build_experimental_reference(exp_zs, s)
    aligned_by_group: dict[str, np.ndarray] = {EXPERIMENT_LABEL: exp_aligned}

    # Align every simulation realization by phase only before averaging, so
    # arbitrary stochastic start phases do not suppress the ensemble mean.
    for group in groups:
        if group.label == EXPERIMENT_LABEL:
            continue
        aligned_by_group[group.label] = np.stack(
            [_align_to(exp_reference, gamma, s)[0] for gamma in interpolated[group.label]]
        )

    # The experiment is the fixed reference: no fitted phase or scale is applied.
    exp_mean_unrotated = np.mean(aligned_by_group[EXPERIMENT_LABEL], axis=0)
    adjustments: dict[str, tuple[complex, float, float]] = {
        EXPERIMENT_LABEL: (1.0 + 0.0j, 1.0, 0.0)
    }
    r2_before_adjustment: dict[str, float] = {EXPERIMENT_LABEL: 1.0}

    # For each model, fit one group-level complex coefficient to its ensemble
    # mean. Multiplying every realization by the same coefficient preserves the
    # within-group variability while making the displayed mean the exact
    # phase-and-scale least-squares optimum under Cass Eq. (9).
    for group in groups:
        if group.label == EXPERIMENT_LABEL:
            continue
        model_mean = np.mean(aligned_by_group[group.label], axis=0)
        r2_before_adjustment[group.label] = _cass_r2(exp_mean_unrotated, model_mean)
        coefficient, scale, phase = _optimal_complex_adjustment(
            exp_mean_unrotated,
            model_mean,
        )
        aligned_by_group[group.label] = aligned_by_group[group.label] * coefficient
        adjustments[group.label] = (coefficient, scale, phase)

    print("\nApplied group-level model adjustments minimizing the Cass residual:")
    print("  (These are additional to the per-realization phase alignment used before averaging.)")
    for group in groups:
        coefficient, scale, phase = adjustments[group.label]
        if group.label == EXPERIMENT_LABEL:
            print("  Experiment fixed: scale=1, phase=0 rad")
        else:
            print(
                f"  {group.label:10s} scale={scale:.8g}, "
                f"phase={phase:+.8g} rad ({np.degrees(phase):+.6g} deg), "
                f"c={coefficient.real:+.8g}{coefficient.imag:+.8g}j"
            )

    # Literature display gauge: one common rotation for experiment and models.
    # It is display-only and does not alter any relative fit or R^2 value.
    display_rotation, jmax, display_alpha = _literature_display_rotation(exp_mean_unrotated)
    for label in aligned_by_group:
        aligned_by_group[label] = aligned_by_group[label] * display_rotation

    print(
        "Literature display gauge: "
        f"s/L={s[jmax]:.6g}, alpha={display_alpha:.6g} rad "
        "(experimental maximum amplitude -> negative real axis)"
    )

    # Experimental ensemble mean used in the Cass R^2 denominator. The common
    # literature display rotation leaves the R^2 unchanged.
    gamma_exp = np.mean(aligned_by_group[EXPERIMENT_LABEL], axis=0)

    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        gammas = aligned_by_group[group.label]
        mean_gamma = np.mean(gammas, axis=0)
        std_real = np.std(gammas.real, axis=0, ddof=1) if len(gammas) > 1 else np.zeros_like(s)
        std_imag = np.std(gammas.imag, axis=0, ddof=1) if len(gammas) > 1 else np.zeros_like(s)
        amplitudes = np.abs(gammas)
        amplitude, phi = _phase_profile(mean_gamma, phase_mask_fraction)
        std_amp = np.std(amplitudes, axis=0, ddof=1) if len(gammas) > 1 else np.zeros_like(s)
        phi_linear, slope, intercept = _linear_phase_fit(s, phi)

        r2_individual = np.asarray([_cass_r2(gamma_exp, gamma) for gamma in gammas], dtype=float)
        r2_mean_mode = _cass_r2(gamma_exp, mean_gamma)
        coefficient, fit_scale, fit_phase = adjustments[group.label]

        result[group.label] = {
            "gammas": gammas,
            "mean_gamma": mean_gamma,
            "std_real": std_real,
            "std_imag": std_imag,
            "amplitude": amplitude,
            "std_amp": std_amp,
            "phi": phi,
            "phi_linear": phi_linear,
            "phase_slope": slope,
            "phase_intercept": intercept,
            "fit_coefficient": coefficient,
            "fit_scale": fit_scale,
            "fit_phase_rad": fit_phase,
            "fit_phase_deg": float(np.degrees(fit_phase)),
            "r2_before_adjustment": r2_before_adjustment[group.label],
            "r2_mean_mode": r2_mean_mode,
            "r2_individual": r2_individual,
            "n": np.array([len(gammas)]),
        }

    return s, result, display_alpha


def _linear_phase_fit(s: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit Phi(s) = slope * (s/L) + intercept on finite phase values."""
    mask = np.isfinite(s) & np.isfinite(phi)
    if np.count_nonzero(mask) < 2:
        return np.full_like(s, np.nan, dtype=float), float("nan"), float("nan")

    slope, intercept = np.polyfit(s[mask], phi[mask], 1)
    fit = np.full_like(s, np.nan, dtype=float)
    fit[mask] = slope * s[mask] + intercept
    return fit, float(slope), float(intercept)


def _save_profiles_csv(path: Path, s: np.ndarray, result: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "n_realizations",
        "s_over_L",
        "gamma_tilde_real_rad",
        "gamma_tilde_imag_rad",
        "amplitude_rad",
        "phase_rad",
        "phase_linear_rad",
        "phase_slope_rad_per_s_over_L",
        "phase_intercept_rad",
        "cass_r2_before_phase_scale_adjustment",
        "cass_r2_mean_mode",
        "applied_model_scale",
        "applied_model_phase_rad",
        "applied_model_phase_deg",
        "std_real",
        "std_imag",
        "std_amplitude",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, data in result.items():
            gamma_tilde = data["mean_gamma"]
            for i, sval in enumerate(s):
                writer.writerow(
                    {
                        "dataset": label,
                        "n_realizations": int(data["n"][0]),
                        "s_over_L": float(sval),
                        "gamma_tilde_real_rad": float(gamma_tilde.real[i]),
                        "gamma_tilde_imag_rad": float(gamma_tilde.imag[i]),
                        "amplitude_rad": float(data["amplitude"][i]),
                        "phase_rad": float(data["phi"][i]) if np.isfinite(data["phi"][i]) else "",
                        "phase_linear_rad": (
                            float(data["phi_linear"][i]) if np.isfinite(data["phi_linear"][i]) else ""
                        ),
                        "phase_slope_rad_per_s_over_L": (
                            float(data["phase_slope"]) if np.isfinite(data["phase_slope"]) else ""
                        ),
                        "phase_intercept_rad": (
                            float(data["phase_intercept"]) if np.isfinite(data["phase_intercept"]) else ""
                        ),
                        "cass_r2_before_phase_scale_adjustment": (
                            float(data["r2_before_adjustment"])
                            if np.isfinite(data["r2_before_adjustment"]) else ""
                        ),
                        "cass_r2_mean_mode": (
                            float(data["r2_mean_mode"]) if np.isfinite(data["r2_mean_mode"]) else ""
                        ),
                        "applied_model_scale": float(data["fit_scale"]),
                        "applied_model_phase_rad": float(data["fit_phase_rad"]),
                        "applied_model_phase_deg": float(data["fit_phase_deg"]),
                        "std_real": float(data["std_real"][i]),
                        "std_imag": float(data["std_imag"][i]),
                        "std_amplitude": float(data["std_amp"][i]),
                    }
                )


def _save_r2_csv(path: Path, groups: Sequence[ModeGroup], result: dict[str, dict[str, Any]]) -> None:
    """Save mean-mode and realization-level Cass R^2 summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "n_realizations",
        "applied_model_scale",
        "applied_model_phase_rad",
        "applied_model_phase_deg",
        "cass_r2_before_phase_scale_adjustment",
        "cass_r2_mean_mode",
        "cass_r2_realization_mean",
        "cass_r2_realization_sd",
        "cass_r2_realization_min",
        "cass_r2_realization_max",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            data = result[group.label]
            values = np.asarray(data["r2_individual"], dtype=float)
            finite = values[np.isfinite(values)]
            writer.writerow(
                {
                    "dataset": group.label,
                    "n_realizations": int(data["n"][0]),
                    "applied_model_scale": float(data["fit_scale"]),
                    "applied_model_phase_rad": float(data["fit_phase_rad"]),
                    "applied_model_phase_deg": float(data["fit_phase_deg"]),
                    "cass_r2_before_phase_scale_adjustment": data["r2_before_adjustment"],
                    "cass_r2_mean_mode": data["r2_mean_mode"],
                    "cass_r2_realization_mean": float(np.mean(finite)) if finite.size else "",
                    "cass_r2_realization_sd": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0 if finite.size == 1 else "",
                    "cass_r2_realization_min": float(np.min(finite)) if finite.size else "",
                    "cass_r2_realization_max": float(np.max(finite)) if finite.size else "",
                }
            )


def _save_adjustments_csv(
    path: Path,
    groups: Sequence[ModeGroup],
    result: dict[str, dict[str, Any]],
) -> None:
    """Save the model-only phase and amplitude adjustments used for plotting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "applied_scale",
        "applied_phase_rad",
        "applied_phase_deg",
        "complex_coefficient_real",
        "complex_coefficient_imag",
        "cass_r2_before_adjustment",
        "cass_r2_after_adjustment",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in groups:
            data = result[group.label]
            coefficient = complex(data["fit_coefficient"])
            writer.writerow(
                {
                    "dataset": group.label,
                    "applied_scale": float(data["fit_scale"]),
                    "applied_phase_rad": float(data["fit_phase_rad"]),
                    "applied_phase_deg": float(data["fit_phase_deg"]),
                    "complex_coefficient_real": float(coefficient.real),
                    "complex_coefficient_imag": float(coefficient.imag),
                    "cass_r2_before_adjustment": float(data["r2_before_adjustment"]),
                    "cass_r2_after_adjustment": float(data["r2_mean_mode"]),
                }
            )


def _print_r2(groups: Sequence[ModeGroup], result: dict[str, dict[str, Any]]) -> None:
    print("\nCass-style complex R^2 against the experimental ensemble mean:")
    for group in groups:
        data = result[group.label]
        values = np.asarray(data["r2_individual"], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size > 1:
            spread = f"; realizations {np.mean(finite):.6f} +/- {np.std(finite, ddof=1):.6f}"
        elif finite.size == 1:
            spread = f"; realization {finite[0]:.6f}"
        else:
            spread = ""
        adjustment = (
            f"; scale={data['fit_scale']:.8g}, "
            f"phase={data['fit_phase_rad']:+.8g} rad "
            f"({data['fit_phase_deg']:+.6g} deg)"
        )
        print(
            f"  {group.label:10s} R^2(mean mode) "
            f"{data['r2_before_adjustment']:.6f} -> {data['r2_mean_mode']:.6f}"
            f"{spread}{adjustment}"
        )


def _save_experimental_phase_rotation_debug(
    output_stem: Path,
    s: np.ndarray,
    result: dict[str, dict[str, np.ndarray]],
    n_rotations: int,
    font_size: float,
) -> None:
    """Save experimental Re/Im modes for uniformly spaced global phase rotations.

    Only the nonredundant interval 0 <= alpha < pi is sampled. A further pi
    rotation changes gamma_tilde only by an overall minus sign.
    """
    if n_rotations <= 0:
        return
    if EXPERIMENT_LABEL not in result:
        raise ComparisonError("Experimental result is missing from phase-rotation debug output")

    z_exp = np.asarray(result[EXPERIMENT_LABEL]["mean_gamma"], dtype=complex)
    rotations = np.linspace(0.0, np.pi, n_rotations, endpoint=False)
    debug_dir = output_stem.with_name(output_stem.name + "_experimental_phase_rotations")
    debug_dir.mkdir(parents=True, exist_ok=True)

    ymax = float(np.nanmax(np.abs(np.concatenate([z_exp.real, z_exp.imag]))))
    if not np.isfinite(ymax) or ymax <= 0:
        ymax = 1.0
    ylim = 1.08 * ymax
    manifest_path = debug_dir / "rotations.csv"

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "alpha_rad", "alpha_over_pi", "filename"])
        writer.writeheader()

        with plt.rc_context(_font_rc(font_size)):
            for index, alpha in enumerate(rotations):
                z_rot = z_exp * np.exp(1j * alpha)
                filename = f"experimental_mode_rotation_{index:02d}_{alpha / np.pi:.2f}pi.png"
                path = debug_dir / filename

                fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
                ax.plot(
                    s,
                    z_rot.real,
                    color=DATASET_COLORS[EXPERIMENT_LABEL],
                    linewidth=2.6,
                    linestyle="-",
                    label=r"$\operatorname{Re}\!\left[\tilde{\gamma}(s)e^{i\alpha}\right]$",
                )
                ax.plot(
                    s,
                    z_rot.imag,
                    color=DATASET_COLORS[EXPERIMENT_LABEL],
                    linewidth=2.6,
                    linestyle="--",
                    label=r"$\operatorname{Im}\!\left[\tilde{\gamma}(s)e^{i\alpha}\right]$",
                )
                ax.axhline(0.0, color="black", linewidth=0.9)
                ax.set_xlim(float(s[0]), float(s[-1]))
                ax.set_ylim(-ylim, ylim)
                ax.set_xlabel(r"$s/L$")
                ax.set_ylabel(r"$\tilde{\gamma}(s)$ [rad]")
                ax.grid(alpha=0.25)
                ax.legend(loc="best", frameon=False)
                ax.text(
                    0.03,
                    0.95,
                    rf"$\alpha={alpha / np.pi:.2f}\pi$",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )
                fig.savefig(path, dpi=250, bbox_inches="tight")
                plt.close(fig)

                writer.writerow(
                    {
                        "index": index,
                        "alpha_rad": float(alpha),
                        "alpha_over_pi": float(alpha / np.pi),
                        "filename": filename,
                    }
                )

    print(f"Saved {n_rotations} experimental phase-rotation plots: {debug_dir}")
    print(f"Saved phase-rotation manifest: {manifest_path}")


def _plot(
    output_stem: Path,
    s: np.ndarray,
    groups: Sequence[ModeGroup],
    result: dict[str, dict[str, Any]],
    show_uncertainty: bool,
    font_size: float,
) -> None:
    with plt.rc_context(_font_rc(font_size)):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(14.2, 10.2),
            sharex="col",
            constrained_layout=True,
        )
        ax_real, ax_amp = axes[0, 0], axes[0, 1]
        ax_imag, ax_phi = axes[1, 0], axes[1, 1]

        dataset_handles = []
        for group in groups:
            data = result[group.label]
            gamma_tilde = data["mean_gamma"]
            color = DATASET_COLORS[group.label]
            n = int(data["n"][0])

            real_line, = ax_real.plot(s, gamma_tilde.real, color=color, linewidth=2.8, label=group.label)
            ax_imag.plot(s, gamma_tilde.imag, color=color, linewidth=2.8)
            ax_amp.plot(s, data["amplitude"], color=color, linewidth=2.8)
            ax_phi.plot(s, data["phi"], color=color, linewidth=2.5)
            ax_phi.plot(s, data["phi_linear"], color=color, linewidth=2.2, linestyle=":")
            dataset_handles.append(real_line)

            print(f"{group.label}: dPhi/d(s/L) = {data['phase_slope']:.6g} rad")

            if show_uncertainty and n > 1:
                sem_real = data["std_real"] / math.sqrt(n)
                sem_imag = data["std_imag"] / math.sqrt(n)
                sem_amp = data["std_amp"] / math.sqrt(n)
                ax_real.fill_between(
                    s, gamma_tilde.real - sem_real, gamma_tilde.real + sem_real,
                    color=color, alpha=0.12,
                )
                ax_imag.fill_between(
                    s, gamma_tilde.imag - sem_imag, gamma_tilde.imag + sem_imag,
                    color=color, alpha=0.12,
                )
                ax_amp.fill_between(
                    s, data["amplitude"] - sem_amp, data["amplitude"] + sem_amp,
                    color=color, alpha=0.10,
                )

        ax_real.legend(handles=dataset_handles, loc="best", frameon=False, ncol=2)
        phase_handles = [
            plt.Line2D([], [], color="black", linewidth=2.5, linestyle="-", label=r"$\Phi(s)$"),
            plt.Line2D([], [], color="black", linewidth=2.2, linestyle=":", label=r"$\Phi_{\mathrm{lin}}(s)$"),
        ]
        ax_phi.legend(handles=phase_handles, loc="best", frameon=False)

        for ax in axes.flat:
            ax.grid(alpha=0.25)
            ax.set_xlim(float(s[0]), float(s[-1]))

        ax_real.axhline(0.0, color="black", linewidth=0.9)
        ax_imag.axhline(0.0, color="black", linewidth=0.9)

        # Use one common, symmetric y-range for the real and imaginary modes.
        component_limit = max(
            max(
                np.nanmax(np.abs(result[group.label]["mean_gamma"].real)),
                np.nanmax(np.abs(result[group.label]["mean_gamma"].imag)),
            )
            for group in groups
        )
        if show_uncertainty:
            for group in groups:
                data = result[group.label]
                n = int(data["n"][0])
                if n > 1:
                    sem_real = data["std_real"] / math.sqrt(n)
                    sem_imag = data["std_imag"] / math.sqrt(n)
                    component_limit = max(
                        component_limit,
                        np.nanmax(np.abs(data["mean_gamma"].real) + sem_real),
                        np.nanmax(np.abs(data["mean_gamma"].imag) + sem_imag),
                    )
        component_limit *= 1.05
        ax_real.set_ylim(-component_limit, component_limit)
        ax_imag.set_ylim(-component_limit, component_limit)

        ax_phi.set_yticks([0.0, -0.5 * np.pi, -np.pi])
        ax_phi.set_yticklabels([r"$0$", r"$-\pi/2$", r"$-\pi$"])
        # ax_phi.set_ylim(-np.pi, 0.0)
        ax_amp.set_ylim(bottom=0)

        ax_real.set_ylabel(r"$\operatorname{Re}\,\tilde{\gamma}(s)$ [rad]")
        ax_amp.set_ylabel(r"$a(s)=|\tilde{\gamma}(s)|$ [rad]")
        ax_imag.set_ylabel(r"$\operatorname{Im}\,\tilde{\gamma}(s)$ [rad]")
        ax_phi.set_ylabel(r"$\Phi(s)$ [rad]")
        ax_imag.set_xlabel(r"$s/L$")
        ax_phi.set_xlabel(r"$s/L$")

        output_stem.parent.mkdir(parents=True, exist_ok=True)
        png_path = output_stem.with_suffix(".png")
        pdf_path = output_stem.with_suffix(".pdf")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved figure: {png_path}")
        print(f"Saved figure: {pdf_path}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--structured-root",
        type=Path,
        default=Path("/media/ssd2/phd_cilia/structured"),
        help="Directory containing cass_2d_phasespace, cass_2d_extraction, cass_3d_extraction and sharma",
    )
    parser.add_argument(
        "--cilia-root",
        type=Path,
        default=None,
        help="Parent directory containing the cilia package, or the cilia package directory itself",
    )
    parser.add_argument(
        "--original-mu-a",
        type=float,
        default=1570.0,
        help="mu_a for the original infinite-motor CASS 2D point; use --all-infinite-original to ignore mu_a",
    )
    parser.add_argument(
        "--all-infinite-original",
        action="store_true",
        help="Use every Nmotor=infinity realization from cass_2d_phasespace instead of selecting mu_a",
    )
    parser.add_argument("--atp", type=float, default=750.0)
    parser.add_argument("--kcl", type=float, default=0.0)
    parser.add_argument("--genotype", default="WT")
    parser.add_argument("--n-grid", type=int, default=201)
    parser.add_argument(
        "--phase-mask-fraction",
        type=float,
        default=0.10,
        help="Start Phi where amplitude first reaches this fraction of its maximum",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("fundamental_modes_comparison"),
        help="Output path without extension",
    )
    parser.add_argument(
        "--font-size",
        type=float,
        default=16.0,
        help="Base font size used for the main and debugging figures",
    )
    parser.add_argument(
        "--phase-rotation-debug-count",
        type=int,
        default=10,
        help="Number of additional experimental rotations around the literature display gauge; set to 0 to disable",
    )
    parser.add_argument("--no-uncertainty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _import_cilia(args.cilia_root)

    try:
        # Import here so --cilia-root is applied first.
        from cilia.transformers.periodic_avg_transformer import periodic_avg_segment_t, periodic_avg_t

        root = args.structured_root.expanduser().resolve()
        path_original = root / "cass_2d_phasespace"
        path_fit_2d = root / "cass_2d_extraction"
        path_fit_3d = root / "cass_3d_extraction"
        path_experiment = root / "sharma"

        max_motor_2d = _max_motor(path_fit_2d)
        max_motor_3d = _max_motor(path_fit_3d)
        print(f"CASS 2D fit: maximum finite Nmotor = {max_motor_2d:g}")
        print(f"CASS 3D fit: maximum finite Nmotor = {max_motor_3d:g}")

        def original_predicate(condition: dict[str, Any]) -> bool:
            nmotor = _condition_value(condition, "Nmotor", "N_motor")
            if not _is_infinite_motor(nmotor):
                return False
            if args.all_infinite_original:
                return True
            mu_a = _as_float(_condition_value(condition, "mu_a"))
            return np.isfinite(mu_a) and np.isclose(mu_a, args.original_mu_a)

        def fit_2d_predicate(condition: dict[str, Any]) -> bool:
            return np.isclose(
                _finite_motor(_condition_value(condition, "Nmotor", "N_motor")),
                max_motor_2d,
            )

        def fit_3d_predicate(condition: dict[str, Any]) -> bool:
            return np.isclose(
                _finite_motor(_condition_value(condition, "Nmotor", "N_motor")),
                max_motor_3d,
            )

        genotype = args.genotype.strip().upper()

        def experiment_predicate(condition: dict[str, Any]) -> bool:
            atp = _as_float(_condition_value(condition, "ATP", "ATP_uM", "ATP_in_uM"))
            kcl = _as_float(_condition_value(condition, "KCl", "KCl_mM", "KCl_in_mM"))
            sexp = _condition_value(condition, "sexp", "genotype")
            return (
                np.isfinite(atp)
                and np.isfinite(kcl)
                and np.isclose(atp, args.atp)
                and np.isclose(kcl, args.kcl)
                and _genotype(sexp) == genotype
            )

        groups = [
            _collect_group(
                "Cass",
                path_original,
                original_predicate,
                periodic_avg_t.func_name + "_",
            ),
            _collect_group(
                "2D",
                path_fit_2d,
                fit_2d_predicate,
                periodic_avg_t.func_name + "_",
                selected_nmotor=max_motor_2d,
            ),
            _collect_group(
                "3D",
                path_fit_3d,
                fit_3d_predicate,
                periodic_avg_t.func_name + "_",
                selected_nmotor=max_motor_3d,
            ),
            _collect_group(
                EXPERIMENT_LABEL,
                path_experiment,
                experiment_predicate,
                periodic_avg_segment_t.func_name + "_",
            ),
        ]

        output_stem = args.output_stem.expanduser().resolve()

        # 1) Stored/base gauge.  This preserves the original behavior and file names.
        s, result, display_alpha = _prepare_groups(
            groups,
            args.n_grid,
            args.phase_mask_fraction,
            gauge="base",
        )
        _plot(
            output_stem,
            s,
            groups,
            result,
            show_uncertainty=not args.no_uncertainty,
            font_size=args.font_size,
        )
        _save_experimental_phase_rotation_debug(
            output_stem,
            s,
            result,
            n_rotations=args.phase_rotation_debug_count,
            font_size=args.font_size,
        )
        print("\nBase-gauge results")
        _print_r2(groups, result)
        csv_path = output_stem.with_name(output_stem.name + "_profiles").with_suffix(".csv")
        _save_profiles_csv(csv_path, s, result)
        print(f"Saved profiles: {csv_path}")
        r2_path = output_stem.with_name(output_stem.name + "_r2_scores").with_suffix(".csv")
        _save_r2_csv(r2_path, groups, result)
        print(f"Saved R^2 scores: {r2_path}")
        adjustments_path = output_stem.with_name(
            output_stem.name + "_model_adjustments"
        ).with_suffix(".csv")
        _save_adjustments_csv(adjustments_path, groups, result)
        print(f"Saved model adjustments: {adjustments_path}")

        # 2) Spatial-mean gauge.  Start again from the unmodified records so all
        # realization alignment, model fitting, display rotation, R^2 values, and
        # phase fits are recomputed independently in this gauge.
        mean_output_stem = output_stem.with_name(output_stem.name + "_mean_gauge")
        s_mean, result_mean, mean_display_alpha = _prepare_groups(
            groups,
            args.n_grid,
            args.phase_mask_fraction,
            gauge="mean",
        )
        _plot(
            mean_output_stem,
            s_mean,
            groups,
            result_mean,
            show_uncertainty=not args.no_uncertainty,
            font_size=args.font_size,
        )
        print("\nMean-gauge results")
        _print_r2(groups, result_mean)
        mean_profiles_path = mean_output_stem.with_name(
            mean_output_stem.name + "_profiles"
        ).with_suffix(".csv")
        _save_profiles_csv(mean_profiles_path, s_mean, result_mean)
        print(f"Saved mean-gauge profiles: {mean_profiles_path}")
        mean_r2_path = mean_output_stem.with_name(
            mean_output_stem.name + "_r2_scores"
        ).with_suffix(".csv")
        _save_r2_csv(mean_r2_path, groups, result_mean)
        print(f"Saved mean-gauge R^2 scores: {mean_r2_path}")
        mean_adjustments_path = mean_output_stem.with_name(
            mean_output_stem.name + "_model_adjustments"
        ).with_suffix(".csv")
        _save_adjustments_csv(mean_adjustments_path, groups, result_mean)
        print(f"Saved mean-gauge model adjustments: {mean_adjustments_path}")

        # Keep the returned angles named and visible for debugging/static analysis.
        _ = display_alpha, mean_display_alpha
        return 0

    except (ComparisonError, ImportError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
