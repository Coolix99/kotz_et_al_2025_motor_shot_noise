import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import eigh, LinAlgError

from utils_rdf.config import OUTPUT_DATA_SBI_S3_DIR

# -------------------------
# Default configuration
# -------------------------
CSV_PATH = OUTPUT_DATA_SBI_S3_DIR/"summary_s3.csv"
OUT_PATH = "figures/final/final_corner.png"
LIKELIHOOD_THRESHOLD = 1e-15
TOP_N = 2000

# Parameters
PARAMS = ["zeta_hat", "eta", "fstar", "beta", "mu"]
LINEAR_SCALE_PARAMS = ["eta"]

AX_LIMITS = {
    "zeta_hat": (100, 500),
    "eta": (0.1, 0.7),
    "fstar": (2.0, 2.5),
    "beta": (1, 5),
    "mu": (1, 30),
}

# LaTeX-style axis labels
LABELS_LATEX = {
    "zeta_hat": r"$\hat{\zeta} = \mu_a \, \zeta$",
    "eta": r"$\eta$",
    "fstar": r"$f^{\ast}$",
    "beta": r"$\beta$",
    "mu": r"$\mu$",
}

mpl.rcParams.update({
    "font.size": 15,
    "axes.labelsize": 17,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.figsize": (10, 10),
    "axes.linewidth": 1.3,
    "text.usetex": True,
    "font.family": "serif",
})


# -------------------------
# Utilities
# -------------------------
def load_and_prepare(csv_path, params, lik_threshold):
    df = pd.read_csv(csv_path)
    if "likelihood" not in df.columns:
        raise ValueError("CSV must contain a 'likelihood' column")

    # Filter by likelihood
    df["likelihood"] = df["likelihood"].astype(float)
    df = df[df["likelihood"] > lik_threshold].copy()
    df["logL"] = np.log(df["likelihood"] + 1e-50)

    # Derived parameter
    if "mu_a" in df.columns and "zeta" in df.columns:
        df["zeta_hat"] = df["mu_a"].astype(float) * df["zeta"].astype(float)
    else:
        raise ValueError("mu_a and zeta must be in CSV to compute zeta_hat")

    # Drop non-finite
    df = df[np.isfinite(df[params]).all(axis=1)]

    # Filter by axis limits
    global AX_LIMITS
    for p in params:
        if p in AX_LIMITS and AX_LIMITS[p] is not None:
            lo, hi = AX_LIMITS[p]
            n_before = len(df)
            df = df[(df[p] >= lo) & (df[p] <= hi)]
            n_after = len(df)
            if n_after < n_before:
                print(f"[filter] {p}: removed {n_before - n_after} outside ({lo}, {hi})")

    df = df.reset_index(drop=True)
    return df

import matplotlib.ticker as mticker

def apply_ticks(ax, xscale, yscale, nticks=4):
    """
    Force each axis to show a fixed number of ticks (approx. nticks).
    Works for both linear and log scales.
    """
    if xscale == "log":
        ax.xaxis.set_major_locator(mticker.LogLocator(numticks=nticks))
        ax.xaxis.set_minor_locator(mticker.LogLocator(subs='auto'))
    else:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nticks))

    if yscale == "log":
        ax.yaxis.set_major_locator(mticker.LogLocator(numticks=nticks))
        ax.yaxis.set_minor_locator(mticker.LogLocator(subs='auto'))
    else:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nticks))


def topn_points(df, n):
    return df.sort_values("logL", ascending=False).head(n).copy()


def fit_gaussian(df_top, params):
    X = df_top[params].values
    weights = np.exp(df_top["logL"].values - np.max(df_top["logL"].values))
    weights /= np.sum(weights)
    mu_hat = np.sum(X * weights[:, None], axis=0)
    Xc = X - mu_hat
    Sigma_hat = (Xc.T * weights) @ Xc
    try:
        vals, _ = eigh(Sigma_hat)
        if np.any(vals <= 0):
            raise LinAlgError("Covariance not PD")
    except LinAlgError:
        Sigma_hat += np.eye(len(params)) * 1e-10
    return mu_hat, Sigma_hat


def plot_gaussian_density_contours(ax, mu, Sigma, idxs, color="red"):
    mu2 = mu[idxs]
    cov2 = Sigma[np.ix_(idxs, idxs)]
    vals, vecs = eigh(cov2)
    vals = np.clip(vals, 1e-12, None)
    t = np.linspace(0, 2 * np.pi, 400)
    circle = np.stack([np.cos(t), np.sin(t)])
    for s in [1, 2, 3]:
        ell = mu2[:, None] + vecs @ np.diag(np.sqrt(vals)) @ (s * circle)
        ax.plot(ell[0], ell[1], color=color, lw=1.0, alpha=0.7)


# -------------------------
# Plot
# -------------------------
def plot_corner(df, params, mu_hat, Sigma, best_point, out_path):
    D = len(params)
    fig, axes = plt.subplots(D, D, figsize=(3.2 * D, 3.2 * D))

    cmap = plt.cm.jet
    norm = mpl.colors.LogNorm(vmin=df["likelihood"].min(), vmax=df["likelihood"].max())

    for i, py in enumerate(params):
        for j, px in enumerate(params):
            ax = axes[i, j]
            if i < j:
                ax.axis("off")
                continue

            # Skip diagonal panels
            if i == j:
                ax.axis("off")
                continue

            sc = ax.scatter(df[px], df[py],
                            c=df["likelihood"], cmap=cmap, norm=norm,
                            s=35, alpha=0.85, edgecolor="none")

            # Best and Gaussian fit markers
            max_lik = df["likelihood"].max()
            color_max = cmap(norm(max_lik))

            ax.scatter(best_point[j], best_point[i], s=120, marker="x", c=color_max, lw=2)
            #ax.scatter(mu_hat[j], mu_hat[i], s=120, marker="+", c="red", lw=2)
            try:
                plot_gaussian_density_contours(ax, mu_hat, Sigma, [j, i], color="red")
            except Exception as e:
                print(f"[warn] contour failed for {px},{py}: {e}")

            # Scales and limits
            if px not in LINEAR_SCALE_PARAMS:
                ax.set_xscale("log")
            if py not in LINEAR_SCALE_PARAMS:
                ax.set_yscale("log")
            if AX_LIMITS.get(px):
                ax.set_xlim(AX_LIMITS[px])
            if AX_LIMITS.get(py):
                ax.set_ylim(AX_LIMITS[py])

            


            print(i,j)
            # --- Hide all ticks/labels first (both major & minor) ---
            ax.tick_params(axis="both", which="both",
                        labelbottom=False, labelleft=False,
                        bottom=False, left=False)

            # --- Re-enable only the outer axes ---
            if i == D - 1:
                ax.set_xlabel(LABELS_LATEX.get(px, px))
                ax.tick_params(axis="x", which="both",
                            labelbottom=True, bottom=True)

            if j == 0:
                ax.set_ylabel(LABELS_LATEX.get(py, py))
                ax.tick_params(axis="y", which="both",
                            labelleft=True, left=True)

            # --- NOW enforce 3 ticks (MUST be after re-enabling ticks) ---
            apply_ticks(
                ax,
                xscale="log" if px not in LINEAR_SCALE_PARAMS else "linear",
                yscale="log" if py not in LINEAR_SCALE_PARAMS else "linear",
                nticks=3
            )

    # Log-scale colorbar for likelihood
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, orientation="vertical", fraction=0.02, pad=0.01
    )
    cbar.set_label(r"$\mathrm{Likelihood}$")
    cbar.ax.set_yscale("log")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.show()


# -------------------------
# Main
# -------------------------
def main():
    print("=== Running SBI corner plot ===")
    df = load_and_prepare(CSV_PATH, PARAMS, LIKELIHOOD_THRESHOLD)
    df_top = topn_points(df, TOP_N)
    mu_hat, Sigma_hat = fit_gaussian(df_top, PARAMS)
    best_idx = np.argmax(df["logL"].values)
    best_point = df.iloc[best_idx][PARAMS].values

    print("\n--- Gaussian Fit Summary ---")
    for p, v in zip(PARAMS, mu_hat):
        print(f"{p:>12s} = {v:.4g}")
    eigvals = np.linalg.eigvalsh(Sigma_hat)
    print("Eigenvalues of covariance:", eigvals)
    if np.any(eigvals <= 0):
        print("⚠️  Covariance matrix not positive definite!")
    print("Covariance:\n", Sigma_hat)

    plot_corner(df, PARAMS, mu_hat, Sigma_hat, best_point, OUT_PATH)


if __name__ == "__main__":
    main()
