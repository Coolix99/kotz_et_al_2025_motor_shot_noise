#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd

from pipeline_utils import (
    compute_l1,
    compute_l1_s3,
    compute_l2,
    gaussian_loglik,
    infer_biological_params,
)

EPS = 1e-12

RAW_FEATURES = ["mu", "eta", "zeta", "mu_a", "fstar", "beta"]


# ============================================================
# LOAD
# ============================================================

def load_df(csv_path):
    df = pd.read_csv(csv_path)

    required = RAW_FEATURES + ["strategy"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # oscillation definition (robust)
    if "oscillatory_full" in df.columns:
        df["oscillatory"] = (
            df["oscillatory_full"]
            .fillna(0)
            .replace([np.inf, -np.inf], 0)
            .astype(int)
        )
    elif "oscillatory" in df.columns:
        df["oscillatory"] = df["oscillatory"].astype(int)
    else:
        df["oscillatory"] = 1

    df["accepted"] = df["oscillatory"] == 1

    # derived params
    df["mu_a_times_zeta"] = df["mu_a"] * df["zeta"]
    df["mu_a_div_zeta"] = df["mu_a"] / (df["zeta"] + EPS)

    return df


# ============================================================
# SUMMARY
# ============================================================

def summarize(df):
    rows = []

    for strat, dfg in df.groupby("strategy"):
        n = len(dfg)
        acc = np.mean(dfg["accepted"])

        rows.append({
            "strategy": strat,
            "n": n,
            "acceptance": acc,
            "mu_a_div_zeta_mean": dfg["mu_a_div_zeta"].mean(),
        })

    out = pd.DataFrame(rows).sort_values("strategy")

    print("\n=== Summary ===")
    print(out.to_string(index=False))

    return out


# ============================================================
# RANKING
# ============================================================

def report_top(df, key="log_posterior", topk=5):
    if key not in df.columns:
        print(f"Column {key} missing")
        return

    print(f"\n=== TOP {topk} by {key} ===")

    df_sorted = df.sort_values(key, ascending=False).head(topk)

    cols = [
        "experiment_id",
        "strategy",
        "logL1",
        "log_posterior",
        "tau_opt",
        "mu", "eta", "zeta", "mu_a", "fstar", "beta",
    ]
    cols = [c for c in cols if c in df.columns]

    print(df_sorted[cols].to_string(index=False))


# ============================================================
# EXPLAIN SINGLE EXPERIMENT
# ============================================================

def explain_experiment(df, exp_id, dimension):
    row = df[df["experiment_id"] == exp_id]

    if len(row) == 0:
        print(f"{exp_id} not found")
        return

    row = row.iloc[0]

    print("\n" + "=" * 60)
    print(f"EXPERIMENT {exp_id}")
    print("=" * 60)

    # --- L1 ---
    if "logQ_full" in row and "logQ_reduced" in row:
        l1 = compute_l1_s3(
            row.get("amplitude_ratio", np.nan),
            row.get("frequency_ratio", np.nan),
            row.get("lambda_full", np.nan),
            row.get("lambda_reduced", np.nan),
            row.get("logQ_full", np.nan),
            row.get("logQ_reduced", np.nan),
        )
        print("Using S3 likelihood (with Q)")
    else:
        l1 = compute_l1(
            row.get("amplitude_ratio", np.nan),
            row.get("frequency_ratio", np.nan),
            row.get("lambda_full", np.nan),
            row.get("lambda_reduced", np.nan),
        )
        print("Using S2 likelihood")

    print(f"logL1 (recomputed): {l1:.3f}")
    print(f"logL1 (csv)       : {row.get('logL1', np.nan):.3f}")

    # --- L2 ---
    sim_A = np.sqrt(row["amplitude_full"] * row["amplitude_reduced"])
    sim_f = np.sqrt(row["f_full"] * row["f_reduced"])

    theta_scaled = {
        "mu": row["mu"],
        "eta": row["eta"],
        "mu_a_times_zeta": row["mu_a"] * row["zeta"],
        "mu_a_div_zeta_0": row["mu_a"] / row["zeta"],
        "zeta": row["zeta"],
        "fstar": row["fstar"],
        "beta": row["beta"],
    }

    tau = row.get("tau_opt", np.nan)
    mu_a_div_zeta = row.get("mu_a_div_zeta_opt", np.nan)

    l2 = compute_l2(
        sim_A,
        sim_f,
        tau,
        mu_a_div_zeta,
        theta_scaled["mu_a_div_zeta_0"],
    )

    print(f"logL2 (recomputed): {l2:.3f}")

    # --- BIO PARAMS ---
    params = infer_biological_params(
        theta_scaled,
        row["B_opt"],
        mu_a_div_zeta,
        tau, dimension
    )
    print("\n--- Scaling ---")
    print(f"{'tau_opt':20s}: {tau:.4e}")
    print(f"{'mu_a/zeta':20s}: {mu_a_div_zeta:.4e}")
    print(f"{'B':20s}: {row['B_opt']:.4e}")

    print("\n--- Biological parameters ---")
    for k, v in params.items():
        print(f"{k:20s}: {v:10.4e}")

    print("\n--- TOTAL ---")
    print(f"log_posterior (csv): {row['log_posterior']:.3f}")
    return params

# ============================================================
# EXPORT FILTERED CANDIDATES
# ============================================================

def export_candidates(df,dimension, threshold=-120, out_csv="top_candidates.csv"):
    df_sel = df[df["logL1"] > threshold].copy()

    print(f"\nSelected {len(df_sel)} / {len(df)} candidates")

    records = []

    for _, row in df_sel.iterrows():
        try:
            theta_scaled = {
                "mu": row["mu"],
                "eta": row["eta"],
                "mu_a_times_zeta": row["mu_a"] * row["zeta"],
                "mu_a_div_zeta_0": row["mu_a"] / row["zeta"],
                "zeta": row["zeta"],
                "fstar": row["fstar"],
                "beta": row["beta"],
            }

            params = infer_biological_params(
                theta_scaled,
                row["B_opt"],
                row["mu_a_div_zeta_opt"],
                row["tau_opt"], dimension
            )

        except Exception:
            params = {}

        rec = row.to_dict()
        rec.update(params)
        records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)

    print(f"Saved → {out_csv}")
    return df_out

from config_sbi import BIO_PRIORS
def decompose_bio_prior(params):
    rows = []

    total = 0.0

    for key, prior in BIO_PRIORS.items():
        val = params.get(key)

        if val is None or val <= 0 or not np.isfinite(val):
            rows.append({
                "param": key,
                "value": val,
                "log10(value)": np.nan,
                "prior_mean": prior["mean_log10"],
                "sigma": prior["sigma_log10"],
                "log_contribution": -np.inf,
                "optimal": 0.0,
                "penalty": -np.inf,
            })
            continue

        logv = np.log10(val)

        logp = gaussian_loglik(
            logv,
            prior["mean_log10"],
            prior["sigma_log10"]
        )

        total += logp

        rows.append({
            "param": key,
            "value": val,
            "log10(value)": logv,
            "prior_mean": prior["mean_log10"],
            "sigma": prior["sigma_log10"],
            "log_contribution": logp,
            "optimal": 0.0,
            "penalty": logp,
        })

    df = pd.DataFrame(rows)

    print("\n=== BIO PRIOR BREAKDOWN ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\nTotal log prior:", total)

    return df

# ============================================================
# MAIN
# ============================================================

def run_diagnostics(csv_path, dimension, threshold=-120):
    df = load_df(csv_path)

    summarize(df)

    if "log_posterior" in df.columns:
        report_top(df, "log_posterior")
    if "logL1" in df.columns:
        report_top(df, "logL1")

    # explain best
    if "log_posterior" in df.columns:
        best = df.sort_values("log_posterior", ascending=False).head(3)
        for exp_id in best["experiment_id"]:
            params = explain_experiment(df, exp_id, dimension)

            print("\n--- PRIOR BREAKDOWN ---")
            decompose_bio_prior(params)

    #export_candidates(df, threshold, out_csv=f"top_candidates_{args.dim}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--stage",
        choices=["S1", "S2", "S3"],
        default="S3",
        help="SBI stage (default: S3)",
    )

    ap.add_argument(
        "--dim",
        choices=["2d", "3d"],
        default="3d",
        help="Simulation dimension (default: 3d)",
    )

    ap.add_argument(
        "--csv",
        default=None,
        help="Optional: override CSV path",
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=-5,
    )

    args = ap.parse_args()

    # -------------------------------------------------
    # AUTO-SELECT CSV
    # -------------------------------------------------
    if args.csv is not None:
        csv_path = args.csv
    else:
        stage = args.stage.lower()
        csv_path = f"scalar_observables_sbi_{stage}_{args.dim}.csv"

    print(f"\nUsing CSV: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    run_diagnostics(csv_path, args.dim, args.threshold)