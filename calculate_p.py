# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
from scipy import stats

model_names = [
    "Unet",
    "DeepLabV3",
    "miniseg",
    "segnet",
    "unetpp",
    "SwinUNETR",
    "AttentionUnet",
    "UNETR",
    "MISSFormer",
    "UCTransNet",
    "TransFuse",
]

def read_dice_csv(path: str) -> pd.Series:
    """
    Read dice_scores.csv with lines like:
      01_test.png,0.5726
    Return: pd.Series indexed by filename -> dice (float)
    """
    df = pd.read_csv(path, header=None, names=["fname", "dice"])
    df["fname"] = df["fname"].astype(str).str.strip()
    df["dice"] = pd.to_numeric(df["dice"], errors="coerce")
    df = df.dropna(subset=["fname", "dice"])
    # if duplicate filenames exist, average them
    s = df.groupby("fname")["dice"].mean()
    return s

def paired_tests(baseline: pd.Series, other: pd.Series):
    """
    Align by fname intersection, return stats.
    """
    common = baseline.index.intersection(other.index)
    b = baseline.loc[common].astype(float).to_numpy()
    o = other.loc[common].astype(float).to_numpy()
    diff = o - b

    n = len(common)
    mean_b, mean_o = float(np.mean(b)), float(np.mean(o))
    mean_diff = float(np.mean(diff))

    # paired t-test
    t_res = stats.ttest_rel(o, b, nan_policy="omit")
    p_t = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else np.nan

    # Wilcoxon signed-rank (more robust)
    # If all diffs are zero, wilcoxon will fail; handle it.
    try:
        if np.allclose(diff, 0):
            p_w = 1.0
        else:
            w_res = stats.wilcoxon(diff, zero_method="wilcox", correction=False, alternative="two-sided")
            p_w = float(w_res.pvalue)
    except Exception:
        p_w = np.nan

    return {
        "n_common": n,
        "mean_dice_baseline": mean_b,
        "mean_dice_model": mean_o,
        "mean_diff(model-baseline)": mean_diff,
        "p_paired_t": p_t,
        "p_wilcoxon": p_w,
    }

def main(root: str):
    baseline_csv = os.path.join(root, "dice_scores.csv")
    print(baseline_csv)
    if not os.path.exists(baseline_csv):
        raise FileNotFoundError(f"Baseline dice_scores.csv not found: {baseline_csv}")

    baseline = read_dice_csv(baseline_csv)

    rows = []
    for m in model_names:
        model_csv = os.path.join(root, m, "predict", "dice_scores.csv")
        if not os.path.exists(model_csv):
            rows.append({"model": m, "error": f"missing: {model_csv}"})
            continue

        other = read_dice_csv(model_csv)
        r = paired_tests(baseline, other)
        r["model"] = m
        r["model_csv"] = model_csv
        rows.append(r)

    out = pd.DataFrame(rows)

    # Sort: most significant first by wilcoxon p
    if "p_wilcoxon" in out.columns:
        out = out.sort_values(["p_wilcoxon", "p_paired_t"], ascending=True, na_position="last")

    out_path = os.path.join(root, "p_values_vs_baseline.csv")
    out.to_csv(out_path, index=False, float_format="%.6g")
    print("Saved:", out_path)
    print(out)

if __name__ == "__main__":
    # 改成你的 root 路径（包含 baseline dice_scores.csv 以及各模型子目录）
    # 例如：root = r"D:\exp_results" 或 r"/home/ma-user/work/xxx"
    root = r"E:\python\PythonProject\mrseg\Pytorch_Medical_Segmentation\results\drive"
    main(root)
