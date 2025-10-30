#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foundations demo on NDVI:
- Deterministic vs. Stochastic summaries
- Bootstrap uncertainty (CIs) for statistics and area proportions
- Sampling concepts: LLN convergence & biased sampling example
- NDVI area-by-class: classified map (GeoTIFF + PNG) and equal-area area stats + chart

Usage:
  python ndvi_foundations.py "/path/to/ndvi.tif" --boot 1000 --seed 42
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject, calculate_default_transform
except ImportError:
    print("This script requires rasterio. Try: pip install rasterio", file=sys.stderr)
    sys.exit(1)

# ------------------------ helpers ------------------------
def read_ndvi_values(tif_path: str) -> Tuple[np.ndarray, Dict]:
    """Read first band, mask nodata, return 1D array of valid NDVI values and metadata."""
    with rasterio.open(tif_path) as src:
        band1 = src.read(1)
        nodata = src.nodata
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
            "nodata": nodata,
        }
    if nodata is not None:
        mask = band1 == nodata
    else:
        mask = np.isnan(band1)
    vals = band1[~mask].astype(np.float64)
    domain_mask = (vals >= -1.0) & (vals <= 1.0)
    vals = vals[domain_mask]
    return vals, meta


def basic_stats(x: np.ndarray) -> Dict[str, float]:
    q = np.nanpercentile
    return {
        "count_valid": int(x.size),
        "min": float(np.nanmin(x)),
        "p5": float(q(x, 5)),
        "p25": float(q(x, 25)),
        "p50": float(q(x, 50)),
        "p75": float(q(x, 75)),
        "p95": float(q(x, 95)),
        "max": float(np.nanmax(x)),
        "mean": float(np.nanmean(x)),
        "std": float(np.nanstd(x, ddof=1)),
    }


def bootstrap_stats(x: np.ndarray, n_boot: int, rng: np.random.Generator,
                    thresholds: List[float]) -> pd.DataFrame:
    n = x.size
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n, endpoint=False)
        xb = x[idx]
        row = {
            "mean": np.mean(xb),
            "median": np.median(xb),
            "p95": np.percentile(xb, 95),
        }
        for t in thresholds:
            row[f"prop_ge_{t}"] = np.mean(xb >= t)
        draws.append(row)
    return pd.DataFrame(draws)


def ci(series: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    lo = series.quantile(alpha / 2.0)
    hi = series.quantile(1 - alpha / 2.0)
    return float(lo), float(hi)


def save_hist(vals: np.ndarray, out_png: str, bins: int = 60):
    plt.figure(figsize=(7, 4.5))
    plt.hist(vals, bins=bins, edgecolor="black")
    plt.title("NDVI Histogram")
    plt.xlabel("NDVI")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_ecdf(vals: np.ndarray, out_png: str):
    x = np.sort(vals)
    y = np.arange(1, x.size + 1) / x.size
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y)
    plt.title("NDVI ECDF (Deterministic distribution view)")
    plt.xlabel("NDVI")
    plt.ylabel("Cumulative probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_bootstrap_dist(df: pd.DataFrame, out_png: str, columns: List[str]):
    plt.figure(figsize=(8, 5))
    for col in columns:
        xs = df[col].values
        plt.hist(xs, bins=40, alpha=0.4, label=col)
    plt.title("Bootstrap distributions (Stochastic view)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_lln_convergence(vals: np.ndarray, out_png: str,
                         sample_sizes=(50, 100, 250, 500, 1000, 5000, 10000),
                         reps=100, rng=None):
    true_mean = float(np.mean(vals))
    records = []
    for n in sample_sizes:
        for _ in range(reps):
            idx = rng.integers(0, vals.size, size=n, endpoint=False)
            m = float(np.mean(vals[idx]))
            records.append({"n": n, "sample_mean": m, "abs_error": abs(m - true_mean)})
    df = pd.DataFrame(records)
    grp = df.groupby("n")["abs_error"]
    n_list, med_err, p95_err = [], [], []
    for n, series in grp:
        n_list.append(n)
        med_err.append(series.median())
        p95_err.append(series.quantile(0.95))
    plt.figure(figsize=(7, 4.5))
    plt.loglog(n_list, med_err, marker="o", label="Median |error|")
    plt.loglog(n_list, p95_err, marker="s", label="95th perc |error|")
    plt.title("Sampling concepts: error vs. sample size (LLN)")
    plt.xlabel("Sample size n (log scale)")
    plt.ylabel("Absolute error of mean (log scale)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return df


def demo_biased_sampling(vals_2d: np.ndarray, mask_2d: np.ndarray, out_png: str, rng=None):
    vals = vals_2d[mask_2d]
    true_mean = float(np.mean(vals))
    H, W = vals_2d.shape
    h2, w2 = H // 2, W // 2
    quad_mask = np.zeros_like(mask_2d, dtype=bool)
    quad_mask[:h2, :w2] = True  # top-left quadrant
    biased_vals = vals_2d[quad_mask & mask_2d]

    ns = [100, 500, 1000, 5000]
    recs = []
    for n in ns:
        idx = rng.integers(0, vals.size, size=n, endpoint=False)
        m_rand = float(np.mean(vals[idx]))
        if biased_vals.size == 0:
            m_biased = np.nan
        else:
            idxb = rng.integers(0, biased_vals.size, size=n, endpoint=False)
            m_biased = float(np.mean(biased_vals[idxb]))
        recs.append({"n": n, "true_mean": true_mean, "random_mean": m_rand, "biased_mean": m_biased})

    df = pd.DataFrame(recs)
    plt.figure(figsize=(7, 4.5))
    plt.axhline(true_mean, linestyle="--", label=f"True mean={true_mean:.4f}")
    plt.plot(df["n"], df["random_mean"], marker="o", label="Random sample mean")
    if not np.isnan(df["biased_mean"]).any():
        plt.plot(df["n"], df["biased_mean"], marker="s", label="Biased (top-left) sample mean")
    plt.title("Biased vs. random sampling (spatial quadrant)")
    plt.xlabel("Sample size n")
    plt.ylabel("Estimated mean NDVI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return df

# ---------- NEW: NDVI classes map + equal-area area stats ----------
def classify_ndvi(arr2d: np.ndarray, mask2d: np.ndarray, thresholds: List[float]) -> Tuple[np.ndarray, List[str]]:
    """
    Classify NDVI into bins using thresholds.
    bins = [-inf, t1), [t1, t2), ..., [tK, +inf)
    Returns class_id array (0=nodata, 1..K+1 classes) and human-readable labels.
    """
    thr = sorted(thresholds)
    bins = [-np.inf] + thr + [np.inf]
    class_ids = np.zeros_like(arr2d, dtype=np.uint8)  # 0=nodata
    valid = mask2d & np.isfinite(arr2d)
    # np.digitize with right=False -> bin i when bins[i-1] < x <= bins[i] if using right=True
    cls = np.digitize(arr2d, bins[1:]) + 1  # shift to 1..K+1
    class_ids[valid] = cls[valid].astype(np.uint8)

    labels = []
    for i in range(len(bins) - 1):
        a, b = bins[i], bins[i + 1]
        if np.isneginf(a):
            labels.append(f"NDVI < {b:.3f}")
        elif np.isposinf(b):
            labels.append(f"NDVI ≥ {a:.3f}")
        else:
            labels.append(f"{a:.3f} ≤ NDVI < {b:.3f}")
    return class_ids, labels


def write_class_geotiff(path_out: str, class_ids: np.ndarray, src_meta: Dict, cmap: Dict[int, Tuple[int,int,int]]):
    meta = {
        "driver": "GTiff",
        "height": class_ids.shape[0],
        "width": class_ids.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": src_meta["crs"],
        "transform": src_meta["transform"],
        "compress": "LZW",
        "nodata": 0,
    }
    with rasterio.open(path_out, "w", **meta) as dst:
        dst.write(class_ids, 1)
        dst.write_colormap(1, cmap)


def render_class_png(path_out: str, class_ids: np.ndarray, labels: List[str], colors: List[Tuple[float,float,float]]):
    # Build a ListedColormap with 0 as transparent
    import matplotlib.colors as mcolors
    rgba = [(0,0,0,0)] + [(*c, 1.0) for c in colors]  # 0=nodata transparent
    cmap = mcolors.ListedColormap(rgba)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(class_ids, cmap=cmap, interpolation="nearest")
    plt.title("NDVI classes")
    plt.axis("off")

    # Legend
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=colors[i], label=labels[i]) for i in range(len(labels))]
    plt.legend(handles=patches, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(path_out, dpi=180, bbox_inches="tight")
    plt.close()


def equal_area_class_area(class_ids: np.ndarray, src_meta: Dict, epsg=6933) -> Tuple[np.ndarray, Dict]:
    """
    Reproject class_ids (categorical) to equal-area EPSG:6933 and compute area by class.
    Returns (areas_m2_by_class_id, target_meta)
    """
    dst_crs = rasterio.crs.CRS.from_epsg(epsg)
    transform, width, height = calculate_default_transform(
        src_meta["crs"], dst_crs, src_meta["width"], src_meta["height"], *rasterio.transform.array_bounds(
            src_meta["height"], src_meta["width"], src_meta["transform"]
        )
    )
    dst = np.zeros((height, width), dtype=np.uint8)
    reproject(
        source=class_ids,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest,
        dst_nodata=0,
    )
    # Pixel area (m^2) in equal-area
    px_area = abs(transform.a * transform.e)  # (xres * yres), e is negative
    classes = np.unique(dst)
    areas = {}
    for cid in classes:
        if cid == 0:
            continue
        count = int((dst == cid).sum())
        areas[cid] = count * px_area
    tgt_meta = {"crs": dst_crs, "transform": transform, "width": width, "height": height}
    return np.array([areas.get(cid, 0.0) for cid in range(0, dst.max() + 1)]), tgt_meta


def save_area_by_class_csv_png(areas_m2_by_cid: np.ndarray, labels: List[str], out_csv: str, out_png: str):
    # Build tidy DataFrame skipping 0 (nodata)
    rows = []
    for i, label in enumerate(labels, start=1):
        m2 = float(areas_m2_by_cid[i]) if i < len(areas_m2_by_cid) else 0.0
        rows.append({"class_id": i, "label": label, "area_m2": m2, "area_km2": m2 / 1e6})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(df["label"], df["area_km2"])
    plt.ylabel("Area (km²)")
    plt.title("NDVI area by class (equal-area EPSG:6933)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ------------------------ main workflow ------------------------
@dataclass
class Config:
    input_tif: str
    out_dir: str
    seed: int = 42
    n_boot: int = 1000
    alpha: float = 0.05
    thresholds: Tuple[float, ...] = (0.2, 0.3, 0.5)


def main(cfg: Config):
    os.makedirs(cfg.out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    print(f"Reading: {cfg.input_tif}")
    vals, meta = read_ndvi_values(cfg.input_tif)
    print(f"Valid pixels: {vals.size:,}")

    # Deterministic summaries
    det = basic_stats(vals)
    det_df = pd.DataFrame([det])
    det_df.to_csv(os.path.join(cfg.out_dir, "deterministic_stats.csv"), index=False)
    print("Deterministic stats:")
    for k, v in det.items():
        print(f"  {k:>12}: {v}")

    # Plots: histogram + ECDF
    save_hist(vals, os.path.join(cfg.out_dir, "ndvi_histogram.png"))
    save_ecdf(vals, os.path.join(cfg.out_dir, "ndvi_ecdf.png"))

    # Stochastic: bootstrap
    print(f"Bootstrapping ({cfg.n_boot} draws) ...")
    boot = bootstrap_stats(vals, cfg.n_boot, rng, list(cfg.thresholds))
    boot.to_csv(os.path.join(cfg.out_dir, "bootstrap_draws.csv"), index=False)

    # Summarize CIs
    rows = []
    for col in ["mean", "median", "p95"] + [f"prop_ge_{t}" for t in cfg.thresholds]:
        lo, hi = ci(boot[col], alpha=cfg.alpha)
        rows.append({"metric": col, "estimate": det_df.iloc[0].get(col, np.nan),
                     f"CI{int(100*(1-cfg.alpha))}_lo": lo, f"CI{int(100*(1-cfg.alpha))}_hi": hi})
    ci_df = pd.DataFrame(rows)
    ci_df.to_csv(os.path.join(cfg.out_dir, "bootstrap_confidence_intervals.csv"), index=False)

    # Plot bootstrap distributions
    plot_cols = ["mean", "median", "p95"] + [f"prop_ge_{t}" for t in cfg.thresholds]
    plot_bootstrap_dist(boot[plot_cols], os.path.join(cfg.out_dir, "bootstrap_distributions.png"), plot_cols)

    # Sampling concepts: LLN
    lln_df = plot_lln_convergence(vals, os.path.join(cfg.out_dir, "lln_error_vs_n.png"), rng=rng)
    lln_df.to_csv(os.path.join(cfg.out_dir, "lln_sampling_errors.csv"), index=False)

    # Load 2D for spatial things
    with rasterio.open(cfg.input_tif) as src:
        arr2d = src.read(1).astype(np.float64)
        nodata = src.nodata
        if nodata is not None:
            mask2d = arr2d != nodata
        else:
            mask2d = ~np.isnan(arr2d)
        valid_domain = (arr2d >= -1.0) & (arr2d <= 1.0)
        mask2d &= valid_domain
        src_meta_full = {
            "crs": src.crs, "transform": src.transform,
            "width": src.width, "height": src.height
        }

    # Biased sampling demo
    biased_df = demo_biased_sampling(arr2d, mask2d, os.path.join(cfg.out_dir, "biased_vs_random.png"), rng=rng)
    biased_df.to_csv(os.path.join(cfg.out_dir, "biased_vs_random.csv"), index=False)

    # Deterministic area proportions at thresholds (point estimates)
    prop_rows = []
    for t in cfg.thresholds:
        prop = float(np.mean(vals >= t))
        prop_rows.append({"threshold": t, "prop_ge_threshold": prop})
    pd.DataFrame(prop_rows).to_csv(os.path.join(cfg.out_dir, "deterministic_area_proportions.csv"), index=False)

    # ---------- NEW: NDVI classes map + equal-area area stats ----------
    class_ids, labels = classify_ndvi(arr2d, mask2d, list(cfg.thresholds))

    # Colors for classes (RGB in 0..1)
    colors = [
        (0.65, 0.0, 0.15),   # lowest NDVI (red-ish)
        (0.95, 0.67, 0.0),   # orange
        (0.6, 0.8, 0.2),     # yellow-green
        (0.1, 0.5, 0.1),     # green
    ][:len(labels)]  # ensure right length

    # GeoTIFF colormap: 0 nodata transparent/black, 1..N classes colored (0..255)
    cmap = {0: (0, 0, 0)}  # nodata
    for i, c in enumerate(colors, start=1):
        cmap[i] = tuple(int(255 * v) for v in c)

    write_class_geotiff(
        os.path.join(cfg.out_dir, "ndvi_classes.tif"),
        class_ids, src_meta_full, cmap
    )
    render_class_png(
        os.path.join(cfg.out_dir, "ndvi_classes.png"),
        class_ids, labels, colors
    )

    # Equal-area reprojection and class areas
    areas_by_cid, _ = equal_area_class_area(class_ids, src_meta_full, epsg=6933)
    save_area_by_class_csv_png(
        areas_by_cid, labels,
        os.path.join(cfg.out_dir, "ndvi_area_by_class.csv"),
        os.path.join(cfg.out_dir, "ndvi_area_by_class.png"),
    )

    # Console summary
    print("\n=== FOUNDATIONS SUMMARY ===")
    print("- Deterministic (point estimates): deterministic_stats.csv, deterministic_area_proportions.csv")
    print("- Stochastic (uncertainty): bootstrap_draws.csv; CIs -> bootstrap_confidence_intervals.csv")
    print("- Sampling concepts: lln_error_vs_n.png + lln_sampling_errors.csv; biased_vs_random.png + CSV")
    print("- NDVI classes map: ndvi_classes.tif (paletted), ndvi_classes.png (with legend)")
    print("- NDVI area by class: ndvi_area_by_class.csv, ndvi_area_by_class.png (equal-area EPSG:6933)")
    print(f"Output folder: {cfg.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foundations demo on NDVI raster")
    parser.add_argument("input", help="Path to NDVI GeoTIFF")
    parser.add_argument("--out", default="ndvi_foundations_report", help="Output folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--boot", type=int, default=1000, help="Number of bootstrap resamples")
    parser.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level (e.g., 0.05 -> 95% CI)")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.3, 0.5],
                        help="NDVI thresholds for class breaks & proportions")
    args = parser.parse_args()

    cfg = Config(
        input_tif=args.input,
        out_dir=args.out,
        seed=args.seed,
        n_boot=args.boot,
        alpha=args.alpha,
        thresholds=tuple(args.thresholds),
    )
    main(cfg)
