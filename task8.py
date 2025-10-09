#!/usr/bin/env python3
"""
Task 8 — NDVI cell-wise geometrical probability, fuzzy membership, CLT uncertainty,
priority score — with expanded visualization pack.

This version keeps the fixed rasterio context (no 'Dataset is closed' error)
and adds extra plots: NDVI quicklook, CDF, scatter (P vs μ), priority histogram,
CLT small multiples, and side-by-side raster thumbnails.
"""

import os, math, json
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio.transform import Affine
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
TIF_PATH = "/workspaces/geostatistics_masters/ndvi_s2_2024_growing_season.tif"
OUT_DIR  = "/workspaces/geostatistics_masters/outputs"
MAP_DIR  = os.path.join(OUT_DIR, "task08_map_pack")
os.makedirs(MAP_DIR, exist_ok=True)

CELL_PX   = 25                 # grid size (square cells, in pixels)
T_THRESH  = 0.50               # crisp vegetation threshold for "NDVI >= t"

FUZZY_A, FUZZY_B = 0.20, 0.70  # fuzzy membership breakpoints

RANDOM_SEED = 7
MEAN_SAMPLE_N = 1000
CLT_SIZES = [10, 30, 100]

WRITE_GEOTIFFS = False

# ----------------------------
# LOAD NDVI (inside context)
# ----------------------------
assert os.path.exists(TIF_PATH), f"Not found: {TIF_PATH}"
with rio.open(TIF_PATH) as src:
    ndvi = src.read(1).astype("float32")
    nodata = src.nodata
    profile = src.profile.copy()
    transform: Affine = src.transform
    crs = src.crs
    width, height = src.width, src.height

if nodata is not None:
    ndvi = np.where(ndvi == nodata, np.nan, ndvi)

valid = np.isfinite(ndvi)
vals  = ndvi[valid]
assert vals.size > 0, "No valid NDVI pixels."
H, W = ndvi.shape

# ----------------------------
# HELPERS
# ----------------------------
def block_iter(h, w, cell):
    for r0 in range(0, h, cell):
        for c0 in range(0, w, cell):
            r1 = min(h, r0 + cell)
            c1 = min(w, c0 + cell)
            yield r0, r1, c0, c1

def mu_veg(x, a=FUZZY_A, b=FUZZY_B):
    out = np.full_like(x, np.nan, dtype="float32")
    finite = np.isfinite(x)
    xf = x[finite]
    denom = max(1e-6, (b - a))
    y = np.empty_like(xf, dtype="float32")
    y[xf <= a] = 0.0
    y[xf >= b] = 1.0
    mid = (xf > a) & (xf < b)
    y[mid] = (xf[mid] - a) / denom
    out[finite] = y
    return out

def write_geotiff(path, array, base_profile):
    prof = base_profile.copy()
    prof.update(dtype="float32", count=1, nodata=np.nan, compress="deflate", predictor=3, tiled=True)
    with rio.open(path, "w", **prof) as dst:
        dst.write(array.astype("float32"), 1)

# ----------------------------
# GEOMETRICAL PROBABILITY per cell
# ----------------------------
grid_records = []
geom_map = np.full_like(ndvi, np.nan, dtype="float32")

for r0, r1, c0, c1 in block_iter(H, W, CELL_PX):
    block = ndvi[r0:r1, c0:c1]
    m = np.isfinite(block)
    n_valid = int(m.sum())
    if n_valid == 0:
        frac = np.nan
    else:
        frac = float((block[m] >= T_THRESH).sum() / n_valid)

    grid_records.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "valid": n_valid, "p_ge_t": frac})
    if n_valid > 0:
        geom_map[r0:r1, c0:c1] = frac

df_geom = pd.DataFrame(grid_records)
df_geom.to_csv(os.path.join(OUT_DIR, "task08_geom_cell_probs.csv"), index=False)

# ----------------------------
# FUZZY MEMBERSHIP map μ_veg
# ----------------------------
fuzzy_map = mu_veg(ndvi, FUZZY_A, FUZZY_B)
fuzzy_summary = {
    "mu_mean": float(np.nanmean(fuzzy_map)),
    "mu_std": float(np.nanstd(fuzzy_map)),
    "mu_p05": float(np.nanpercentile(fuzzy_map, 5)),
    "mu_p50": float(np.nanpercentile(fuzzy_map, 50)),
    "mu_p95": float(np.nanpercentile(fuzzy_map, 95)),
}
pd.DataFrame([fuzzy_summary]).to_csv(os.path.join(OUT_DIR, "task08_fuzzy_summary.csv"), index=False)

# ----------------------------
# DISTRIBUTION SNAPSHOT
# ----------------------------
hist_counts, hist_edges = np.histogram(vals, bins=60, range=(-0.2, 0.9))
cdf_x = np.sort(vals)
cdf_y = np.linspace(0, 1, cdf_x.size, endpoint=True)

pd.DataFrame({"bin_left": hist_edges[:-1], "bin_right": hist_edges[1:], "count": hist_counts})\
  .to_csv(os.path.join(OUT_DIR, "task08_hist_counts.csv"), index=False)
pd.DataFrame({"ndvi": cdf_x, "cdf": cdf_y}).to_csv(os.path.join(OUT_DIR, "task08_empirical_cdf.csv"), index=False)

# ----------------------------
# CLT insight for uncertainty on mean NDVI
# ----------------------------
rng = np.random.default_rng(RANDOM_SEED)
draws = rng.choice(vals, size=MEAN_SAMPLE_N, replace=True)
hat_mu = float(np.mean(draws))
hat_sigma = float(np.std(draws, ddof=1))

clt_table = []
for n in CLT_SIZES:
    se = hat_sigma / math.sqrt(n)
    ci_lo = hat_mu - 1.96 * se
    ci_hi = hat_mu + 1.96 * se
    clt_table.append({"n": n, "mean_hat": hat_mu, "se": float(se), "ci_lo": float(ci_lo), "ci_hi": float(ci_hi)})
pd.DataFrame(clt_table).to_csv(os.path.join(OUT_DIR, "task08_mean_uncertainty.csv"), index=False)

# ----------------------------
# PRIORITY SCORE
# ----------------------------
priority_map = np.full_like(ndvi, np.nan, dtype="float32")
cell_scores = []

for r0, r1, c0, c1 in block_iter(H, W, CELL_PX):
    gm_block = geom_map[r0:r1, c0:c1]
    mu_block = fuzzy_map[r0:r1, c0:c1]
    vmask = np.isfinite(gm_block) & np.isfinite(mu_block)
    if np.count_nonzero(vmask) == 0:
        score = np.nan
    else:
        score = 0.6 * float(np.nanmean(gm_block)) + 0.4 * float(np.nanmean(mu_block))
    cell_scores.append({"r0": r0, "r1": r1, "c0": c0, "c1": c1, "score": score})
    if np.isfinite(score):
        priority_map[r0:r1, c0:c1] = score

df_scores = pd.DataFrame(cell_scores)
df_scores.to_csv(os.path.join(OUT_DIR, "task08_priority_scores.csv"), index=False)

# ----------------------------
# BASE MAPS (existing)
# ----------------------------
plt.figure(figsize=(7.5,5.5))
plt.imshow(geom_map, vmin=0, vmax=1)
plt.colorbar(label=f"P(NDVI ≥ {T_THRESH})")
plt.title("Geometrical Probability by Cell")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "geom_prob_map.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7.5,5.5))
plt.imshow(fuzzy_map, vmin=0, vmax=1)
plt.colorbar(label="μ_veg (0..1)")
plt.title(f"Fuzzy Membership (A={FUZZY_A}, B={FUZZY_B})")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "fuzzy_map.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7.5,5.5))
plt.imshow(priority_map, vmin=0, vmax=1)
plt.colorbar(label="Priority score (0..1)")
plt.title("Priority Map (0.6·P + 0.4·μ)")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "priority_map.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# NEW: NDVI quicklook + valid mask
# ----------------------------
plt.figure(figsize=(7.5,5.5))
plt.imshow(ndvi, vmin=-0.2, vmax=0.9)
plt.colorbar(label="NDVI")
plt.title("NDVI (Quicklook)")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "ndvi_quicklook.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7.5,5.5))
plt.imshow(valid.astype(float), vmin=0, vmax=1)
plt.colorbar(label="Valid mask (1=valid)")
plt.title("Valid Data Mask")
plt.axis("off")
plt.savefig(os.path.join(MAP_DIR, "valid_mask.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# NEW: Empirical CDF with key quantiles
# ----------------------------
pcts = [5, 25, 50, 75, 95]
q = np.nanpercentile(vals, pcts)
plt.figure(figsize=(7.5,4.5))
plt.plot(cdf_x, cdf_y)
for px, qq in zip(pcts, q):
    yy = px/100
    plt.axvline(qq, linestyle="--", alpha=0.5)
    plt.text(qq, 0.02+yy*0, f"P{px}={qq:.2f}", rotation=90, va="bottom", ha="right")
plt.xlabel("NDVI")
plt.ylabel("Empirical CDF")
plt.title("Empirical CDF of NDVI (valid pixels)")
plt.grid(alpha=0.2)
plt.savefig(os.path.join(MAP_DIR, "ndvi_empirical_cdf.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# NEW: Scatter P(NDVI≥t) vs mean μ_veg per cell (colored by priority)
# ----------------------------
# aggregate cell-wise P and mean μ for scatter
cell_vis = []
for r0, r1, c0, c1 in block_iter(H, W, CELL_PX):
    gm = geom_map[r0:r1, c0:c1]
    mu = fuzzy_map[r0:r1, c0:c1]
    if np.isfinite(gm).any() and np.isfinite(mu).any():
        p = float(np.nanmean(gm))
        m = float(np.nanmean(mu))
        s = 0.6*p + 0.4*m
        cell_vis.append((p, m, s))
cell_vis = np.array(cell_vis, dtype="float32")
if cell_vis.size:
    plt.figure(figsize=(6.8,5.4))
    sc = plt.scatter(cell_vis[:,0], cell_vis[:,1], c=cell_vis[:,2], s=18, alpha=0.8)
    cb = plt.colorbar(sc)
    cb.set_label("Priority score")
    plt.xlabel(f"P(NDVI ≥ {T_THRESH})")
    plt.ylabel("mean μ_veg")
    plt.title("Cell-wise Relationship: P vs μ (colored by priority)")
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(MAP_DIR, "scatter_p_vs_mu_priority.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ----------------------------
# NEW: Priority histogram with simple smooth overlay
# ----------------------------
cell_scores_only = df_scores["score"].replace([np.inf, -np.inf], np.nan).dropna().values
if cell_scores_only.size:
    plt.figure(figsize=(7.5,4.5))
    plt.hist(cell_scores_only, bins=40, alpha=0.8)
    # crude smooth via CDF finite-diff trick (visual aid without seaborn)
    xs = np.linspace(0, 1, 256)
    cdf = np.searchsorted(np.sort(cell_scores_only), xs, side="right") / cell_scores_only.size
    # central diff to suggest a PDF-ish curve, scaled for visibility
    pdf_like = np.gradient(cdf, xs)
    pdf_like = pdf_like / pdf_like.max() * (np.histogram(cell_scores_only, bins=40, range=(0,1))[0].max())
    plt.plot(xs, pdf_like, linewidth=2)
    plt.xlabel("Priority score (0..1)")
    plt.ylabel("Frequency (bars) / Relative density (line)")
    plt.title("Priority Score Distribution")
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(MAP_DIR, "priority_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()

# ----------------------------
# NEW: CLT panel — sampling distributions for n in CLT_SIZES
# ----------------------------
plt.figure(figsize=(7.8,4.8))
for i, n in enumerate(CLT_SIZES, 1):
    se = hat_sigma / math.sqrt(n)
    x = np.linspace(hat_mu - 4*se, hat_mu + 4*se, 300)
    y = (1/np.sqrt(2*np.pi)/se) * np.exp(-0.5*((x-hat_mu)/se)**2)
    # separate plots, no subplots (per notebook rule), but layered curves are OK
    plt.plot(x, y, label=f"n={n} (±1.96·SE ≈ [{hat_mu-1.96*se:.3f},{hat_mu+1.96*se:.3f}])")
plt.xlabel("Sample mean NDVI")
plt.ylabel("Relative density (normal approx)")
plt.title("CLT: Sampling Distribution of Mean NDVI")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig(os.path.join(MAP_DIR, "clt_sampling_distributions.png"), dpi=150, bbox_inches="tight")
plt.close()

# ----------------------------
# NEW: Small multiples — thumbnails of the three key maps
# (One figure, but still a single Axes; we compose via simple montage layout)
# ----------------------------
# We’ll create a single tall strip stacking images to respect “one plot per figure” style.
def save_strip(images, titles, outfile, vmins, vmaxs):
    h = 5.0 * len(images)
    plt.figure(figsize=(7.5, h))
    y = 1.0
    step = 1.0 / len(images)
    for i, (arr, ttl, vmin, vmax) in enumerate(zip(images, titles, vmins, vmaxs)):
        ax = plt.axes([0.05, 1 - (i+1)*step + 0.05, 0.9, step - 0.1])
        im = ax.imshow(arr, vmin=vmin, vmax=vmax)
        ax.set_title(ttl)
        ax.axis("off")
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

save_strip(
    images=[geom_map, fuzzy_map, priority_map],
    titles=[f"P(NDVI ≥ {T_THRESH})", f"μ_veg (A={FUZZY_A}, B={FUZZY_B})", "Priority (0.6·P + 0.4·μ)"],
    outfile=os.path.join(MAP_DIR, "strip_geom_fuzzy_priority.png"),
    vmins=[0,0,0], vmaxs=[1,1,1]
)

# ----------------------------
# OPTIONAL GEOTIFF EXPORTS
# ----------------------------
if WRITE_GEOTIFFS:
    base_prof = profile.copy()
    base_prof.update(count=1, dtype="float32", nodata=np.nan)
    write_geotiff(os.path.join(OUT_DIR, "task08_geom_prob.tif"), geom_map, base_prof)
    write_geotiff(os.path.join(OUT_DIR, "task08_fuzzy_mu.tif"),   fuzzy_map, base_prof)
    write_geotiff(os.path.join(OUT_DIR, "task08_priority.tif"),   priority_map, base_prof)

# ----------------------------
# Console summary
# ----------------------------
print("\n=== TASK 8 SYNTHESIS ===")
print(f"Raster shape: {H}x{W}, valid={vals.size}")
print(f"Fuzzy μ summary: {json.dumps(fuzzy_summary, indent=2)}")
print("Mean uncertainty (CLT overlay):")
for r in clt_table:
    print(f" n={r['n']:>3d}  mean≈{r['mean_hat']:.3f}  95%CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]")
print("\nWrote figures to:", MAP_DIR)
for p in [
    "geom_prob_map.png",
    "fuzzy_map.png",
    "priority_map.png",
    "distribution_and_CLT.png",          # original panel
    "ndvi_quicklook.png",
    "valid_mask.png",
    "ndvi_empirical_cdf.png",
    "scatter_p_vs_mu_priority.png",
    "priority_histogram.png",
    "clt_sampling_distributions.png",
    "strip_geom_fuzzy_priority.png",
]:
    print(" -", os.path.join(MAP_DIR, p))
print("CSVs:", "task08_geom_cell_probs.csv, task08_fuzzy_summary.csv, task08_mean_uncertainty.csv, task08_priority_scores.csv, task08_hist_counts.csv, task08_empirical_cdf.csv")
if WRITE_GEOTIFFS:
    print("GeoTIFFs: task08_geom_prob.tif, task08_fuzzy_mu.tif, task08_priority.tif")
