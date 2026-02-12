# eddy_fp
a deterministic, training‑free framework that transforms 3‑D ocean eddies into a canonical, rotation‑ and scale‑invariant coordinate system.

Compared with existing mesoscale eddy detection approaches—including classical SSH‑based methods, hybrid SSH–velocity systems, and recent 3‑D deep learning architectures such as 3D‑U‑Res‑Net and 3D‑EddyNet—F2 offers a fundamentally different capability.
Whereas current methods focus on classification, segmentation, or contour extraction, F2 provides a deterministic, training‑free framework that transforms every 3‑D eddy into a canonical, rotation‑ and scale‑invariant coordinate system. This enables direct, interpretable cross‑eddy comparisons using morphometric and topological metrics—capabilities absent from all reviewed literature. Deep learning methods can reproduce known eddies but rely heavily on training labels and do not offer canonical shape normalization; hybrid geometric methods extract 3‑D volumes but remain non‑comparable; SSH‑based detectors retain radius biases and miss essential vertical structure. In contrast, F2 produces a stable, physics‑aware fingerprint fully compatible with next‑generation high‑resolution datasets such as SWOT, enabling unified analysis across regions, datasets, and model resolutions.

# eddy_fp – Canonical Fingerprinting & Clustering of 3‑D Ocean Feature Point Clouds
Bernd Becker, 2025–2026  
Co-developed with Copilot

---

## Overview

**F2FP** (Fingerprint‑2 Package) implements a complete workflow for turning
unstructured **3‑D ocean feature point clouds**  
(lat, lon, depth) into:

1. **Canonical, fixed‑size fingerprints** (F2 descriptors)
2. **Unsupervised HDBSCAN‑2 clustering** on fingerprints only
3. **3‑D reconstruction** from fingerprints (no original points required)
4. **Optional co‑location validation** against the original point cloud
5. **Global 3‑D plotting** using GeoVista + PyVista

The system is designed for detecting:
- Eddies  
- Gyres  
- Rings  
- Swirls  
- Drum‑like vortical structures  

and for separating them from:
- Curtains  
- Walls  
- Fronts  
- Sheet‑like boundaries.

f2fp_pkg/
├─ f2fp/
│  ├─ __init__.py
│  ├─ fingerprint.py         # canonicalization + metrics (grid default 12×12×12)
│  ├─ dataset.py             # SVD + robust scaling + UMAP (optional) + HDBSCAN-2
│  ├─ verify_plot.py         # reconstruction + validation + GeoVista plotting + GyreScore coloring
│  └─ (CLI can be added later if you want)
├─ test_synthetic_generators.py   # rings + curtains generators
├─ test_build_fingerprints.py     # writes run_fp.jsonl with dummy labels
├─ test_cluster.py                # builds vectors, runs UMAP + HDBSCAN-2, updates labels in-place
├─ test_plot.py                   # reconstruct & plot, colored by GyreScore
├─ test_validate.py               # co-location validation against synthetic originals
└─ README.md                      # usage and quick start

## Key Concepts

### 1. Canonical Fingerprints (F2)

Each feature is transformed into a **canonical coordinate frame**:

- Translate to geocentric center  
- PCA rotation → canonical axes  
- Scale to `[-1, 1]^3` cube  
- Build a **12×12×12 sparse density grid**  
- Compute:
  - PCA spectrum (`l1, l2, l3`)
  - Linearity, planarity, scattering, anisotropy
  - Ringness, circularity (LAEA projection)
  - Vertical metrics (`v_extent`, persistence, aspect, coherence)
  - Topological hole metrics (Ripser)
  - Radial distribution metrics (RDF)
  - Convexity defect
  - Thickness / planarity ratios
  - Minimal legacy metrics (tiltdeg, radius_m, has_holes, match_ratio)

Everything is stored in a single **JSON lines** file.

### 2. Stage 2 Clustering (HDBSCAN‑2)

Fingerprints are vectorized via:
- Core statistics (≈30 numbers)
- Dense 3‑D grid flattened → **TruncatedSVD** → ≈128 dims  
- Robust scaling → HDBSCAN clustering

Cluster labels are written **back into the same JSONL** file  
(no bitsy mapping files).

### 3. Reconstruction

Each fingerprint contains:
- Center (lon, lat, depth)
- Rotation matrix `R`  
- Scale factor `S`  
- Sparse canonical grid

Reconstruction in world coordinates:


X_world = (X_can @ Rᵀ) * S + center_xyz

### 4. Co‑Location Validation

Using KD‑tree nearest neighbor distances to compare:
- Reconstructed surface  
vs  
- Original point cloud

Provides:
- Fraction within τ  
- d50, d90, d99 metrics  

### 5. Global 3‑D Plotting

Uses **GeoVista + PyVista** to plot reconstructed features on a **3‑D Earth**, with optional overlay of original point clouds.

---

## Dependencies

---

numpy
scipy
scikit-learn
cartopy
pyvista
geovista

**Optional** (recommended):

hdbscan
ripser
umap-learn

Install dependencies manually or via `pip install -r requirements.txt`
(if you create one).

---

## Installation

You can use the package **directly** from the archive:

```bash
unzip f2fp_package.zip
cd f2fp
python -c "import f2fp; print('OK')"

Or install in editable mode:
Shellpip install -e .``

pip install -e .
``
Workflow (Quick Start)
1. Generate synthetic dataset (for testing)
Shellpython test_build_fingerprints.pyShow more lines
This writes:
run_fp.jsonl

A single file with one fingerprint per feature, each with a dummy label.

2. Run HDBSCAN‑2 clustering
ython test_cluster.py

This updates labels inside run_fp.jsonl:

label = "eddy" or "noise"
cluster_id = integer
label_meta = {stage, algo, score}


3. Visualize reconstructed features
Shellpython test_plot.pyShow more lines
Produces:
eddies_test.html

Open in any browser.

4. Optional: Validate reconstructed vs original geometry
Shellpython test_validate.pyShow more lines
Outputs statistics:
frac_within_tau
d50, d90, d99



## Package Structure
