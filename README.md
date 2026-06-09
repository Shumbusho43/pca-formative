# Principal Component Analysis (PCA) from Scratch

**Course:** Mathematics for Machine Learning — Advanced Linear Algebra
**Assignment:** Formative 1 — PCA (eigenvalues, eigenvectors & covariance)
**Cohort / Program:** ALU BSE
**Team (Peer Pair):** David SHUMBUSHO · Keyla Nyacyesa

A from-scratch implementation of Principal Component Analysis using **NumPy only** for all
numerical work (`matplotlib` is used solely for the required plots). The pipeline reduces the
dimensionality of a real African development dataset while retaining as much variance as possible.

## Use case

**Economic activity & population pressure across African countries.** The dataset covers 57
African nations described by economic indicators (GDP per capita, phones, industry, services)
and population-pressure indicators (population, density, birth rate, infant mortality).

## Dataset — `african_development_indicators.csv`

Sourced from World Bank / CIA World Factbook country indicators, filtered to the two African
regions (Northern Africa, Sub-Saharan Africa).

| Property | Value |
|---|---|
| Rows (countries) | 57 |
| Columns | 20 (18 numeric + 2 non-numeric) |
| Non-numeric columns | `Country`, `Region` |
| Missing values | 24 (across 7 countries) — handled by mean imputation |

## What the notebook does

1. **Load** the data with NumPy and set aside the non-numeric columns.
2. **Handle missing values** via column-mean imputation.
3. **Standardize** features: `z = (x - μ) / σ`.
4. **Covariance matrix:** `Σ = ZᵀZ / (n - 1)`.
5. **Eigendecomposition** with `np.linalg.eigh`; sort components by eigenvalue (descending).
6. **Task 2 — dynamic selection:** keep the smallest number of components reaching a 90%
   explained-variance threshold → **10 of 18 components (92.56% variance retained)**.
7. **Project** the data onto the principal components; quantify reconstruction error.
8. **Visualize** the data before vs. after PCA (coloured by region).
9. **Task 3 — optimization & benchmarking:** naive (loops + `eig`) vs. vectorized (`eigh`) vs.
   SVD. On a 2000×200 matrix the vectorized version is **~28× faster** than the naive baseline.

## Files

| File | Description |
|---|---|
| `Template_PCA_Formative_1.ipynb` | Completed notebook (all cells with visible outputs) |
| `african_development_indicators.csv` | Cleaned dataset used by the notebook |
| `PCA_Formative_Submission_David_Keyla.pdf` | Combined PDF (task sheet + notebook) |
| `requirements.txt` | Python dependencies |

## How to run / verify

```bash
# 1. (recommended) create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3a. open interactively
jupyter notebook Template_PCA_Formative_1.ipynb

# 3b. OR run all cells from the command line to verify it executes end-to-end
jupyter nbconvert --to notebook --execute --inplace Template_PCA_Formative_1.ipynb
```

> Keep `african_development_indicators.csv` in the same folder as the notebook so the load cell
> can find it.

## Library policy

Only **NumPy** is used for all PCA computation (imputation, standardization, covariance,
eigendecomposition, projection, benchmarking). `matplotlib` is used only to draw the plots that
the assignment explicitly requires; no `sklearn`, `pandas`, or other libraries are used.
