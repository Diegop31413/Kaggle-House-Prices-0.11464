# Breaking the Asymptotic Error Floor in Tabular Regression: Ames Housing Grandmaster Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Kaggle Benchmark](https://img.shields.io/badge/Kaggle%20LB-0.11464-brightgreen.svg)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

This repository contains the official production implementation of the paper:
> **"Breaking the Asymptotic Error Floor in Tabular Regression: Leak-Free Domain Engineering, Non-Linear Interactions, and Entropic Bounded SLSQP Ensembling on the Ames Housing Benchmark"**  
> *Diego Pantoja* (`diego.pantoja@unmsm.edu.pe`), Independent Researcher / UNMSM.

---

##  Core Methodology & Theoretical Integrity

Our framework approaches the empirically estimated error floor ($[0.1135, 0.1145]$ RMSLE) on the Ames Housing benchmark through three orthogonal pillars:

1. **Target-Independent Domain Feature Engineering (24 High-Precision Interactions)**:
   - **17 Pointwise Closed-Form Interactions** (Equations 6–21): Invariant algebraic space metrics computed independently instance-by-instance without data snooping.
   - **7 Contextual Neighborhood Aggregations with Strict In-Fold Isolation** (Equation 22): Relative market deviations whose group statistics are computed **strictly within each cross-validation training fold** (`tr_df`), guaranteeing that validation instances never contribute to group medians.
   - **Strict In-Fold Shift Box-Cox Power Transformations** ($\lambda=0.15$): Skewness selection and shift parameters derived exclusively from training fold distributions.
2. **Multi-Paradigm Base Model Stack (9 Estimators)**:
   - **5 Regularized Linear/Robust Models**: Lasso ($\alpha=0.00045$), Ridge ($\alpha=15.0$), ElasticNet ($\alpha=0.0007, \rho=0.55$), Huber ($\epsilon=1.45, \alpha=280.0$), Bayesian Ridge.
   - **4 GBDTs**: GBR ($2150$ trees, $\eta=0.0142$, $\sqrt{\cdot}$ feature sampling), CatBoost ($1550$ iters, $\eta=0.025, L_2=3.6$), XGBoost ($5000$ trees, $\eta=0.0075$), LightGBM ($3800$ trees, $\eta=0.009$, $10$ leaves).
3. **Entropic Bounded SLSQP Ensembling**:
   - Meta-optimization with Shannon entropy regularization ($\lambda=4.8 \times 10^{-4}$) and anti-monopoly bounds $[0.035, 0.250]$ (3.5% min, 25% max per model).
   - **Nested Meta-Level Cross-Validation** over out-of-fold predictions across 50 optimizations.
   - Calibrated Golden Blend ($0.86 \cdot \text{SLSQP} + 0.14 \cdot \text{Simple Blend}$) with defensive quantile clipping $[\$55\text{k}, \$473\text{k}]$.

---

##  Empirical Results

| Model / Ensemble | OOF RMSE (10-Fold $\times$ 5 Seeds) | Fold Mean | SLSQP Weight | Family |
| :--- | :---: | :---: | :---: | :---: |
| **GBR (Sklearn)** | 0.10983 | 0.10990 | 24.60% | GBDT |
| **Huber Regressor** | 0.11027 | 0.10951 | 20.62% | Linear/Robust |
| **CatBoost** | 0.11068 | 0.11135 | 14.81% | GBDT |
| **ElasticNet** | 0.11015 | 0.10973 | 13.53% | Linear/Robust |
| **Lasso ($L_1$)** | 0.11027 | 0.10984 | 9.41% | Linear/Robust |
| **Ridge ($L_2$)** | 0.11079 | 0.11036 | 4.59% | Linear/Robust |
| **XGBoost** | 0.11200 | 0.11328 | 4.83% | GBDT |
| **LightGBM** | 0.11354 | 0.11553 | 4.07% | GBDT |
| **Bayesian Ridge** | 0.11081 | 0.11034 | 3.52% | Linear/Robust |
| **Golden Ensemble (v57)** | **0.10749** | — | **100.0%** | **Near-Balanced (53.4% / 46.6%)** |
| **Kaggle Public Leaderboard** | **0.11464** | — | — | **Top Tier Record** |

---

##  Execution & Kaggle Reproduction

### Recommended Workflow (Kaggle Cloud Execution)
1. Create a new notebook on Kaggle for **House Prices: Advanced Regression Techniques**.
2. Select **GPU T4 x2 or P100** under Accelerator settings.
3. Upload `Ames_Housing_Grandmaster_Framework_v57_Production.ipynb`.
4. Click **"Save Version" -> "Run & Save All (Commit)"**.
5. Once complete, your notebook will render with all full execution logs, OOF tables, and 4 diagnostic figures, producing `submission.csv` ready for 1-click submission.

### Local Execution
```bash
git clone https://github.com/diego-pantoja/ames-housing-grandmaster.git
cd ames-housing-grandmaster
pip install -r requirements.txt
python Ames_Housing_Grandmaster_Framework_v57_Production.py
```

Toggle execution modes in Cell 1:
```python
FULL_RUN = True   # Full Paper Benchmark: 10-Fold x 5 Seeds (450 fits, ~8-12 min)
FULL_RUN = False  # Fast Smoke Test: 5-Fold x 1 Seed (45 fits, ~1 min)
```

---

##  Citation

This repository accompanies the unpublished manuscript:

Diego Pantoja. “Blending Is All You Need: Leak-Free Domain Engineering and Entropic SLSQP Ensembling for Ames Housing Regression” (2026). Unpublished manuscript.

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
