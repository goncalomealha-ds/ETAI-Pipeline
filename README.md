# Baseline Predictive Pipeline -- ETAI

## Gonçalo Mealha
## Student number: 20260565

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` <br/> 
| 3 | EDA + preprocessing -- diagnose the data, then fix it | src/data_diagnostics.py (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and src/preprocessing.py (leak-safe category cleanup, mechanism-matched imputation with _was_missing indicators for MNAR columns, a deployable ColumnTransformer, and the train/test split itself, all in the one file rather than split across two) replace the old naive dropna()/pd.get_dummies() preprocessing; encoder/scaler pair (count encoding + robust scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; config.yaml gains diagnostics and preprocessing sections -- see "Preprocessing decisions" below.

# My progress & results

**Gonçalo Mealha**  
**Student number:** 20260565

---

## Overview
This project focuses on predicting two-year recidivism using ProPublica's COMPAS dataset, which originates from an algorithm utilized across US courts to guide bail and sentencing decisions. The primary goal is to establish an end-to-end, leak-safe machine learning pipeline while systematically auditing predictive performance against ethical fairness (specifically examining disparity in false-positive rates across racial groups).

---

| Week | Practical Class Focus | Added / Changed in the Pipeline | Results & Impact |
|---|---|---|---|
| **2** | Introduction & baseline pipeline | Initial setup: naive single train/test split, simple `dropna()`, dummy encoding, baseline Decision Tree and Logistic Regression models. | **DT (`max_depth=5`):** Train: 0.680, Test: 0.668<br/>**LR (`max_iter=1000`):** Train: 0.679, Test: 0.6803 |
| **3** | EDA & Diagnosis-Preprocessing | Added `diagnostics.py` (statistical missingness mechanism checks via $\chi^2$ and Cramér's V, domain validity rules, duplicate detection) and `preprocessing.py` (leak-safe category standardization, MNAR indicators, empirical encoder/scaler pairing via `ColumnTransformer`, redundant feature elimination). | **DT (`max_depth=5`):** Train: 0.684, Test: 0.665<br/>**LR (`max_iter=1000`):** Train: 0.676, Test: 0.657 |
| **4** | Model comparison & cross-validation | Added a Dummy majority-class baseline and evaluated Dummy, Logistic Regression, Decision Tree and Random Forest using 5-fold stratified cross-validation. Training-validation gaps are also monitored to assess potential overfitting. | **Dummy:** CV accuracy = 0.549 ± 0.000<br/>**LR:** CV accuracy = 0.673 ± 0.013<br/>**DT:** CV accuracy = 0.675 ± 0.013<br/>**RF:** CV accuracy = 0.641 ± 0.017 |
| **5** | Hyperparameter tuning & model selection | Implemented Bayesian hyperparameter optimization (Optuna) across tree-based models on the development set using 5-fold stratified CV, followed by out-of-fold diagnostic classification reports and fairness audits. | **Tuned DT (trial 14):** CV accuracy = 0.680<br/>**Tuned RF (trial 20):** CV accuracy = 0.684 |
| **6** | Ensemble methods | Extended pipeline to advanced ensemble architectures (AdaBoost with decision stumps and Oblique Random Forest with multivariate linear projections), evaluating baseline vs. tuned variants using 5-fold stratified CV. | **AdaBoost (baseline / tuned):** CV accuracy = 0.676 ± 0.011 / 0.676 ± 0.012<br/>**Oblique RF (baseline / tuned):** CV accuracy = 0.649 ± 0.008 / 0.646 ± 0.010 |


---

## Preprocessing Decisions

* **`age`:** Filtered through domain validity rules ($18 \le \text{age} \le 100$). Impossible boundary values are converted to `NaN` and imputed using the median.
* **`decile_score` & `score_text`:** Excluded from the model feature set via `drop_columns` to avoid target leakage, as they represent COMPAS's proprietary predictions rather than raw individual characteristics.
* **`priors_count`:** Diagnosed as Missing Not At Random (MNAR) via $\chi^2$ test and Cramér's V association against demographic predictors. An explicit `priors_count_was_missing` binary indicator was constructed prior to median imputation so the informative absence pattern is preserved for downstream models.
* **`prior_offenses`, `age_in_months`, `juvenile_total`:** Dropped permanently due to severe multicollinearity confirmed through high Variance Inflation Factor (VIF) scores and direct linear dependency on `priors_count` and `age`.
* **`sex` & `c_charge_degree`:** Standardized via canonical category mapping to resolve capitalization and whitespace inconsistencies; missing values imputed with the mode (`most_frequent`) along with an MNAR flag for charge degree.
* **`race`:** Dropped from model training inputs to prevent explicit proxy bias, but tracked alongside `y_test` strictly for the disparate impact and fairness audit.
* **Transformations:** Encoder and scaler selections configured through the `ColumnTransformer` (`TargetEncoder` + `RobustScaler`), fitted strictly on `X_train` to prevent test-fold leakage.

---

## Best Model & Results

### Performance Summary

| Model | Baseline / Untuned CV | Tuned CV accuracy (mean ± std) | CV Train–Val Gap | OOF accuracy ($n=5771$) | OOF Class 0 Recall | OOF Class 1 Recall |
|---|---:|---:|---:|---:|---:|---:|
| Dummy (majority) | 0.549 ± 0.000 | — | -0.000 | 0.549 | 1.00 | 0.00 |
| Logistic regression | 0.673 ± 0.013 | — | +0.002 | 0.673 | 0.78 | 0.54 |
| Decision tree | 0.675 ± 0.013 | 0.680 (trial 14) | +0.010 | 0.680 | 0.78 | 0.55 |
| Random forest | 0.641 ± 0.017 | **0.684 (trial 20)** | +0.097 (untuned) | **0.684** | 0.78 | **0.56** |
| AdaBoost | 0.676 ± 0.011 | 0.676 ± 0.012 | **+0.006** | 0.680 | 0.79 | 0.55 |
| Oblique Random Forest | 0.649 ± 0.008 | 0.646 ± 0.010 | +0.111 | 0.680 | **0.80** | 0.53 |


### Analysis


* **Model comparison:** The current cross-validation results show very similar validation accuracy for Logistic Regression and Decision Tree (**0.673 ± 0.013** and **0.675 ± 0.013**, respectively). The Decision Tree has a slightly higher mean CV accuracy, but the difference is very small. Therefore, the current results do not indicate a clear performance separation between these two models.

* **Generalization:** Logistic Regression has a smaller mean train-validation gap (**+0.002**) than the Decision Tree (**+0.010**), indicating a smaller difference between training and validation performance in the current 5-fold cross-validation.

* **Random Forest:** The Random Forest obtained a lower mean validation accuracy (**0.641 ± 0.017**) and a substantially larger train-validation gap (**+0.097**), indicating a larger discrepancy between training and validation performance in the current configuration.

* **Dummy baseline:** The Dummy classifier achieved **0.549 ± 0.000** accuracy by predicting the majority class. This provides a baseline against which the other models can be compared. Both Logistic Regression and Decision Tree substantially exceed this baseline in cross-validation.

* **Class performance:** For the Logistic Regression, Class 0 has recall of **0.78**, compared with **0.54** for Class 1. For the Decision Tree, the corresponding recalls are **0.78** and **0.54**. Both models therefore show different recall levels across the two classes, despite having similar overall accuracy.

* **Fairness audit:** The fairness analysis reports the false positive rate (FPR) separately by race. The results should be interpreted alongside the group sample sizes, particularly for groups with very few observations (e.g., Native American, $n=6$). The FPRs are calculated on the development set using out-of-fold predictions and are compared with the corresponding FPRs from COMPAS's own score.

* **Tuning impact and model selection:** Both tree models benefited from hyperparameter tuning, but Random Forest saw a massive improvement of $+0.043$ (from $0.641$ to $0.684$), which is more than double its fold-to-fold standard deviation ($\pm 0.017$), whereas Decision Tree gained only $+0.005$ (from $0.675$ to $0.680$), falling well within its fold noise ($\pm 0.013$). Tuning fundamentally altered which model is best, moving Random Forest from the worst-performing baseline to the top performer. For the final report, **0.684** (or $68.4\%$ out-of-fold accuracy on the development set, accompanied by the test holdout result once evaluated) should be reported for the best model (Random Forest, trial 20), as it reflects unbiased out-of-fold predictions without data leakage while demonstrating balanced sensitivity gains on the minority reoffending class (Class 1 recall improved to $0.56$).

- **Class performance:** Both tuned models exhibit identical overall accuracy ($0.68$) and Class 0 recall ($0.78$). However, the tuned Random Forest marginally outperforms the tuned Decision Tree on the positive class, achieving a recall of **0.56** ($F_1 = 0.61$) versus **0.55** ($F_1 = 0.61$).

- **Fairness audit:** Evaluated across out-of-fold predictions on the development set, both tuned models substantially reduce the racial disparities present in COMPAS's own proprietary scores. Against COMPAS's African-American FPR of **0.45**, the tuned Random Forest achieves an FPR of **0.29** and the Decision Tree achieves **0.30**. For Caucasian defendants, COMPAS yields an FPR of **0.23**, while the Random Forest achieves **0.15** and the Decision Tree achieves **0.16**. While a disparity remains between African-American and Caucasian groups ($\Delta \text{FPR} \approx 0.14$), it represents a marked improvement over COMPAS's gap ($\Delta \text{FPR} = 0.22$). Sample sizes for Asian ($n=21$) and Native American ($n=6$) remain too small for statistically reliable standalone conclusions.

- **Ensemble comparisons (Week 6 — AdaBoost vs. Oblique Random Forest):**
  - **AdaBoost:** Demonstrates superior generalization stability, maintaining a negligible train–validation gap (**+0.006 ± 0.015**), confirming that shallow decision stumps (`max_depth=1`) effectively prevent variance expansion. Tuning led to minor refinement in CV accuracy ($0.676$) while enhancing the out-of-fold positive class sensitivity to **0.55** ($F_1 = 0.61$).
  - **Oblique Random Forest:** Showed persistent overfitting due to the increased capacity of multivariate projection splits, exhibiting a wide train–validation gap (**+0.111 ± 0.010**, train: $0.757$, validation: $0.646$). Although tuning improved aggregate out-of-fold accuracy to $0.680$, this gain occurred entirely by prioritizing the majority class (Class 0 recall rose to **0.80**, while Class 1 recall dropped to **0.53**). Consequently, oblique projection trees offer no empirical advantage over standard axis-aligned Random Forests on this tabular feature space.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
