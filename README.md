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

| Model                | Holdout accuracy (W3) | CV accuracy (mean ± std) | CV train–val gap |
|----------------------|-----------------------:|--------------------------:|-----------------:|
| Dummy (majority)     | 0.550                  | 0.549 ± 0.000             | -0.000           |
| Logistic regression  | 0.657                  | 0.673 ± 0.013             | +0.002           |
| Decision tree        | 0.665                  | 0.675 ± 0.013             | +0.010           |
| Random forest        | 0.655                  | 0.641 ± 0.017             | +0.097           |


### Analysis

* **Model comparison:** The current cross-validation results show very similar validation accuracy for Logistic Regression and Decision Tree (**0.673 ± 0.013** and **0.675 ± 0.013**, respectively). The Decision Tree has a slightly higher mean CV accuracy, but the difference is very small. Therefore, the current results do not indicate a clear performance separation between these two models.

* **Generalization:** Logistic Regression has a smaller mean train-validation gap (**+0.002**) than the Decision Tree (**+0.010**), indicating a smaller difference between training and validation performance in the current 5-fold cross-validation.

* **Random Forest:** The Random Forest obtained a lower mean validation accuracy (**0.641 ± 0.017**) and a substantially larger train-validation gap (**+0.097**), indicating a larger discrepancy between training and validation performance in the current configuration.

* **Dummy baseline:** The Dummy classifier achieved **0.549 ± 0.000** accuracy by predicting the majority class. This provides a baseline against which the other models can be compared. Both Logistic Regression and Decision Tree substantially exceed this baseline in cross-validation.

* **Class performance:** For the Logistic Regression, Class 0 has recall of **0.78**, compared with **0.54** for Class 1. For the Decision Tree, the corresponding recalls are **0.78** and **0.54**. Both models therefore show different recall levels across the two classes, despite having similar overall accuracy.

* **Fairness audit:** The fairness analysis reports the false positive rate (FPR) separately by race. The results should be interpreted alongside the group sample sizes, particularly for groups with very few observations (e.g., Native American, $n=6$). The FPRs are calculated on the development set using out-of-fold predictions and are compared with the corresponding FPRs from COMPAS's own score.

* **Impact of Week 3 preprocessing:** Test accuracy dropped slightly across both models compared to the naive Week 2 baseline ($-0.003$ for DT, $-0.023$ for LR). This is consistent with the change from complete-case evaluation to a more comprehensive preprocessing pipeline:
  1. The Week 2 naive baseline evaluated only on rows surviving complete-case deletion (`dropna()`), potentially producing a different and less representative evaluation sample.
  2. Introducing MNAR indicators and categorical transformations changes the feature representation and can affect model performance.
  3. The current pipeline enforces strict leak-safe preprocessing boundaries, prioritizing a more robust evaluation methodology over potentially optimistic baseline results.
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
