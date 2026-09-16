# Stroke Prediction Model (R)

A logistic regression model predicting stroke risk from clinical/demographic features. Picked up from the "Machine Learning Projects" GitHub board.

## Data

`data/healthcare-dataset-stroke-data.csv` — the public Kaggle "Stroke Prediction Dataset" (fedesoriano), 5,110 patient records, fetched via a public GitHub mirror. 4,908 rows remain after dropping missing BMI values and one `gender == "Other"` row (too few to model).

## Run it

```bash
Rscript stroke_model.R
```

Uses only base R (`glm`) — no packages to install.

## Results

### Logistic regression (base R, no packages)

- **AUC: 0.807** on a held-out 20% test set — solid discrimination.
- The dataset is heavily imbalanced (~5% stroke cases), so the default 0.5 classification threshold catches almost no real cases (sensitivity 0.029) despite 96% accuracy — accuracy is the wrong metric here.
- Threshold sweep from 0.5 down to 0.05: at 0.10, sensitivity rises to 0.43 (specificity 0.89); at 0.05, sensitivity reaches 0.60 (specificity 0.78).
- Statistically significant predictors (p < 0.05): age, hypertension, average glucose level, and smoking status ("smokes" category). Heart disease is marginal (p = 0.07); BMI and gender are not significant in this model.

### Random forest with class-balanced sampling (`randomForest` package)

- **AUC: 0.806** — essentially identical to logistic regression. AUC measures ranking quality, not the imbalance problem, so this is expected.
- But balanced-sampling changes the *practical* tradeoff a lot: at the default 0.5 threshold, sensitivity is **0.60** (vs. 0.029 for logistic regression at the same threshold), at a cost of specificity dropping to 0.82.
- Variable importance (Gini) ranks age, average glucose level, and BMI as the top three predictors — age dominates, consistent with the logistic model's largest, most significant coefficient.
- **Takeaway:** for an imbalanced medical screening problem, *how you sample during training* matters as much as *which model* you pick — the balanced random forest is far more usable out of the box than the same-AUC logistic model, without needing manual threshold tuning.

## Next steps

- Try SMOTE or class-weighted logistic regression as a third comparison point
- Cross-validate rather than a single train/test split
- Try gradient boosting (xgboost) to see if it beats both on AUC, not just usability
