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

- **AUC: 0.807** on a held-out 20% test set — solid discrimination.
- The dataset is heavily imbalanced (~5% stroke cases), so the default 0.5 classification threshold catches almost no real cases (sensitivity 0.029) despite 96% accuracy — accuracy is the wrong metric here.
- The script sweeps thresholds from 0.5 down to 0.05: at 0.10, sensitivity rises to 0.43 (specificity 0.89); at 0.05, sensitivity reaches 0.60 (specificity 0.78). The right operating point is a clinical tradeoff (missed stroke vs. unnecessary follow-up), not a modeling choice.
- Statistically significant predictors (p < 0.05): age, hypertension, average glucose level, and smoking status ("smokes" category). Heart disease is marginal (p = 0.07); BMI and gender are not significant in this model.

## Next steps

- Try a model that handles imbalance directly (e.g., class weighting, SMOTE) instead of just threshold-shifting
- Cross-validate rather than a single train/test split
- Compare against a non-linear model (random forest, gradient boosting) to see if AUC improves
