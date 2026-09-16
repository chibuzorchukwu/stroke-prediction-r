#!/usr/bin/env Rscript
# Stroke prediction: logistic regression (base R) vs. random forest.
# Data: data/healthcare-dataset-stroke-data.csv (public Kaggle dataset, via public GitHub mirror).

.libPaths(c("~/R/library", .libPaths()))
suppressPackageStartupMessages(library(randomForest))

set.seed(42)

df <- read.csv("data/healthcare-dataset-stroke-data.csv", stringsAsFactors = FALSE)

cat("Loaded", nrow(df), "rows,", ncol(df), "columns\n")
cat("Stroke cases:", sum(df$stroke == 1), "/", nrow(df),
    sprintf("(%.1f%%)\n", 100 * mean(df$stroke == 1)))

## --- Cleaning ---
df$bmi <- suppressWarnings(as.numeric(df$bmi))  # "N/A" -> NA
df <- df[!is.na(df$bmi), ]
df <- df[df$gender != "Other", ]  # drop the single "Other" row; too few to model

df$gender <- factor(df$gender)
df$ever_married <- factor(df$ever_married)
df$work_type <- factor(df$work_type)
df$residence_type <- factor(df$residence_type)
df$smoking_status <- factor(df$smoking_status)
df$stroke <- as.integer(df$stroke)

cat("After cleaning:", nrow(df), "rows\n\n")

## --- Train/test split (80/20) ---
n <- nrow(df)
train_idx <- sample(seq_len(n), size = floor(0.8 * n))
train <- df[train_idx, ]
test <- df[-train_idx, ]

## --- Fit logistic regression ---
model <- glm(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level + bmi +
    gender + ever_married + work_type + residence_type + smoking_status,
  data = train,
  family = binomial(link = "logit")
)

cat("=== Model summary ===\n")
print(summary(model))

## --- Evaluate on test set ---
probs <- predict(model, newdata = test, type = "response")
preds <- ifelse(probs >= 0.5, 1, 0)

confusion <- table(Predicted = preds, Actual = test$stroke)
accuracy <- mean(preds == test$stroke)

tp <- sum(preds == 1 & test$stroke == 1)
fn <- sum(preds == 0 & test$stroke == 1)
tn <- sum(preds == 0 & test$stroke == 0)
fp <- sum(preds == 1 & test$stroke == 0)
sensitivity <- if ((tp + fn) > 0) tp / (tp + fn) else NA
specificity <- if ((tn + fp) > 0) tn / (tn + fp) else NA

## Manual AUC (rank-based, no pROC/caret dependency)
compute_auc <- function(probs, actual) {
  pos <- probs[actual == 1]
  neg <- probs[actual == 0]
  if (length(pos) == 0 || length(neg) == 0) return(NA)
  ranks <- rank(c(pos, neg))
  n1 <- length(pos)
  n2 <- length(neg)
  (sum(ranks[1:n1]) - n1 * (n1 + 1) / 2) / (n1 * n2)
}
auc <- compute_auc(probs, test$stroke)

cat("\n=== Test set performance (n =", nrow(test), ") ===\n")
print(confusion)
cat(sprintf("Accuracy:    %.3f\n", accuracy))
cat(sprintf("Sensitivity: %.3f (recall on actual stroke cases)\n", sensitivity))
cat(sprintf("Specificity: %.3f\n", specificity))
cat(sprintf("AUC:         %.3f\n", auc))

cat("\nNote: the dataset is heavily imbalanced (~5% stroke cases), so accuracy alone\n")
cat("is misleading — a model predicting 'no stroke' for everyone would score ~95%\n")
cat("accuracy while catching zero real cases. Sensitivity and AUC matter more here.\n")

## --- Threshold sweep: the model's ranking (AUC 0.81) is decent, the default
## 0.5 cutoff just throws most of it away in a screening context where missing
## a real stroke is far costlier than a false alarm. ---
cat("\n=== Threshold sweep (screening tradeoff) ===\n")
cat(sprintf("%-10s %-10s %-12s %-10s\n", "Threshold", "Accuracy", "Sensitivity", "Specificity"))
for (thresh in c(0.5, 0.3, 0.2, 0.15, 0.1, 0.05)) {
  p <- ifelse(probs >= thresh, 1, 0)
  acc <- mean(p == test$stroke)
  tp <- sum(p == 1 & test$stroke == 1)
  fn <- sum(p == 0 & test$stroke == 1)
  tn <- sum(p == 0 & test$stroke == 0)
  fp <- sum(p == 1 & test$stroke == 0)
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA
  cat(sprintf("%-10.2f %-10.3f %-12.3f %-10.3f\n", thresh, acc, sens, spec))
}
cat("\nLowering the threshold trades false alarms for catching more real cases —\n")
cat("the right tradeoff depends on the cost of a missed stroke vs. an unnecessary\n")
cat("follow-up test, which is a clinical decision, not a modeling one.\n")

## --- Comparison model: random forest ---
## Class-imbalance-aware: sample equal stroke/no-stroke cases per tree via
## `strata`/`sampsize`, which tends to raise sensitivity more than the logistic
## model's raw class frequencies do.
cat("\n=== Comparison: random forest ===\n")

train_rf <- train
train_rf$stroke <- factor(train_rf$stroke, levels = c(0, 1))
n_pos <- sum(train_rf$stroke == 1)

rf_model <- randomForest(
  stroke ~ age + hypertension + heart_disease + avg_glucose_level + bmi +
    gender + ever_married + work_type + residence_type + smoking_status,
  data = train_rf,
  ntree = 500,
  strata = train_rf$stroke,
  sampsize = c(`0` = n_pos, `1` = n_pos)  # balanced sampling per tree
)

rf_probs <- predict(rf_model, newdata = test, type = "prob")[, "1"]
rf_auc <- compute_auc(rf_probs, test$stroke)

cat("\nThreshold sweep (random forest, class-balanced sampling):\n")
cat(sprintf("%-10s %-10s %-12s %-10s\n", "Threshold", "Accuracy", "Sensitivity", "Specificity"))
for (thresh in c(0.5, 0.4, 0.3, 0.2)) {
  p <- ifelse(rf_probs >= thresh, 1, 0)
  acc <- mean(p == test$stroke)
  tp <- sum(p == 1 & test$stroke == 1)
  fn <- sum(p == 0 & test$stroke == 1)
  tn <- sum(p == 0 & test$stroke == 0)
  fp <- sum(p == 1 & test$stroke == 0)
  sens <- if ((tp + fn) > 0) tp / (tp + fn) else NA
  spec <- if ((tn + fp) > 0) tn / (tn + fp) else NA
  cat(sprintf("%-10.2f %-10.3f %-12.3f %-10.3f\n", thresh, acc, sens, spec))
}
cat(sprintf("\nRandom forest AUC: %.3f (logistic regression: %.3f)\n", rf_auc, auc))
cat("\nVariable importance (random forest):\n")
print(importance(rf_model)[order(-importance(rf_model)[, 1]), , drop = FALSE])
