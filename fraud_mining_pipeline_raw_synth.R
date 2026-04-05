#############################################################
# Data Mining
# Fraud Detection Project
#############################################################

rm(list = ls())

# FALSE to keep Risk_Score in dataset
DROP_RISK_SCORE <- TRUE
# FALSE to keep Failed_Transaction_Count_7d in dataset
DROP_FAILED_TXN_COUNT_7D <- FALSE

#####################
# Required Packages #
#####################

library(ggplot2)
library(dplyr)
library(rlang)
library(doBy)
library(rpart)
library(rpart.plot)
library(class)
library(caret)
library(randomForest)
library(klaR)
library(pROC)
library(corrplot)
library(xgboost)
library(glmnet)
library(Rtsne)
library(uwot)

################
# Data Loading #
################

script_dir <- getwd()
data_path <- "data/raw/synthetic_raw_cleaned.csv"
fraud.data <- read.csv(data_path, na.strings = c("", "NA", "NULL"))

output_dir <- file.path(script_dir, "output")
dir.create(output_dir, recursive = TRUE)
plots_dir <- file.path(output_dir, "plots")
dir.create(plots_dir, recursive = TRUE)

if (isTRUE(DROP_RISK_SCORE) && "Risk_Score" %in% names(fraud.data)) {
  fraud.data$Risk_Score <- NULL
}

if (isTRUE(DROP_FAILED_TXN_COUNT_7D) && "Failed_Transaction_Count_7d" %in% names(fraud.data)) {
  fraud.data$Failed_Transaction_Count_7d <- NULL
}

cols <- c(
  "Transaction_Type",
  "Device_Type",
  "Location",
  "Merchant_Category",
  "IP_Address_Flag",
  "Previous_Fraudulent_Activity",
  "Card_Type",
  "Authentication_Method",
  "Is_Weekend",
  "Fraud_Label"
)

fraud.data <- fraud.data %>%
  mutate(
    target = factor(ifelse(Fraud_Label == 1, "FRAUD", "LEGIT"),
                    levels = c("LEGIT", "FRAUD")),
    across(where(is.character), as.factor)
  )

####################
# Data Exploration #
####################

head(fraud.data)
str(fraud.data)

print(table(fraud.data$Fraud_Label))
print(round(prop.table(table(fraud.data$Fraud_Label)) * 100, 2))

for (col in cols) {
  print(dplyr::count(fraud.data, !!rlang::sym(col), sort = TRUE))
}

print(summary(fraud.data$Transaction_Amount))
print(summary(fraud.data$Timestamp))
print(summary(fraud.data$Account_Balance))
print(summary(fraud.data$Daily_Transaction_Count))
print(summary(fraud.data$Avg_Transaction_Amount_7d))
print(summary(fraud.data$Failed_Transaction_Count_7d))
print(summary(fraud.data$Card_Age))
print(summary(fraud.data$Transaction_Distance))
print(summary(fraud.data$Failed_Transaction_Count_7d))

print(sum(duplicated(fraud.data$Transaction_ID)))

na_count <- sum(is.na(fraud.data))
print(paste("Total NA:", na_count))

fraud.data %>%
  group_by(Failed_Transaction_Count_7d) %>%
  summarise(
    count = n(),
    fraud_rate = mean(Fraud_Label)
  ) %>%
  arrange(Failed_Transaction_Count_7d) %>%
  mutate(
    fraud_rate = round(fraud_rate * 100, 1)
  ) %>%
  print()

unique(fraud.data$Failed_Transaction_Count_7d)

##############################
#   Feature Enineering       #
##############################

if (FALSE) {
  build_features <- function(df, night_hours = c(1, 2, 3, 4, 5)) {
    df %>%
      mutate(row_id = row_number()) %>%
      group_by(User_ID) %>%
      group_modify(~{
        g <- .x %>% arrange(row_id)
        n <- nrow(g)
        
        g$is_new_location <- sapply(seq_len(n), function(i) {
          if (i == 1) 0 else as.integer(!(g$Location[i] %in% g$Location[1:(i - 1)]))
        })
        g$is_new_device <- sapply(seq_len(n), function(i) {
          if (i == 1) 0 else as.integer(!(g$Device_Type[i] %in% g$Device_Type[1:(i - 1)]))
        })
        g$is_new_card <- sapply(seq_len(n), function(i) {
          if (i == 1) 0 else as.integer(!(g$Card_Type[i] %in% g$Card_Type[1:(i - 1)]))
        })
        g$is_new_txn_type <- sapply(seq_len(n), function(i) {
          if (i == 1) 0 else as.integer(!(g$Transaction_Type[i] %in% g$Transaction_Type[1:(i - 1)]))
        })
        g$is_location_in_top2_history <- sapply(seq_len(n), function(i) {
          if (i == 1) {
            0
          } else {
            prev_locs <- g$Location[1:(i - 1)]
            if (length(prev_locs) == 0) {
              0
            } else {
              loc_counts <- sort(table(prev_locs), decreasing = TRUE)
              top2_locs <- names(loc_counts)[1:min(2, length(loc_counts))]
              as.integer(g$Location[i] %in% top2_locs)
            }
          }
        })
        g$is_night_txn <- ifelse(g$Timestamp %in% night_hours, 1, 0)
        g
      }) %>%
      ungroup() %>%
      arrange(row_id) %>%
      select(-row_id)
  }
}

##############################
#   Heatmaps                 #
##############################

# Correlation matrix
eda_numeric <- fraud.data[, sapply(fraud.data, is.numeric), drop = FALSE]
eda_numeric <- eda_numeric[, setdiff(names(eda_numeric), "Fraud_Label"), drop = FALSE]
eda_numeric_sd <- sapply(eda_numeric, sd, na.rm = TRUE)
eda_numeric <- eda_numeric[, eda_numeric_sd > 0 & !is.na(eda_numeric_sd), drop = FALSE]
eda_corr <- cor(eda_numeric, use = "pairwise.complete.obs")
pdf(file.path(plots_dir, "numeric_corrplot.pdf"), width = 11, height = 8.5)
corrplot::corrplot(
  eda_corr,
  method = "color",
  type = "upper",
  tl.cex = 0.8,
  addCoef.col = "black",
  number.cex = 0.7
)
dev.off()

# Pearson for features and Fraud_Label
y_fl <- fraud.data$Fraud_Label
feat_names <- names(eda_numeric)
r_tgt <- rep(NA_real_, length(feat_names))
names(r_tgt) <- feat_names
for (nm in feat_names) {
  r_tgt[nm] <- cor(eda_numeric[[nm]], y_fl, use = "pairwise.complete.obs")
}
corr_tgt_df <- data.frame(feature = feat_names, r = as.numeric(r_tgt), stringsAsFactors = FALSE)
corr_tgt_df$feature <- reorder(corr_tgt_df$feature, corr_tgt_df$r)

pdf(file.path(plots_dir, "numeric_corr_with_target.pdf"), width = 11, height = 8.5)
print(
  ggplot(corr_tgt_df, aes(x = r, y = feature, fill = r > 0)) +
    geom_col(width = 0.7) +
    geom_vline(xintercept = 0, linewidth = 0.3, color = "gray40") +
    scale_fill_manual(values = c(`FALSE` = "#e07a7a", `TRUE` = "#2ca3a3"), guide = "none") +
    labs(x = "r", y = NULL, title = "Numeric correlation with target") +
    theme_bw() +
    theme(panel.grid.major.y = element_blank())
)
dev.off()

###############################
# PCA                         #
###############################

numeric_cols <- names(fraud.data)[sapply(fraud.data, is.numeric)]
numeric_cols <- setdiff(numeric_cols, "Fraud_Label")

fraud.numeric <- fraud.data[, numeric_cols]

set.seed(2568)
pairs_sample_idx <- sample(seq_len(nrow(fraud.numeric)), min(1500, nrow(fraud.numeric)))
pdf(file.path(plots_dir, "pairs_numeric.pdf"), width = 11, height = 8.5)
pairs(fraud.numeric[pairs_sample_idx, ], pch = 19, cex = 0.4)
dev.off()

pca_prcomp <- prcomp(fraud.numeric, scale. = TRUE)
print(summary(pca_prcomp))

pdf(file.path(plots_dir, "pca_scree_plot.pdf"), width = 11, height = 8.5)
explained_var <- (pca_prcomp$sdev^2) / sum(pca_prcomp$sdev^2)
plot(
  explained_var,
  type = "b",
  pch = 19,
  xlab = "Principal Component",
  ylab = "Explained Variance Ratio",
  main = "PCA Scree Plot"
)
dev.off()

pdf(file.path(plots_dir, "pca_pc1_pc2_by_target.pdf"), width = 11, height = 8.5)
point_cols <- ifelse(fraud.data$target == "FRAUD", "red3", "darkgreen")
plot(
  pca_prcomp$x[, 1],
  pca_prcomp$x[, 2],
  col = grDevices::adjustcolor(point_cols, alpha.f = 0.35),
  pch = 16,
  xlab = "PC1",
  ylab = "PC2",
  main = "PCA: PC1 vs PC2"
)
legend(
  "topright",
  legend = c("LEGIT", "FRAUD"),
  col = c("darkgreen", "red3"),
  pch = 16,
  bty = "n"
)
dev.off()

pdf(file.path(plots_dir, "pca_pc1_pc3_by_target.pdf"), width = 11, height = 8.5)
point_cols <- ifelse(fraud.data$target == "FRAUD", "red3", "darkgreen")
plot(
  pca_prcomp$x[, 1],
  pca_prcomp$x[, 3],
  col = grDevices::adjustcolor(point_cols, alpha.f = 0.35),
  pch = 16,
  xlab = "PC1",
  ylab = "PC3",
  main = "PCA: PC1 vs PC3"
)
legend(
  "topright",
  legend = c("LEGIT", "FRAUD"),
  col = c("darkgreen", "red3"),
  pch = 16,
  bty = "n"
)
dev.off()

#########################
# K-means            #
#########################

fraud.zs <- as.data.frame(scale(fraud.numeric))

set.seed(2568)
cl <- kmeans(fraud.zs, centers = 2, nstart = 20)
cat("\nKMeans summary\n")
cat("Cluster sizes:\n")
print(cl$size)
cat("Cluster centers:\n")
print(round(cl$centers, 3))
cat("Total withinss:", round(cl$tot.withinss, 3), "\n")
cat("Between/Total SS:", round(cl$betweenss / cl$totss, 3), "\n")
cat("True class vs cluster:\n")
print(table(TrueClass = fraud.data$target, Cluster = cl$cluster))

pdf(file.path(plots_dir, "kmeans_clusters_scaled.pdf"), width = 11, height = 8.5)
km_cols <- c("#377eb8", "#e41a1c")
plot(
  pca_prcomp$x[, 1],
  pca_prcomp$x[, 2],
  col = km_cols[cl$cluster],
  pch = 19,
  main = "KMeans (k=2) — PC1 vs PC2",
  xlab = "PC1",
  ylab = "PC2"
)
legend("topright", legend = c("Cluster 1", "Cluster 2"), col = km_cols, pch = 19, bty = "n")
dev.off()

set.seed(2568)
hc_idx <- sample(seq_len(nrow(fraud.zs)), min(1000, nrow(fraud.zs)))
hc <- hclust(dist(fraud.zs[hc_idx, ]), method = "ave")
pdf(file.path(plots_dir, "hclust_sample.pdf"), width = 11, height = 8.5)
plot(hc, main = "Hierarchical Clustering")
dev.off()

####################################
# Class weightening and resampling #
####################################

model_df <- fraud.data %>%
  dplyr::select(-Transaction_ID, -User_ID, -Fraud_Label)

set.seed(2568)
train_main_idx <- caret::createDataPartition(model_df$target, p = 0.6, list = FALSE)
train_df <- model_df[train_main_idx, ]
tmp_df <- model_df[-train_main_idx, ]

val_idx <- caret::createDataPartition(tmp_df$target, p = 0.5, list = FALSE)
val_df <- tmp_df[val_idx, ]
test_df <- tmp_df[-val_idx, ]

class_weights <- c(LEGIT = 1, FRAUD = 2)
train_case_weights <- class_weights[as.character(train_df$target)]

# resampling for methods without native weights
set.seed(2568)
RESAMPLE_FRAUD_WEIGHT <- 3
resample_prob <- ifelse(train_df$target == "FRAUD", RESAMPLE_FRAUD_WEIGHT, 1)
resample_idx <- sample(
  seq_len(nrow(train_df)),
  size = nrow(train_df),
  replace = TRUE,
  prob = resample_prob
)
train_resampled <- train_df[resample_idx, ]

cat("\nClass proportions before resampling:\n")
print(prop.table(table(train_df$target)))

cat("\nClass proportions after resampling:\n")
print(prop.table(table(train_resampled$target)))

cat("\nClass counts before resampling:\n")
print(table(train_df$target))

cat("\nClass counts after resampling:\n")
print(table(train_resampled$target))

fraud_label <- "FRAUD"
threshold_grid <- seq(0.05, 0.95, by = 0.05)

#####################
# Service functions #
#####################

compute_metrics <- function(y_true, y_prob, threshold = 0.5, positive = "FRAUD") {
  y_true <- factor(y_true, levels = c("LEGIT", "FRAUD"))
  y_pred <- factor(ifelse(y_prob >= threshold, positive, "LEGIT"), levels = c("LEGIT", "FRAUD"))
  tp <- sum(y_pred == positive & y_true == positive)
  fp <- sum(y_pred == positive & y_true != positive)
  fn <- sum(y_pred != positive & y_true == positive)
  tn <- sum(y_pred != positive & y_true != positive)
  recall <- tp / (tp + fn)
  precision <- tp / (tp + fp)
  f1 <- 2 * recall * precision / (recall + precision)
  beta <- 2
  f2 <- (1 + beta^2) * recall * precision / (beta^2 * precision + recall)
  data.frame(
    recall = recall,
    precision = precision,
    f1 = f1,
    f2 = f2,
    tp = tp,
    fp = fp,
    fn = fn,
    tn = tn
  )
}

print_confusion_matrix <- function(y_true, y_prob, threshold, model_name, dataset_name, positive = "FRAUD") {
  y_true <- factor(y_true, levels = c("LEGIT", "FRAUD"))
  y_pred <- factor(ifelse(y_prob >= threshold, positive, "LEGIT"), levels = c("LEGIT", "FRAUD"))
  cm <- table(True = y_true, Pred = y_pred)
  print(paste("Confusion matrix:", model_name, "-", dataset_name, "(threshold =", round(threshold, 3), ")"))
  print(cm)
}

make_roc_df <- function(y_true, y_prob, dataset_name, model_name, positive = "FRAUD") {
  y_num <- ifelse(as.character(y_true) == positive, 1, 0)
  roc_obj <- pROC::roc(response = y_num, predictor = y_prob, direction = "<")
  data.frame(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    dataset = dataset_name,
    model = model_name,
    auc = as.numeric(pROC::auc(roc_obj))
  )
}

make_f1_threshold_df <- function(y_true, y_prob, positive = "FRAUD") {
  out <- data.frame(threshold = numeric(0), f1 = numeric(0))
  for (th in threshold_grid) {
    m <- compute_metrics(y_true, y_prob, threshold = th, positive = positive)
    out <- rbind(out, data.frame(threshold = th, f1 = m$f1))
  }
  out
}

#################
# Decision Tree #
#################

tree_model <- rpart(
  target ~ .,
  data = train_df,
  method = "class",
  weights = train_case_weights,
  control = rpart.control(
    cp = 0.001,
    minsplit = 10,
    minbucket = 5,
    maxdepth = 5
  )
)
pdf(file.path(plots_dir, "rpart_tree.pdf"), width = 11, height = 8.5)
rpart.plot(tree_model)
dev.off()

pdf(file.path(plots_dir, "rpart_cp_plot.pdf"), width = 11, height = 8.5)
plotcp(tree_model)
dev.off()

tree_prob_train <- predict(tree_model, newdata = train_df, type = "prob")[, fraud_label]
tree_prob_val <- predict(tree_model, newdata = val_df, type = "prob")[, fraud_label]
tree_prob_test <- predict(tree_model, newdata = test_df, type = "prob")[, fraud_label]

########
# KNN  #
########

knn_predictor_cols <- setdiff(names(train_df), "target")
knn_formula <- as.formula(paste("~", paste(knn_predictor_cols, collapse = " + ")))
knn_dummies <- caret::dummyVars(knn_formula, data = train_resampled, fullRank = FALSE)

knn_x_train <- as.data.frame(predict(knn_dummies, newdata = train_resampled))
knn_x_train_eval <- as.data.frame(predict(knn_dummies, newdata = train_df))
knn_x_val <- as.data.frame(predict(knn_dummies, newdata = val_df))
knn_x_test <- as.data.frame(predict(knn_dummies, newdata = test_df))

knn_pp <- caret::preProcess(knn_x_train, method = c("center", "scale"))
knn_x_train <- as.matrix(predict(knn_pp, knn_x_train))
knn_x_train_eval <- as.matrix(predict(knn_pp, knn_x_train_eval))
knn_x_val <- as.matrix(predict(knn_pp, knn_x_val))
knn_x_test <- as.matrix(predict(knn_pp, knn_x_test))

knn_y_train <- train_resampled$target

pred_knn_train <- knn(train = knn_x_train, test = knn_x_train_eval, cl = knn_y_train, k = 10, prob = TRUE)
knn_prob_train_raw <- attr(pred_knn_train, "prob")
knn_prob_train <- ifelse(pred_knn_train == fraud_label, knn_prob_train_raw, 1 - knn_prob_train_raw)

pred_knn_val <- knn(train = knn_x_train, test = knn_x_val, cl = knn_y_train, k = 10, prob = TRUE)
knn_prob_val_raw <- attr(pred_knn_val, "prob")
knn_prob_val <- ifelse(pred_knn_val == fraud_label, knn_prob_val_raw, 1 - knn_prob_val_raw)

pred_knn_test <- knn(train = knn_x_train, test = knn_x_test, cl = knn_y_train, k = 10, prob = TRUE)
knn_prob_test_raw <- attr(pred_knn_test, "prob")
knn_prob_test <- ifelse(pred_knn_test == fraud_label, knn_prob_test_raw, 1 - knn_prob_test_raw)

#################################
# Random Forest                 #
#################################

fit_rf <- randomForest(
  target ~ .,
  data = train_df,
  classwt = class_weights,
  ntree = 200,
  mtry = max(2, floor(sqrt(ncol(train_df) - 1))),
  nodesize = 10,
  maxnodes = 30
)

rf_prob_train <- predict(fit_rf, newdata = train_df, type = "prob")[, fraud_label]
rf_prob_val <- predict(fit_rf, newdata = val_df, type = "prob")[, fraud_label]
rf_prob_test <- predict(fit_rf, newdata = test_df, type = "prob")[, fraud_label]

roc_rf <- rbind(
  make_roc_df(train_df$target, rf_prob_train, "train", "rf", fraud_label),
  make_roc_df(val_df$target, rf_prob_val, "val", "rf", fraud_label),
  make_roc_df(test_df$target, rf_prob_test, "test", "rf", fraud_label)
)
p_roc_rf <- ggplot(roc_rf, aes(x = fpr, y = tpr, color = dataset)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_bw() +
  ggtitle("Random Forest - ROC (Train/Val/Test)") +
  xlab("False Positive Rate") +
  ylab("True Positive Rate")
pdf(file.path(plots_dir, "qc_rf_roc.pdf"), width = 11, height = 8.5)
print(p_roc_rf)
dev.off()

###############
# Naive Bayes #
###############

fit_nb <- NaiveBayes(target ~ ., data = train_resampled)
nb_prob_train <- as.numeric(predict(fit_nb, newdata = train_df)$posterior[, fraud_label])
nb_prob_val <- as.numeric(predict(fit_nb, newdata = val_df)$posterior[, fraud_label])
nb_prob_test <- as.numeric(predict(fit_nb, newdata = test_df)$posterior[, fraud_label])

###############
# XGBoost     #
###############

x_train <- knn_x_train_eval
x_val <- knn_x_val
x_test <- knn_x_test
y_train_num <- ifelse(train_df$target == fraud_label, 1, 0)

dtrain <- xgboost::xgb.DMatrix(data = x_train, label = y_train_num)
dval <- xgboost::xgb.DMatrix(data = x_val)
dtest <- xgboost::xgb.DMatrix(data = x_test)

xgb_scale_pos_weight <- sum(y_train_num == 0) / sum(y_train_num == 1)

fit_xgb <- xgboost::xgb.train(
  params = list(
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.05,
    max_depth = 3,
    subsample = 0.7,
    colsample_bytree = 0.7,
    min_child_weight = 5,
    gamma = 0.2,
    scale_pos_weight = xgb_scale_pos_weight
  ),
  data = dtrain,
  nrounds = 120,
  verbose = 0
)

xgb_prob_train <- as.numeric(predict(fit_xgb, dtrain))
xgb_prob_val <- as.numeric(predict(fit_xgb, dval))
xgb_prob_test <- as.numeric(predict(fit_xgb, dtest))

roc_xgb <- rbind(
  make_roc_df(train_df$target, xgb_prob_train, "train", "xgboost", fraud_label),
  make_roc_df(val_df$target, xgb_prob_val, "val", "xgboost", fraud_label),
  make_roc_df(test_df$target, xgb_prob_test, "test", "xgboost", fraud_label)
)
p_roc_xgb <- ggplot(roc_xgb, aes(x = fpr, y = tpr, color = dataset)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  theme_bw() +
  ggtitle("XGBoost - ROC (Train/Val/Test)") +
  xlab("False Positive Rate") +
  ylab("True Positive Rate")
pdf(file.path(plots_dir, "qc_xgboost_roc.pdf"), width = 11, height = 8.5)
print(p_roc_xgb)
dev.off()

#############
# GLMNET    #
#############

ctrl_glmnet <- trainControl(method = "cv", number = 5, classProbs = TRUE)
fit_glmnet <- caret::train(
  target ~ .,
  data = train_df,
  method = "glmnet",
  family = "binomial",
  preProcess = c("center", "scale"),
  metric = "Accuracy",
  trControl = ctrl_glmnet
)

glmnet_prob_train <- as.numeric(predict(fit_glmnet, newdata = train_df, type = "prob")[[fraud_label]])
glmnet_prob_val <- as.numeric(predict(fit_glmnet, newdata = val_df, type = "prob")[[fraud_label]])
glmnet_prob_test <- as.numeric(predict(fit_glmnet, newdata = test_df, type = "prob")[[fraud_label]])

#####################
# Models Comparison  #
#####################

models_probs <- list(
  rpart = list(train = tree_prob_train, val = tree_prob_val, test = tree_prob_test),
  knn = list(train = knn_prob_train, val = knn_prob_val, test = knn_prob_test),
  rf = list(train = rf_prob_train, val = rf_prob_val, test = rf_prob_test),
  nb = list(train = nb_prob_train, val = nb_prob_val, test = nb_prob_test),
  xgboost = list(train = xgb_prob_train, val = xgb_prob_val, test = xgb_prob_test),
  glmnet = list(train = glmnet_prob_train, val = glmnet_prob_val, test = glmnet_prob_test)
)

f1_rf_tr <- make_f1_threshold_df(train_df$target, rf_prob_train, fraud_label)
f1_rf_tr$dataset <- "train"
f1_rf_va <- make_f1_threshold_df(val_df$target, rf_prob_val, fraud_label)
f1_rf_va$dataset <- "val"
f1_rf_te <- make_f1_threshold_df(test_df$target, rf_prob_test, fraud_label)
f1_rf_te$dataset <- "test"
f1_rf_all <- rbind(f1_rf_tr, f1_rf_va, f1_rf_te)
p_f1_rf <- ggplot(f1_rf_all, aes(x = threshold, y = f1, color = dataset)) +
  geom_line(linewidth = 0.9) +
  theme_bw() +
  ggtitle("Random Forest - F1 vs Threshold") +
  xlab("Threshold") +
  ylab("F1-score")
pdf(file.path(plots_dir, "qc_rf_f1_threshold.pdf"), width = 11, height = 8.5)
print(p_f1_rf)
dev.off()

f1_glm_tr <- make_f1_threshold_df(train_df$target, glmnet_prob_train, fraud_label)
f1_glm_tr$dataset <- "train"
f1_glm_va <- make_f1_threshold_df(val_df$target, glmnet_prob_val, fraud_label)
f1_glm_va$dataset <- "val"
f1_glm_te <- make_f1_threshold_df(test_df$target, glmnet_prob_test, fraud_label)
f1_glm_te$dataset <- "test"
f1_glm_all <- rbind(f1_glm_tr, f1_glm_va, f1_glm_te)
p_f1_glm <- ggplot(f1_glm_all, aes(x = threshold, y = f1, color = dataset)) +
  geom_line(linewidth = 0.9) +
  theme_bw() +
  ggtitle("glmnet - F1 vs Threshold") +
  xlab("Threshold") +
  ylab("F1-score")
pdf(file.path(plots_dir, "qc_glmnet_f1_threshold.pdf"), width = 11, height = 8.5)
print(p_f1_glm)
dev.off()

metrics_rows <- list()

for (model_name in names(models_probs)) {
  p_train <- models_probs[[model_name]]$train
  p_val <- models_probs[[model_name]]$val
  p_test <- models_probs[[model_name]]$test

  roc_train <- make_roc_df(train_df$target, p_train, "train", model_name, fraud_label)
  roc_val <- make_roc_df(val_df$target, p_val, "val", model_name, fraud_label)
  roc_test <- make_roc_df(test_df$target, p_test, "test", model_name, fraud_label)

  f1_on_val <- sapply(threshold_grid, function(th) {
    compute_metrics(val_df$target, p_val, threshold = th, positive = fraud_label)$f1
  })
  f1_on_val[is.na(f1_on_val)] <- -Inf
  best_threshold <- threshold_grid[which.max(f1_on_val)]

  m_train <- compute_metrics(train_df$target, p_train, threshold = best_threshold, positive = fraud_label)
  m_val <- compute_metrics(val_df$target, p_val, threshold = best_threshold, positive = fraud_label)
  m_test <- compute_metrics(test_df$target, p_test, threshold = best_threshold, positive = fraud_label)

  print_confusion_matrix(train_df$target, p_train, best_threshold, model_name, "train", fraud_label)
  print_confusion_matrix(val_df$target, p_val, best_threshold, model_name, "val", fraud_label)
  print_confusion_matrix(test_df$target, p_test, best_threshold, model_name, "test", fraud_label)

  metrics_rows[[length(metrics_rows) + 1]] <- data.frame(
    Model = model_name,
    Dataset = "train",
    ROC_AUC = roc_train$auc[1],
    Best_Threshold_Val = best_threshold,
    Recall = m_train$recall,
    Precision = m_train$precision,
    F1 = m_train$f1,
    F2 = m_train$f2
  )
  metrics_rows[[length(metrics_rows) + 1]] <- data.frame(
    Model = model_name,
    Dataset = "val",
    ROC_AUC = roc_val$auc[1],
    Best_Threshold_Val = best_threshold,
    Recall = m_val$recall,
    Precision = m_val$precision,
    F1 = m_val$f1,
    F2 = m_val$f2
  )
  metrics_rows[[length(metrics_rows) + 1]] <- data.frame(
    Model = model_name,
    Dataset = "test",
    ROC_AUC = roc_test$auc[1],
    Best_Threshold_Val = best_threshold,
    Recall = m_test$recall,
    Precision = m_test$precision,
    F1 = m_test$f1,
    F2 = m_test$f2
  )
}

perf <- dplyr::bind_rows(metrics_rows)
perf <- perf %>%
  dplyr::mutate(
    ROC_AUC = round(ROC_AUC, 4),
    Best_Threshold_Val = round(Best_Threshold_Val, 3),
    Recall = round(Recall, 4),
    Precision = round(Precision, 4),
    F1 = round(F1, 4),
    F2 = round(F2, 4)
  )

write.csv(perf, file.path(output_dir, "model_performance.csv"), row.names = FALSE)

print("Fraud mining completed")