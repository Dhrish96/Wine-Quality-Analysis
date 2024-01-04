rm(list=ls())

# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(caret)


# Read red wine dataset
red_wine <- read.csv("C:/Users/dhris/OneDrive/Desktop/Fall Sem 2023/CDAProject/WINE FILES/winequality-red.csv", header = TRUE, sep = ";")

# Read white wine dataset
white_wine <- read.csv("C:/Users/dhris/OneDrive/Desktop/Fall Sem 2023/CDAProject/WINE FILES/winequality-white.csv", header = TRUE, sep = ";")

# Check column names and structure for red wine dataset
colnames_red_wine <- colnames(red_wine)
str(red_wine)

# Check column names and structure for white wine dataset
colnames_white_wine <- colnames(white_wine)
str(white_wine)

#DATA PREPROCESSING
# Summary statistics for red wine dataset
summary(red_wine)

# Summary statistics for white wine dataset
summary(white_wine)

# Check for missing values for red wine dataset
missing_counts_red_wine <- colSums(is.na(red_wine))
print(missing_counts_red_wine)

# Check for missing values for white wine dataset
missing_counts_white_wine <- colSums(is.na(white_wine))
print(missing_counts_white_wine)

# Check for duplicates in red wine dataset
duplicate_rows_red_wine <- red_wine[duplicated(red_wine), ]
num_duplicates_red_wine <- nrow(duplicate_rows_red_wine)
print(num_duplicates_red_wine)

# Remove duplicates in red wine dataset
red_wine <- unique(red_wine)

# Check for duplicates in white wine dataset
duplicate_rows_white_wine <- white_wine[duplicated(white_wine), ]
num_duplicates_white_wine <- nrow(duplicate_rows_white_wine)
print(num_duplicates_white_wine)

# Remove duplicates in white wine dataset
white_wine <- unique(white_wine)

# Data transformation and normalization - Red Wine
red_wine$quality <- factor(red_wine$quality, levels = c(3:8), labels = c("low", "medium", "high", "very_high", "excellent", "superb"))
#preprocessed_red_wine <- as.data.frame(scale(red_wine[, -ncol(red_wine)]))

# Data transformation and normalization - White Wine
white_wine$quality <- factor(white_wine$quality, levels = c(3:9), labels = c("low", "medium", "high", "very_high", "excellent", "superb", "outstanding"))
#preprocessed_white_wine <- as.data.frame(scale(white_wine[, -ncol(white_wine)]))

# For Red Wine
levels(red_wine$quality)
# For White Wine
levels(white_wine$quality)



#VISUALIZATION
# Visualization and exploration - Red Wine
correlation_matrix_red_wine <- cor(red_wine[, sapply(red_wine, is.numeric)])
corrplot(correlation_matrix_red_wine, method = "color")

correlation_matrix_red_wine

#Second Correlation plot
# Extracting columns of interest
selected_cols <- c("fixed.acidity", "pH", "citric.acid", "density")

# Subsetting the data for selected columns
selected_data <- red_wine[, selected_cols]

# Calculating correlation matrix for the selected columns
correlation_matrix_selected_r <- cor(selected_data)

# Displaying the correlation matrix using corrplot
library(corrplot)
# Displaying the correlation matrix with adjusted font size for numbers inside
corrplot(correlation_matrix_selected_r, method = "number", type = "upper", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.5)


# Visualization and exploration - White Wine
correlation_matrix_white_wine <- cor(white_wine[, sapply(white_wine, is.numeric)])
corrplot(correlation_matrix_white_wine, method = "color")

correlation_matrix_white_wine

#Second Correlation plot
# Extracting columns of interest
selected_cols <- c("fixed.acidity", "density", "residual.sugar", "total.sulfur.dioxide", "free.sulfur.dioxide")

# Subsetting the data for selected columns
selected_data <- red_wine[, selected_cols]

# Calculating correlation matrix for the selected columns
correlation_matrix_selected_w <- cor(selected_data)

# Displaying the correlation matrix using corrplot
# Displaying the correlation matrix with adjusted font size for numbers inside
corrplot(correlation_matrix_selected_w, method = "number", type = "upper", 
         tl.col = "black", tl.srt = 45, tl.cex = 0.5)



#TESTING AND TRAINING
# Split the dataset into training and testing sets - Red Wine
set.seed(123)
train_indices_red_wine <- createDataPartition(red_wine$quality, p = 0.7, list = FALSE)
train_data_red_wine <- red_wine[train_indices_red_wine, ]
test_data_red_wine <- red_wine[-train_indices_red_wine, ]

# Load the necessary libraries
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)

# DECISION TREE - Red Wine
tree_model_red_wine <- rpart(quality ~ ., data = train_data_red_wine, method = "class")
par(mar=c(5, 5, 2, 2))  # Adjust margins
# Plot creation 
fancyRpartPlot(tree_model_red_wine)
# Make predictions on the test set
predictions_tree_red_wine <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "class")
# Calculate accuracy
accuracy_tree_red_wine <- mean(predictions_tree_red_wine == test_data_red_wine$quality)
print(paste("Accuracy of Decision Tree:", round(accuracy_tree_red_wine, 4)))


# RANDOM FOREST - Red Wine
rf_model_red_wine <- randomForest(quality ~ ., data = train_data_red_wine, ntree = 100)
print(rf_model_red_wine)
predictions_red_wine <- predict(rf_model_red_wine, newdata = test_data_red_wine)
accuracy_red_wine <- mean(predictions_red_wine == test_data_red_wine$quality)
print(paste("Accuracy Red Wine:", round(accuracy_red_wine, 4)))

# Split the dataset into training and testing sets - White Wine
set.seed(123)
train_indices_white_wine <- createDataPartition(white_wine$quality, p = 0.7, list = FALSE)
train_data_white_wine <- white_wine[train_indices_white_wine, ]
test_data_white_wine <- white_wine[-train_indices_white_wine, ]

# DECISION TREE - White Wine
tree_model_white_wine <- rpart(quality ~ ., data = train_data_white_wine, method = "class")
fancyRpartPlot(tree_model_white_wine)
predictions_tree_white_wine <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "class")
# Calculate accuracy
accuracy_tree_white_wine <- mean(predictions_tree_white_wine == test_data_white_wine$quality)
print(paste("Accuracy of Decision Tree:", round(accuracy_tree_white_wine, 4)))

# RANDOM FOREST - White Wine
rf_model_white_wine <- randomForest(quality ~ ., data = train_data_white_wine, ntree = 100)
print(rf_model_white_wine)
predictions_white_wine <- predict(rf_model_white_wine, newdata = test_data_white_wine)
accuracy_white_wine <- mean(predictions_white_wine == test_data_white_wine$quality)
print(paste("Accuracy White Wine:", round(accuracy_white_wine, 4)))
                          

#KNN
library(class)

#KNN for RED WINE
k <- 5  # Choose the number of neighbors
predicted_red_wine <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k)
# Calculate accuracy for red wine
accuracy_red_knn <- mean(predicted_red_wine == test_data_red_wine$quality)
print(paste("Accuracy Red Wine (KNN without Normalization):", round(accuracy_red_knn, 4)))

#KNN for WHITE WINE dataset
k <- 5  # Choose the number of neighbors
predicted_white_wine <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k)
# Calculate accuracy for white wine
accuracy_white_knn <- mean(predicted_white_wine == test_data_white_wine$quality)
print(paste("Accuracy White Wine (KNN without Normalization):", round(accuracy_white_knn, 4)))


#OPTIMAL K
# Define a range of k values to test
k_values <- seq(1, 20, by = 2)  # Adjust the range as needed

# Initialize variables to store performance metrics
accuracy_values <- numeric(length(k_values))

# Iterate over each k value
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # Perform k-fold cross-validation
  set.seed(123)  # For reproducibility
  model <- train(x = train_data_red_wine[, -ncol(train_data_red_wine)],
                 y = train_data_red_wine$quality,
                 method = "knn",
                 trControl = trainControl(method = "cv", number = 5),  # 5-fold cross-validation
                 tuneGrid = data.frame(k = k))
  
  # Store the mean accuracy across folds
  accuracy_values[i] <- model$results$Accuracy
}

# Find the optimal k value
optimal_k_red <- k_values[which.max(accuracy_values)]

print(paste("Optimal k for Red Wine (KNN):", optimal_k_red))



# Define a range of k values to test
k_values <- seq(1, 20, by = 2)  # Adjust the range as needed

# Initialize variables to store performance metrics
accuracy_values <- numeric(length(k_values))

# Iterate over each k value
for (i in seq_along(k_values)) {
  k <- k_values[i]
  
  # Perform k-fold cross-validation
  set.seed(123)  # For reproducibility
  model <- train(x = train_data_white_wine[, -ncol(train_data_white_wine)],
                 y = train_data_white_wine$quality,
                 method = "knn",
                 trControl = trainControl(method = "cv", number = 5),  # 5-fold cross-validation
                 tuneGrid = data.frame(k = k))
  
  # Store the mean accuracy across folds
  accuracy_values[i] <- model$results$Accuracy
}

# Find the optimal k value
optimal_k_white <- k_values[which.max(accuracy_values)]

print(paste("Optimal k for White Wine (KNN):", optimal_k_white))


#FINAL KNN 
# Final model with optimal k on the entire training set for Red Wine
final_model_red_wine <- knn(train_data_red_wine[, -ncol(train_data_red_wine)],
                            test_data_red_wine[, -ncol(test_data_red_wine)],
                            train_data_red_wine$quality,
                            k = optimal_k_red)

# Calculate accuracy on the test set for Red Wine
accuracy_final_red_wine <- mean(final_model_red_wine == test_data_red_wine$quality)
print(paste("Final Accuracy on Red Wine Test Set (KNN) with Optimal k:", round(accuracy_final_red_wine, 4)))


# Final model with optimal k on the entire training set for White Wine
final_model <- knn(train_data_white_wine[, -ncol(train_data_white_wine)],
                   test_data_white_wine[, -ncol(test_data_white_wine)],
                   train_data_white_wine$quality,
                   k = optimal_k_white)

# Calculate accuracy on the test set
accuracy_final <- mean(final_model == test_data_white_wine$quality)
print(paste("Final Accuracy on Test Set (KNN) with Optimal k:", round(accuracy_final, 4)))


#NAIVE BAYES
library(e1071)

# Naive Bayes for red wine dataset
nb_model_red_wine <- naiveBayes(quality ~ ., data = train_data_red_wine)
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate accuracy for red wine
accuracy_red_wine_nb <- mean(predictions_red_wine_nb == test_data_red_wine$quality)
print(paste("Accuracy Red Wine (Naive Bayes):", round(accuracy_red_wine_nb, 4)))

# Naive Bayes for white wine dataset
nb_model_white_wine <- naiveBayes(quality ~ ., data = train_data_white_wine)
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate accuracy for white wine
accuracy_white_wine_nb <- mean(predictions_white_wine_nb == test_data_white_wine$quality)
print(paste("Accuracy White Wine (Naive Bayes):", round(accuracy_white_wine_nb, 4)))


#CONFUSION MATRIX
#RED WINE
# Confusion Matrix for Decision Tree - Red Wine
confusionMatrix(predictions_tree_red_wine, test_data_red_wine$quality)

# Confusion Matrix for Random Forest - Red Wine
confusionMatrix(predictions_red_wine, test_data_red_wine$quality)

# Confusion Matrix for KNN - Red Wine
confusionMatrix(predicted_red_wine, test_data_red_wine$quality)

# Confusion Matrix for KNN with Optimal k - Red Wine
predicted_red_optimal <- knn(train_data_red_wine[, -ncol(train_data_red_wine)],
                             test_data_red_wine[, -ncol(test_data_red_wine)],
                             train_data_red_wine$quality,
                             k = optimal_k_red)
confusionMatrix(predicted_red_optimal, test_data_red_wine$quality)

# Confusion Matrix for Naive Bayes - Red Wine
confusionMatrix(predictions_red_wine_nb, test_data_red_wine$quality)



#WHITE WINE
# Confusion Matrix for Decision Tree - White Wine
confusionMatrix(predictions_tree_white_wine, test_data_white_wine$quality)

# Confusion Matrix for Random Forest - White Wine
confusionMatrix(predictions_white_wine, test_data_white_wine$quality)

# Confusion Matrix for KNN - White Wine
confusionMatrix(predicted_white_wine, test_data_white_wine$quality)

# Confusion Matrix for KNN with Optimal k - White Wine
predicted_white_optimal <- knn(train_data_white_wine[, -ncol(train_data_white_wine)],
                               test_data_white_wine[, -ncol(test_data_white_wine)],
                               train_data_white_wine$quality,
                               k = optimal_k_white)
confusionMatrix(predicted_white_optimal, test_data_white_wine$quality)

# Confusion Matrix for Naive Bayes - White Wine
confusionMatrix(predictions_white_wine_nb, test_data_white_wine$quality)


#ROC CURVE
#For RED WINE
#TEST DATA
#LOW
library(ROCR) 
# red wine Decision Tree model
predictions_red_wine_tree <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_tree <- prediction(predictions_red_wine_tree[, "low"], test_data_red_wine$quality == "low")

# Compute performance measures ROC
perf_red_wine_tree <- performance(pred_red_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (Red Wine)")

# red wine Random Forest model
predictions_red_wine_rf <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_rf <- prediction(predictions_red_wine_rf[, "low"], test_data_red_wine$quality == "low")

# Compute performance measures ROC
perf_red_wine_rf <- performance(pred_red_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_rf, col = "red", main = "ROC Curve - Random Forest (Red Wine)")

# red wine KNN model
predicted_red_wine_knn <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)

# Calculate predicted probabilities for "low" class
probabilities_red_wine_knn <- ifelse(predicted_red_wine_knn == "low", 1, 0)

# Create prediction object
pred_red_wine_knn <- prediction(probabilities_red_wine_knn, test_data_red_wine$quality == "low")

# Compute performance measures ROC
perf_red_wine_knn <- performance(pred_red_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_knn, col = "green", main = "ROC Curve - KNN (Red Wine)")

# red wine Naive Bayes model
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate predicted probabilities for "excellent" class
probabilities_red_wine_nb <- ifelse(predictions_red_wine_nb == "low", 1, 0)

# Create prediction object
pred_red_wine_nb <- prediction(probabilities_red_wine_nb, test_data_red_wine$quality == "low")

# Compute performance measures ROC
perf_red_wine_nb <- performance(pred_red_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (Red Wine)")

# Plotting all ROC curves together
plot(perf_red_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_Red Wine")  # Decision Tree
plot(perf_red_wine_rf, col = "red", lwd = 2, add = TRUE)  # Random Forest
plot(perf_red_wine_knn, col = "green", lwd = 2, add = TRUE)  # KNN
plot(perf_red_wine_nb, col = "orange", lwd = 2, add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#MEDIUM
# red wine Decision Tree model
predictions_red_wine_tree <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_tree <- prediction(predictions_red_wine_tree[, "medium"], test_data_red_wine$quality == "medium")

# Compute performance measures ROC
perf_red_wine_tree <- performance(pred_red_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (Red Wine)")

# red wine Random Forest model
predictions_red_wine_rf <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_rf <- prediction(predictions_red_wine_rf[, "medium"], test_data_red_wine$quality == "medium")

# Compute performance measures ROC
perf_red_wine_rf <- performance(pred_red_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_rf, col = "red", main = "ROC Curve - Random Forest (Red Wine)")

# red wine KNN model
predicted_red_wine_knn <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)

# Calculate predicted probabilities for "medium" class
probabilities_red_wine_knn <- ifelse(predicted_red_wine_knn == "medium", 1, 0)

# Create prediction object
pred_red_wine_knn <- prediction(probabilities_red_wine_knn, test_data_red_wine$quality == "medium")

# Compute performance measures ROC
perf_red_wine_knn <- performance(pred_red_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_knn, col = "green", main = "ROC Curve - KNN (Red Wine)")

# red wine Naive Bayes model
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate predicted probabilities for "medium" class
probabilities_red_wine_nb <- ifelse(predictions_red_wine_nb == "medium", 1, 0)

# Create prediction object
pred_red_wine_nb <- prediction(probabilities_red_wine_nb, test_data_red_wine$quality == "medium")

# Compute performance measures  ROC
perf_red_wine_nb <- performance(pred_red_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (Red Wine)")

# Plotting all ROC curves together
plot(perf_red_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_Red Wine")  # Decision Tree
plot(perf_red_wine_rf, col = "red", lwd = 2, add = TRUE)  # Random Forest
plot(perf_red_wine_knn, col = "green", lwd = 2, add = TRUE)  # KNN
plot(perf_red_wine_nb, col = "orange", lwd = 2, add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#HIGH
# red wine Decision Tree model
predictions_red_wine_tree <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_tree <- prediction(predictions_red_wine_tree[, "high"], test_data_red_wine$quality == "high")

# Compute performance measures ROC
perf_red_wine_tree <- performance(pred_red_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (Red Wine)")

# red wine Random Forest model
predictions_red_wine_rf <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_rf <- prediction(predictions_red_wine_rf[, "high"], test_data_red_wine$quality == "high")

# Compute performance measures ROC
perf_red_wine_rf <- performance(pred_red_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_rf, col = "red", main = "ROC Curve - Random Forest (Red Wine)")

# red wine KNN model
predicted_red_wine_knn <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)

# Calculate predicted probabilities for "high" class
probabilities_red_wine_knn <- ifelse(predicted_red_wine_knn == "high", 1, 0)

# Create prediction object
pred_red_wine_knn <- prediction(probabilities_red_wine_knn, test_data_red_wine$quality == "high")

# Compute performance measures ROC
perf_red_wine_knn <- performance(pred_red_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_knn, col = "green", main = "ROC Curve - KNN (Red Wine)")

# red wine Naive Bayes model
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate predicted probabilities for "high" class
probabilities_red_wine_nb <- ifelse(predictions_red_wine_nb == "high", 1, 0)

# Create prediction object
pred_red_wine_nb <- prediction(probabilities_red_wine_nb, test_data_red_wine$quality == "high")

# Compute performance measures  ROC
perf_red_wine_nb <- performance(pred_red_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (Red Wine)")

# Plotting all ROC curves together
plot(perf_red_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_Red Wine")  # Decision Tree
plot(perf_red_wine_rf, col = "red", lwd = 2, add = TRUE)  # Random Forest
plot(perf_red_wine_knn, col = "green", lwd = 2, add = TRUE)  # KNN
plot(perf_red_wine_nb, col = "orange", lwd = 2, add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#VERY HIGH
# red wine Decision Tree model
predictions_red_wine_tree <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_tree <- prediction(predictions_red_wine_tree[, "very_high"], test_data_red_wine$quality == "very_high")

# Compute performance measures ROC
perf_red_wine_tree <- performance(pred_red_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (Red Wine)")

# red wine Random Forest model
predictions_red_wine_rf <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_rf <- prediction(predictions_red_wine_rf[, "very_high"], test_data_red_wine$quality == "very_high")

# Compute performance measures ROC
perf_red_wine_rf <- performance(pred_red_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_rf, col = "red", main = "ROC Curve - Random Forest (Red Wine)")

# red wine KNN model
predicted_red_wine_knn <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)

# Calculate predicted probabilities for "very_high" class
probabilities_red_wine_knn <- ifelse(predicted_red_wine_knn == "very_high", 1, 0)

# Create prediction object
pred_red_wine_knn <- prediction(probabilities_red_wine_knn, test_data_red_wine$quality == "very_high")

# Compute performance measures ROC
perf_red_wine_knn <- performance(pred_red_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_knn, col = "green", main = "ROC Curve - KNN (Red Wine)")

# red wine Naive Bayes model
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate predicted probabilities for "very_high" class
probabilities_red_wine_nb <- ifelse(predictions_red_wine_nb == "very_high", 1, 0)

# Create prediction object
pred_red_wine_nb <- prediction(probabilities_red_wine_nb, test_data_red_wine$quality == "very_high")

# Compute performance measures  ROC
perf_red_wine_nb <- performance(pred_red_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (Red Wine)")

# Plotting all ROC curves together
plot(perf_red_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_Red Wine")  # Decision Tree
plot(perf_red_wine_rf, col = "red", lwd = 2, add = TRUE)  # Random Forest
plot(perf_red_wine_knn, col = "green", lwd = 2, add = TRUE)  # KNN
plot(perf_red_wine_nb, col = "orange", lwd = 2, add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)



#EXCELLENT
# red wine Decision Tree model
predictions_red_wine_tree <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_tree <- prediction(predictions_red_wine_tree[, "excellent"], test_data_red_wine$quality == "excellent")

# Compute performance measures ROC
perf_red_wine_tree <- performance(pred_red_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (Red Wine)")

# red wine Random Forest model
predictions_red_wine_rf <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_rf <- prediction(predictions_red_wine_rf[, "excellent"], test_data_red_wine$quality == "excellent")

# Compute performance measures ROC
perf_red_wine_rf <- performance(pred_red_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_rf, col = "red", main = "ROC Curve - Random Forest (Red Wine)")

# red wine KNN model
predicted_red_wine_knn <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)

# Calculate predicted probabilities for "excellent" class
probabilities_red_wine_knn <- ifelse(predicted_red_wine_knn == "excellent", 1, 0)

# Create prediction object
pred_red_wine_knn <- prediction(probabilities_red_wine_knn, test_data_red_wine$quality == "excellent")

# Compute performance measures ROC
perf_red_wine_knn <- performance(pred_red_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_knn, col = "green", main = "ROC Curve - KNN (Red Wine)")

# red wine Naive Bayes model
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate predicted probabilities for "excellent" class
probabilities_red_wine_nb <- ifelse(predictions_red_wine_nb == "excellent", 1, 0)

# Create prediction object
pred_red_wine_nb <- prediction(probabilities_red_wine_nb, test_data_red_wine$quality == "excellent")

# Compute performance measures  ROC
perf_red_wine_nb <- performance(pred_red_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (Red Wine)")

# Plotting all ROC curves together
plot(perf_red_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_Red Wine")  # Decision Tree
plot(perf_red_wine_rf, col = "red", lwd = 2, add = TRUE)  # Random Forest
plot(perf_red_wine_knn, col = "green", lwd = 2, add = TRUE)  # KNN
plot(perf_red_wine_nb, col = "orange", lwd = 2, add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#SUPERB
# red wine Decision Tree model
predictions_red_wine_tree <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_tree <- prediction(predictions_red_wine_tree[, "superb"], test_data_red_wine$quality == "superb")

# Compute performance measures  ROC
perf_red_wine_tree <- performance(pred_red_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (Red Wine)")

# red wine Random Forest model
predictions_red_wine_rf <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
pred_red_wine_rf <- prediction(predictions_red_wine_rf[, "superb"], test_data_red_wine$quality == "superb")

# Compute performance measures ROC
perf_red_wine_rf <- performance(pred_red_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_rf, col = "red", main = "ROC Curve - Random Forest (Red Wine)")

# red wine KNN model
predicted_red_wine_knn <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)

# Calculate predicted probabilities for "superb" class
probabilities_red_wine_knn <- ifelse(predicted_red_wine_knn == "superb", 1, 0)

# Create prediction object
pred_red_wine_knn <- prediction(probabilities_red_wine_knn, test_data_red_wine$quality == "superb")

# Compute performance measures ROC
perf_red_wine_knn <- performance(pred_red_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_knn, col = "green", main = "ROC Curve - KNN (Red Wine)")

# red wine Naive Bayes model
predictions_red_wine_nb <- predict(nb_model_red_wine, test_data_red_wine)

# Calculate predicted probabilities for "superb" class
probabilities_red_wine_nb <- ifelse(predictions_red_wine_nb == "superb", 1, 0)

# Create prediction object
pred_red_wine_nb <- prediction(probabilities_red_wine_nb, test_data_red_wine$quality == "superb")

# Compute performance measures  ROC
perf_red_wine_nb <- performance(pred_red_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_red_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (Red Wine)")

# Plotting all ROC curves together
plot(perf_red_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_Red Wine")  # Decision Tree
plot(perf_red_wine_rf, col = "red", lwd = 2, add = TRUE)  # Random Forest
plot(perf_red_wine_knn, col = "green", lwd = 2, add = TRUE)  # KNN
plot(perf_red_wine_nb, col = "orange", lwd = 2, add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)



#_________________________________________________________________________________________________________________________________________________________#
#For WHITE WINE
#LOW
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "low"], test_data_white_wine$quality == "low")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "low"], test_data_white_wine$quality == "low")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "low" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "low", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "low")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "low" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "low", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "low")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#MEDIUM
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "medium"], test_data_white_wine$quality == "medium")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "medium"], test_data_white_wine$quality == "medium")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "medium" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "medium", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "medium")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "medium" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "medium", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "medium")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)

#HIGH
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "high"], test_data_white_wine$quality == "high")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "high"], test_data_white_wine$quality == "high")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "high" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "high", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "high")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "high" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "high", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "high")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#VERY HIGH
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "very_high"], test_data_white_wine$quality == "very_high")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "very_high"], test_data_white_wine$quality == "very_high")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "very_high" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "very_high", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "very_high")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "very_high" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "very_high", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "very_high")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)

#EXCELLENT
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "excellent"], test_data_white_wine$quality == "excellent")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "excellent"], test_data_white_wine$quality == "excellent")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "excellent" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "excellent", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "excellent")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "excellent" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "excellent", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "excellent")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)

#SUPERB
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "superb"], test_data_white_wine$quality == "superb")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "superb"], test_data_white_wine$quality == "superb")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "superb" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "superb", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "superb")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "superb" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "superb", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "superb")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)


#OUTSTANDING
# White wine Decision Tree model
predictions_white_wine_tree <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_tree <- prediction(predictions_white_wine_tree[, "outstanding"], test_data_white_wine$quality == "outstanding")

# Compute performance measures ROC
perf_white_wine_tree <- performance(pred_white_wine_tree, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_tree, col = "blue", main = "ROC Curve - Decision Tree (White Wine)")

# White wine Random Forest model
predictions_white_wine_rf <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
pred_white_wine_rf <- prediction(predictions_white_wine_rf[, "outstanding"], test_data_white_wine$quality == "outstanding")

# Compute performance measures ROC
perf_white_wine_rf <- performance(pred_white_wine_rf, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_rf, col = "red", main = "ROC Curve - Random Forest (White Wine)")

# White wine KNN model
predicted_white_wine_knn <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)

# Calculate predicted probabilities for "outstanding" class
probabilities_white_wine_knn <- ifelse(predicted_white_wine_knn == "outstanding", 1, 0)

# Create prediction object
pred_white_wine_knn <- prediction(probabilities_white_wine_knn, test_data_white_wine$quality == "outstanding")

# Compute performance measures ROC
perf_white_wine_knn <- performance(pred_white_wine_knn, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_knn, col = "green", main = "ROC Curve - KNN (White Wine)")


# White wine Naive Bayes model
predictions_white_wine_nb <- predict(nb_model_white_wine, test_data_white_wine)

# Calculate predicted probabilities for "outstanding" class
probabilities_white_wine_nb <- ifelse(predictions_white_wine_nb == "outstanding", 1, 0)

# Create prediction object
pred_white_wine_nb <- prediction(probabilities_white_wine_nb, test_data_white_wine$quality == "outstanding")

# Compute performance measures ROC
perf_white_wine_nb <- performance(pred_white_wine_nb, measure = "tpr", x.measure = "fpr")
plot(perf_white_wine_nb, col = "orange", main = "ROC Curve - Naive Bayes (White Wine)")

# Plotting all ROC curves together
plot(perf_white_wine_tree, col = "blue",lwd = 2, main = "ROC Curves Comparison_White Wine")  # Decision Tree
plot(perf_white_wine_rf, col = "red",lwd = 2,add = TRUE)  # Random Forest
plot(perf_white_wine_knn, col = "green",lwd = 2, add = TRUE)  # KNN
plot(perf_white_wine_nb, col = "orange", lwd = 2,add = TRUE)  # Naive Bayes

# Adding a legend
legend("bottomright", legend = c("Decision Tree", "Random Forest", "KNN", "Naive Bayes"),
       col = c("blue", "red", "green", "orange"), lty = 1)

#_____________________________________________________________________________________________________________________________________
#AUC
# RED WINE Test Data
library(pROC)

# Define the levels of quality for the test data
quality_levels_test <- levels(test_data_red_wine$quality)

# Initialize empty vectors to store AUC values for each model using test data
auc_values_tree_test <- numeric(length(quality_levels_test))
auc_values_rf_test <- numeric(length(quality_levels_test))
auc_values_nb_test <- numeric(length(quality_levels_test))
auc_values_knn_test <- numeric(length(quality_levels_test))

# Loop through each level of quality and compute AUC for each model using test data
for (i in seq_along(quality_levels_test)) {
  # Decision Tree - Red Wine (Test Data)
  predictions_tree_red_wine_test <- predict(tree_model_red_wine, newdata = test_data_red_wine, type = "prob")
  perf_tree_red_wine_test <- performance(prediction(predictions_tree_red_wine_test[, quality_levels_test[i]], test_data_red_wine$quality == quality_levels_test[i]), measure = "auc")
  auc_values_tree_test[i] <- as.numeric(perf_tree_red_wine_test@y.values)
  
  # Random Forest - Red Wine (Test Data)
  predictions_rf_red_wine_test <- predict(rf_model_red_wine, newdata = test_data_red_wine, type = "prob")
  perf_rf_red_wine_test <- performance(prediction(predictions_rf_red_wine_test[, quality_levels_test[i]], test_data_red_wine$quality == quality_levels_test[i]), measure = "auc")
  auc_values_rf_test[i] <- as.numeric(perf_rf_red_wine_test@y.values)
  
  # Naive Bayes - Red Wine (Test Data)
  predictions_nb_red_wine_test <- predict(nb_model_red_wine, newdata = test_data_red_wine, type = "raw")
  perf_nb_red_wine_test <- performance(prediction(predictions_nb_red_wine_test[, quality_levels_test[i]], test_data_red_wine$quality == quality_levels_test[i]), measure = "auc")
  auc_values_nb_test[i] <- as.numeric(perf_nb_red_wine_test@y.values)
  
  # KNN - Red Wine (Test Data)
  predicted_knn_red_wine_test <- knn(train_data_red_wine[, -ncol(train_data_red_wine)], test_data_red_wine[, -ncol(test_data_red_wine)], train_data_red_wine$quality, k = optimal_k_red)
  perf_knn_red_wine_test <- performance(prediction(as.numeric(predicted_knn_red_wine_test == quality_levels_test[i]), test_data_red_wine$quality == quality_levels_test[i]), measure = "auc")
  auc_values_knn_test[i] <- as.numeric(perf_knn_red_wine_test@y.values)
}

# Printing AUC values for each quality level using test data
cat("AUC for Decision Tree (Test Data):", auc_values_tree_test, "\n")
cat("AUC for Random Forest (Test Data):", auc_values_rf_test, "\n")
cat("AUC for Naive Bayes (Test Data):", auc_values_nb_test, "\n")
cat("AUC for KNN (Test Data):", auc_values_knn_test, "\n")

# Calculating the average of AUC values 
avg_auc_tree_test <- mean(auc_values_tree_test)
cat("Average AUC for Decision Tree (Test Data):", avg_auc_tree_test, "\n")

avg_auc_rf_test <- mean(auc_values_rf_test)
cat("Average AUC for Random Forest (Test Data):", avg_auc_rf_test, "\n")

avg_auc_nb_test <- mean(auc_values_nb_test)
cat("Average AUC for Naive Bayes (Test Data):", avg_auc_nb_test, "\n")

avg_auc_knn_test <- mean(auc_values_knn_test)
cat("Average AUC for KNN (Test Data):", avg_auc_knn_test, "\n")


#FOR WHITE WINE
#AUC for WHITE WINE Test Data

# Define the levels of quality for the test data
quality_levels_test_white <- levels(test_data_white_wine$quality)

# Initialize empty vectors to store AUC values for each model using test data
auc_values_tree_test_white <- numeric(length(quality_levels_test_white))
auc_values_rf_test_white <- numeric(length(quality_levels_test_white))
auc_values_nb_test_white <- numeric(length(quality_levels_test_white))
auc_values_knn_test_white <- numeric(length(quality_levels_test_white))

# Loop through each level of quality and compute AUC for each model using test data
for (i in seq_along(quality_levels_test_white)) {
  # Decision Tree - White Wine (Test Data)
  predictions_tree_white_test <- predict(tree_model_white_wine, newdata = test_data_white_wine, type = "prob")
  perf_tree_white_test <- performance(prediction(predictions_tree_white_test[, quality_levels_test_white[i]], test_data_white_wine$quality == quality_levels_test_white[i]), measure = "auc")
  auc_values_tree_test_white[i] <- as.numeric(perf_tree_white_test@y.values)
  
  # Random Forest - White Wine (Test Data)
  predictions_rf_white_test <- predict(rf_model_white_wine, newdata = test_data_white_wine, type = "prob")
  perf_rf_white_test <- performance(prediction(predictions_rf_white_test[, quality_levels_test_white[i]], test_data_white_wine$quality == quality_levels_test_white[i]), measure = "auc")
  auc_values_rf_test_white[i] <- as.numeric(perf_rf_white_test@y.values)
  
  # Naive Bayes - White Wine (Test Data)
  predictions_nb_white_test <- predict(nb_model_white_wine, newdata = test_data_white_wine, type = "raw")
  perf_nb_white_test <- performance(prediction(predictions_nb_white_test[, quality_levels_test_white[i]], test_data_white_wine$quality == quality_levels_test_white[i]), measure = "auc")
  auc_values_nb_test_white[i] <- as.numeric(perf_nb_white_test@y.values)
  
  # KNN - White Wine (Test Data)
  predicted_knn_white_test <- knn(train_data_white_wine[, -ncol(train_data_white_wine)], test_data_white_wine[, -ncol(test_data_white_wine)], train_data_white_wine$quality, k = optimal_k_white)
  perf_knn_white_test <- performance(prediction(as.numeric(predicted_knn_white_test == quality_levels_test_white[i]), test_data_white_wine$quality == quality_levels_test_white[i]), measure = "auc")
  auc_values_knn_test_white[i] <- as.numeric(perf_knn_white_test@y.values)
}

# Printing AUC values for each quality level using test data
cat("AUC for Decision Tree (Test Data - White Wine):", auc_values_tree_test_white, "\n")
cat("AUC for Random Forest (Test Data - White Wine):", auc_values_rf_test_white, "\n")
cat("AUC for Naive Bayes (Test Data - White Wine):", auc_values_nb_test_white, "\n")
cat("AUC for KNN (Test Data - White Wine):", auc_values_knn_test_white, "\n")


# Calculating the average of AUC values for each model 
avg_auc_tree_test_white <- mean(auc_values_tree_test_white)
cat("Average AUC for Decision Tree (Test Data - White Wine):", avg_auc_tree_test_white, "\n")

avg_auc_rf_test_white <- mean(auc_values_rf_test_white)
cat("Average AUC for Random Forest (Test Data - White Wine):", avg_auc_rf_test_white, "\n")

avg_auc_nb_test_white <- mean(auc_values_nb_test_white)
cat("Average AUC for Naive Bayes (Test Data - White Wine):", avg_auc_nb_test_white, "\n")

avg_auc_knn_test_white <- mean(auc_values_knn_test_white)
cat("Average AUC for KNN (Test Data - White Wine):", avg_auc_knn_test_white, "\n")










