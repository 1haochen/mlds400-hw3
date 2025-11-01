# Load required libraries
library(readr)
library(dplyr)
library(caret)
library(glmnet)

# Read the training data
train_data <- read_csv('src/data/train.csv')
cat('Training data loaded successfully! Shape:', dim(train_data), '\n\n')

# Show first few rows
print(head(train_data))
cat('\n')

# Check missing values
cat('Checking missing values...\n')
print(colSums(is.na(train_data)))
cat('\n')

# Summary statistics
cat('Summary statistics:\n')
print(summary(train_data))
cat('\n')

# Feature selection and preprocessing
features <- c('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked')
target <- 'Survived'
cat('Selected features:', paste(features, collapse=', '), '\n', 'target:', target, '\n\n')

# Impute missing values
train_data$Age[is.na(train_data$Age)] <- median(train_data$Age, na.rm=TRUE)
train_data$Embarked[is.na(train_data$Embarked)] <- names(sort(table(train_data$Embarked), decreasing=TRUE))[1]
cat('Imputed missing values in age with median, imputed missing values in embarked with mode.\n\n')

# Prepare feature matrix and target vector
X <- train_data[, features]
y <- train_data[[target]]
cat('Feature matrix shape:', dim(X), 'Target vector shape:', length(y), '\n\n')

# Convert categorical variables
X$Sex <- as.factor(X$Sex)
X$Embarked <- as.factor(X$Embarked)
X$Pclass <- as.factor(X$Pclass)

# Create dummy variables
dummies <- dummyVars(~ ., data=X)
X_mat <- predict(dummies, newdata=X)

# Standardize numeric features
preProc <- preProcess(X_mat, method=c('center', 'scale'))
X_scaled <- predict(preProc, X_mat)

# Fit logistic regression model
cat('Building logistic regression model...\n')
model <- glm(y ~ ., data=as.data.frame(X_scaled), family=binomial)
cat('Model training complete!\n\n')

# Training accuracy
y_pred_train <- ifelse(predict(model, type='response') > 0.5, 1, 0)
train_acc <- mean(y_pred_train == y)
cat(sprintf('Training Accuracy: %.4f\n\n', train_acc))

# Load test data
cat('Loading test dataset...\n')
test_data <- read_csv('src/data/test.csv')
cat('Test data loaded successfully! Shape:', dim(test_data), '\n\n')

# Preprocess test data
test_data$Age[is.na(test_data$Age)] <- median(train_data$Age, na.rm=TRUE)
test_data$Fare[is.na(test_data$Fare)] <- median(train_data$Fare, na.rm=TRUE)
test_data$Embarked[is.na(test_data$Embarked)] <- names(sort(table(train_data$Embarked), decreasing=TRUE))[1]
X_test <- test_data[, features]
X_test$Sex <- as.factor(X_test$Sex)
X_test$Embarked <- as.factor(X_test$Embarked)
X_test$Pclass <- as.factor(X_test$Pclass)
dummies_test <- dummyVars(~ ., data=X_test)
X_test_mat <- predict(dummies_test, newdata=X_test)
X_test_scaled <- predict(preProc, X_test_mat)

# Predict on test data
cat('Predicting on test data...\n')
test_predictions <- ifelse(predict(model, newdata=as.data.frame(X_test_scaled), type='response') > 0.5, 1, 0)

# Save output
output <- data.frame(PassengerId=test_data$PassengerId, Survived=test_predictions)
write_csv(output, 'test_predictions-r.csv')
cat("Test predictions saved to 'test_predictions-r.csv'\n")
cat('=== SCRIPT COMPLETE ===\n')
