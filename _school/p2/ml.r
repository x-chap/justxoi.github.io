# Install and load necessary packages
library(readr)
library(dplyr)
library(caret)
library(randomForest)
library(gbm)
library(ggplot2)
library(tidyr)

# Load the dataset
file_path <- 'Project/p2/merged_esports_data_updated.csv' # Update this with the correct path
data <- read_csv(file_path)

# Fill missing values with median or another appropriate value
data_filled <- data %>% mutate(across(everything(), ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

# One-hot encoding for categorical variables
categorical_cols <- c('map_type', 'map_name', 'player_name', 'team_name', 'hero_name')
data_encoded <- data_filled %>% mutate_at(vars(one_of(categorical_cols)), as.factor) %>% dummyVars(~ ., data = .)
data_final <- data.frame(predict(data_encoded, newdata = data_filled))

# Drop irrelevant columns
data_final <- select(data_final, -matches(c("team_one_name", "team_two_name", "match_id"), names(data_final)))

# Split the data into features and target
set.seed(42)
target <- "match_winner"
features <- setdiff(names(data_final), target)
X <- data_final[features]
y <- data_final[[target]]

# Split the data into training and testing sets
trainIndex <- createDataPartition(y, p = .8, list = FALSE)
X_train <- X[trainIndex, ]
y_train <- y[trainIndex]
X_test <- X[-trainIndex, ]
y_test <- y[-trainIndex]

# Feature selection using Random Forest
feature_selector <- randomForest(X_train, y_train, ntree=100)
important_features <- importance(feature_selector)

# Train models
logistic_regression_model <- train(y_train ~ ., data = data.frame(X_train, y_train), method = "glm")
random_forest_model <- randomForest(X_train, y_train)

# Predictions and Evaluation
predictions_lr <- predict(logistic_regression_model, X_test)
predictions_rf <- predict(random_forest_model, X_test)


# Train the Gradient Boosting model with cross-validation
set.seed(42)
gradient_boosting_model <- gbm(
  formula = y_train ~ .,
  distribution = "bernoulli",
  data = data.frame(X_train, y_train),
  n.trees = 5000, # Number of trees can be adjusted based on your dataset
  cv.folds = 5,   # Number of cross-validation folds
  verbose = FALSE
)

# Find the best number of trees
best_trees <- gbm.perf(gradient_boosting_model, method = "cv")

# Predictions using the best number of trees
predictions_gb <- predict(gradient_boosting_model, X_test, n.trees = best_trees, type = "response")

# Convert probabilities to class labels based on a threshold (e.g., 0.5)
predictions_gb <- ifelse(predictions_gb > 0.5, 1, 0)

# Gradient Boosting Classification Report
print("Gradient Boosting Classification Report:")
print(confusionMatrix(factor(predictions_gb), factor(y_test)))

# Classification report
print("Logistic Regression Classification Report:")
print(confusionMatrix(predictions_lr, y_test))

print("Random Forest Classification Report:")
print(confusionMatrix(predictions_rf, y_test))


# Feature Importances
feature_importances_rf <- varImp(random_forest_model)
feature_importances_gb <- summary(gradient_boosting_model)

# Plotting feature importances
# For Random Forest
rf_imp <- data.frame(Importance = feature_importances_rf$Overall, Feature = row.names(feature_importances_rf))
ggplot(rf_imp, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance - Random Forest")

# For Gradient Boosting
gb_imp <- data.frame(Importance = feature_importances_gb, Feature = names(feature_importances_gb))
ggplot(gb_imp, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance - Gradient Boosting")
