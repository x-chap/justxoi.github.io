import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'Project/p2/merged_esports_data_updated.csv'  # Update this with the path to your CSV file
data = pd.read_csv(file_path)

# Fill missing values with 0
data_filled = data.fillna(0)

# One-hot encoding for categorical variables
categorical_cols = ['map_type', 'map_name', 'player_name', 'team_name', 'hero_name']
data_encoded = pd.get_dummies(data_filled, columns=categorical_cols, drop_first=True)

# Drop team name columns and match_id to avoid bias
data_final = data_encoded.drop(columns=['team_one_name', 'team_two_name', 'match_id'])

# Split the data into features and target
X = data_final.drop('match_winner', axis=1)
y = data_final['match_winner']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection using Random Forest
feature_selector = RandomForestClassifier(n_estimators=100)
feature_selector.fit(X_train, y_train)

# Select important features
selector = SelectFromModel(feature_selector, prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Initialize models
logistic_regression_model = LogisticRegression(max_iter=1000)
random_forest_model = RandomForestClassifier()
gradient_boosting_model = GradientBoostingClassifier()

# Train models using selected features
logistic_regression_model.fit(X_train_selected, y_train)
random_forest_model.fit(X_train_selected, y_train)
gradient_boosting_model.fit(X_train_selected, y_train)

# Predictions
logistic_regression_predictions = logistic_regression_model.predict(X_test_selected)
random_forest_predictions = random_forest_model.predict(X_test_selected)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test_selected)

# Classification report
print("Logistic Regression Classification Report:\n", classification_report(y_test, logistic_regression_predictions))
print("Random Forest Classification Report:\n", classification_report(y_test, random_forest_predictions))
print("Gradient Boosting Classification Report:\n", classification_report(y_test, gradient_boosting_predictions))

# Feature Importances
feature_importances_rf = feature_selector.feature_importances_
feature_importances_gb = gradient_boosting_model.feature_importances_

# Sort the feature importances and get the top 20 features
indices_rf = np.argsort(feature_importances_rf)[::-1][:20]
indices_gb = np.argsort(feature_importances_gb)[::-1][:20]

# Prepare the data for plotting
top_features_rf = X_train.columns[indices_rf]
top_importances_rf = feature_importances_rf[indices_rf]

top_features_gb = X_train.columns[indices_gb]
top_importances_gb = feature_importances_gb[indices_gb]

# Plotting feature importances
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Random Forest
sns.barplot(x=top_importances_rf, y=top_features_rf, ax=axes[0], color='skyblue')
axes[0].set_title('Top 20 Feature Importances - Random Forest')
axes[0].set_xlabel('Importance')

# Gradient Boosting
sns.barplot(x=top_importances_gb, y=top_features_gb, ax=axes[1], color='lightgreen')
axes[1].set_title('Top 20 Feature Importances - Gradient Boosting')
axes[1].set_xlabel('Importance')

plt.tight_layout()
plt.show()
