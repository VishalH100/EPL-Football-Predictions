import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("C:/Users/Vishal/Desktop/Dissertation/Datasets/Final EPL 2018-2023.csv", encoding='latin')

# Preprocess the data
df['Home_Team_Score'] = df['Score'].apply(lambda x: int(x[0]))
df['Away_Team_Score'] = df['Score'].apply(lambda x: int(x[-1]))

# Define a function to determine match result
def determine_result(home_score, away_score):
    if home_score > away_score:
        return 'Home Team Wins'
    elif home_score < away_score:
        return 'Away Team Wins'
    else:
        return 'Draw'

# Create 'Result' column
df['Result'] = df.apply(lambda row: determine_result(row['Home_Team_Score'], row['Away_Team_Score']), axis=1)

# Feature Engineering
features = ['Home_Team', 'Away_Team', 'Season', 'Home_xG', 'Away_xG']
target_columns = ['Home_Team_Score', 'Away_Team_Score']

# Filter data for seasons before 2022-2023
train = df[df['Season'] != "2022-2023"]
test = df[df['Season'] == "2022-2023"]

# One-hot encode categorical features using the same encoder instance with handle_unknown='ignore'
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_features_train = encoder.fit_transform(train[['Home_Team', 'Away_Team', 'Season']])
X_train = np.concatenate([encoded_features_train.toarray(), train[['Home_xG', 'Away_xG']].values], axis=1)
y_train = train[target_columns].values

# Prepare test data for 2022-2023 season using the same encoder instance
encoded_features_test = encoder.transform(test[['Home_Team', 'Away_Team', 'Season']])
X_test = np.concatenate([encoded_features_test.toarray(), test[['Home_xG', 'Away_xG']].values], axis=1)
y_test = test[target_columns].values

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions on the test data
predictions = model.predict(X_test_scaled)

# Round predictions to nearest integer
predictions = np.round(predictions)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2_acc = r2_score(y_test, predictions)

# Print the evaluation metrics
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'R^2 Accuracy: {r2_acc:.2f}')

# Create DataFrame for test data with predictions
test_with_predictions = test.copy()
test_with_predictions['Predicted_Home_Score'] = predictions[:, 0]
test_with_predictions['Predicted_Away_Score'] = predictions[:, 1]

# Create 'Prediction_Result' column
test_with_predictions['Prediction_Result'] = test_with_predictions.apply(lambda row: determine_result(row['Predicted_Home_Score'], row['Predicted_Away_Score']), axis=1)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(test_with_predictions['Result'], test_with_predictions['Prediction_Result'])
precision = precision_score(test_with_predictions['Result'], test_with_predictions['Prediction_Result'], average='weighted')
recall = recall_score(test_with_predictions['Result'], test_with_predictions['Prediction_Result'], average='weighted')
f1 = f1_score(test_with_predictions['Result'], test_with_predictions['Prediction_Result'], average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

# Confusion Matrix
conf_matrix = confusion_matrix(test_with_predictions['Result'], test_with_predictions['Prediction_Result'])
print("Confusion Matrix:")
print(conf_matrix)


# Export the final predictions to a CSV file
test_with_predictions.to_csv("C:/Users/Vishal/Desktop/Dissertation/Datasets/RF Predictions.csv", index=False)