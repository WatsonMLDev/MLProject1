import numpy as np
import pandas as pd  # this package is for data formating and processing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import src.models.random_forest as rf
from src.utilities.preprocessing import one_hot_encode_categoricals, get_outlier_indices, remove_outliers_by_indices, \
    limit_outlier_values, normalize
from utilities.postprocessing import drop_non_essential_features

# load data from data file
train_df = pd.read_csv('../data/train.csv')
test_X_df = pd.read_csv('../data/test_X.csv')

# convert features for training and testing data
processed_train_df = one_hot_encode_categoricals(train_df)
processed_test_df = one_hot_encode_categoricals(test_X_df)

# Cap outliers for the 'Oldpeak' column in the training dataset
processed_train_df = limit_outlier_values(processed_train_df, 'Oldpeak')
processed_test_df = limit_outlier_values(processed_test_df, 'Oldpeak')

# removing outliers of oldpeak, maxhr, and age
outlier_rows = get_outlier_indices(processed_train_df, ['MaxHR', 'Age', 'Oldpeak'])
processed_train_df = remove_outliers_by_indices(outlier_rows, processed_train_df)

outlier_rows = get_outlier_indices(processed_test_df, ['MaxHR', 'Age', 'Oldpeak'])
processed_test_df = remove_outliers_by_indices(outlier_rows, processed_test_df)

# scale numerical features
columns_to_scale  = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
processed_train_df, train_scaler = normalize(processed_train_df, columns_to_scale)
processed_test_df, _ = normalize(processed_test_df, columns_to_scale, train_scaler)

print("---------Initial Model---------")
# Initial preparation of features and labels
train_features = processed_train_df.drop(["HeartDisease", "PatientID"], axis=1)
train_labels = processed_train_df["HeartDisease"]
test_features = processed_test_df.drop(["PatientID"], axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

# Train the initial model
initial_model = rf.prepare_model(X_train, y_train, tune=True)

# Validate the initial model
val_predictions_initial = initial_model.predict(X_val)
print(f"Validation F1 Score for initial model: {f1_score(y_val, val_predictions_initial)}")

# Identify and drop non-essential features
processed_train_df_simplified, processed_test_df_simplified = drop_non_essential_features(initial_model, train_features, processed_train_df, processed_test_df)

print("---------Final Model---------")
# Prepare features and labels again using the simplified dataset
train_features_simplified = processed_train_df_simplified.drop(["HeartDisease", "PatientID"], axis=1)
train_labels_simplified = processed_train_df_simplified["HeartDisease"]
test_features_simplified = processed_test_df_simplified.drop(["PatientID"], axis=1)

# Split the simplified data into training and validation sets
X_train_simplified, X_val_simplified, y_train_simplified, y_val_simplified = train_test_split(train_features_simplified, train_labels_simplified, test_size=0.2, random_state=42)

# Train the model again using the simplified dataset
final_model = rf.prepare_model(X_train_simplified, y_train_simplified, tune=True)

# Validate the final model
val_predictions_final = final_model.predict(X_val_simplified)
print(f"Validation F1 Score for final model: {f1_score(y_val_simplified, val_predictions_final)}")

# Make predictions on test data
test_y_pred = final_model.predict(test_features_simplified)

# Prepare the prediction file to submit on Kaggle
submission_df = pd.DataFrame({
    'PatientID': processed_test_df['PatientID'],
    'HeartDisease': test_y_pred
})
submission_df.to_csv("y_predict.csv", index=False)
