import numpy as np
import pandas as pd  # this package is for data formating and processing
from sklearn.metrics import f1_score

import src.models.stacked as stacked
from src.utilities.preprocessing import one_hot_encode_categoricals, get_outlier_indices, remove_outliers_by_indices, \
    limit_outlier_values, normalize
from sklearn.model_selection import train_test_split


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

print("---------Initialize all Models---------")
# Initial preparation of features and labels
train_features = processed_train_df.drop(["HeartDisease", "PatientID"], axis=1)
train_labels = processed_train_df["HeartDisease"]
test_features = processed_test_df.drop(["PatientID"], axis=1)

#split training data and test data
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
    train_features, train_labels, test_size=0.5, random_state=42)

#train models
# knn_model = stacked.prepare_knn_model(X_train_base, y_train_base)
rf_model = stacked.prepare_rf_model(X_train_base, y_train_base, False)
svm_model = stacked.prepare_svm_model(X_train_base, y_train_base, False)
nn_model = stacked.prepare_nn_model(X_train_base, y_train_base, False)

#make predictions
# knn_test_predictions = knn_model.predict(X_train_meta)
rf_test_predictions = rf_model.predict(X_train_meta)
svm_test_predictions = svm_model.predict(X_train_meta)
nn_test_predictions = nn_model.predict(X_train_meta)

#stack the prediciotns
stacked_features = np.column_stack(( rf_test_predictions, svm_test_predictions, nn_test_predictions))

#train the meta model
meta_model = stacked.prepare_svm_model(stacked_features, y_train_meta, False)
stacked_predictions = meta_model.predict(stacked_features)

f1 = f1_score(y_train_meta , stacked_predictions)
print("F1 Score of Stacked Model on Training Data:", f1)

# knn_test_predictions = knn_model.predict(test_features)
rf_test_predictions = rf_model.predict(test_features)
svm_test_predictions = svm_model.predict(test_features)
nn_test_predictions = nn_model.predict(test_features)

stacked_test_predictions = np.column_stack(( rf_test_predictions, svm_test_predictions, nn_test_predictions))

final_predictions = meta_model.predict(stacked_test_predictions)

# prepare the prediction file to submit on Kaggle
submission_df = pd.DataFrame({
    'PatientID': processed_test_df['PatientID'],
    'HeartDisease': final_predictions
}
)
submission_df.to_csv("y_predict.csv", index=False)
