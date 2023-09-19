import pandas as pd  # this package is for data formating and processing
import src.models.svm as svm
from src.utilities.preprocessing import one_hot_encode_categoricals, get_outlier_indices, remove_outliers_by_indices, \
    limit_outlier_values, normalize

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

# Train the initial model
final_model = svm.prepare_model(train_features, train_labels, tune=True)
print("---------default values---------")
svm.prepare_model(train_features, train_labels, tune=False)


# make predictions on test data
test_y_pred = final_model.predict(test_features)

# prepare the prediction file to submit on Kaggle
submission_df = pd.DataFrame({
    'PatientID': processed_test_df['PatientID'],
    'HeartDisease': test_y_pred
}
)
submission_df.to_csv("y_predict.csv", index=False)
