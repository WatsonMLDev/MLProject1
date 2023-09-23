import pandas as pd  # this package is for data formating and processing
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import src.models.knn as knn
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



# prepare features and labels for training/testing
train_features = processed_train_df.drop(["HeartDisease", "PatientID"], axis=1)
train_labels = processed_train_df["HeartDisease"]
test_features = processed_test_df.drop(["PatientID"], axis=1)

#split training data and test data
X_train_base, X_train_meta, y_train_base, y_train_meta = train_test_split(
    train_features, train_labels, test_size=0.5, random_state=42)

final_model = knn.prepare_knn_model(X_train_base, y_train_base)

# evaluate the model
test_predicitons = final_model.predict(X_train_meta)
# frind training acuarcy and f1score
print("Training Accuracy:", accuracy_score(y_train_meta, test_predicitons))
print("Training F1 Score:", f1_score(y_train_meta, test_predicitons))


# make predictions on test data
test_y_pred = final_model.predict(test_features)

# prepare the prediction file to submit on Kaggle
submission_df = pd.DataFrame({
    'PatientID': processed_test_df['PatientID'],
    'HeartDisease': test_y_pred
}
)
submission_df.to_csv("y_predict.csv", index=False)
