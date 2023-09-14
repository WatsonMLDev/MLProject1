import numpy as np  # this package is for matrix computation
import pandas as pd  # this package is for data formating and processing

# load data from data file
train_df = pd.read_csv('../data/train.csv')
test_X_df = pd.read_csv('../data/test_X.csv')
sample_y_df = pd.read_csv('../data/sample_submission.csv')
# take a look at your training set (with features and ground-truth label 'HeartDisease')
train_df.info()
train_df.head(n=5)
# take a look at your test set (with only features)
test_X_df.info()
test_X_df.head(n=5)
# take a look at the format of submission (with only predicted labels)
# your submitted prediction on test_X should follow this format, otherwise you may receive errors on Kaggle
sample_y_df.info()
sample_y_df.head(n=5)

import preprocessing

# You may apply feature proceccing tricks mentioned in class
# e.g., feature normalization/standardization etc
for column_name, column_data in my_train_df.iteritems():
    min_val = column_data.min()
    max_val = column_data.max()
    q1 = column_data.quantile(0.25)
    q3 = column_data.quantile(0.75)

    print(f"{column_name}: min: {min_val} MAX: {max_val} Q1: {q1} Q3: {q3}")
# removing outliers of oldpeak, maxhr, and age
def identify_outliers(df):
  outlier_rows = []
  for column_name, items in df.iteritems():
      if column_name in ["Age", "MaxHR","Oldpeak"]:
          q1 = items.quantile(0.25)
          q3 = items.quantile(0.75)

          iqr = q3 - q1
          lb = q1 - (1.5 * iqr)
          ub = q3 + (1.5 * iqr)

          column_outliers = []

          for index, row in df.iterrows():
              if row[column_name] < lb or row[column_name] > ub:
                  column_outliers.append(index)

          outlier_rows.append((column_name, column_outliers))  # Append outliers for this column
  return outlier_rows

def print_outliers(outlier_rows):
  for column_name, outliers in outlier_rows:
      print(f"For column {column_name}, the outliers are:")
      print(outliers)
      print("------------------")

def drop_outliers(outlier_rows, df):
  for column_name, outliers in outlier_rows:
      df = df.drop(outliers)
  return df


def cap_outliers(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])

    return df

# Cap outliers for the 'Oldpeak' column in the training dataset
my_train_df = cap_outliers(my_train_df, 'Oldpeak')

# Display the updated statistics for 'Oldpeak'
my_train_df['Oldpeak'].describe()




outliers = identify_outliers(my_train_df)

print_outliers(outliers)

my_train_df = drop_outliers(outliers, my_train_df)

print("------------------------------")


outliers = identify_outliers(my_train_df)

print_outliers(outliers)





# make predictions on test data
test_y_pred = final_model.predict(test_X)

# prepare the prediction file to submit on Kaggle
submission_df = pd.DataFrame({
    'PatientID': my_test_X_df['PatientID'],
    'HeartDisease': test_y_pred
    }
)
submission_df.to_csv("y_predict.csv", index=False)
submission_df.head(3)