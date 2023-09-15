
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def one_hot_encode_categoricals(df):
    new_df = df.copy()  # so operations on new_df will not influence df

    # check get_dummies doc: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html for more info
    sex = pd.get_dummies(new_df['Sex'], prefix='sex', dtype=float) # convert Sex to integer values
    chest = pd.get_dummies(new_df['ChestPainType'], prefix='chest', dtype=float) # convert ChestPainType to integer values

    # YOUR TASK: convert other categorical features
    resting = pd.get_dummies(new_df['RestingECG'], prefix="restingECG", dtype=float)
    angina = pd.get_dummies(new_df['ExerciseAngina'], prefix="angina", dtype=float)
    st_slope = pd.get_dummies(new_df['ST_Slope'], prefix="stslope", dtype=float)



    # drop categorical features with their numerical values
    # YOUR TASK: drop other categorical features
    new_df.drop(['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina','ST_Slope' ], axis=1, inplace=True)

    # create new dataframe with only numerical values
    # YOUR TASK: concatenate with other converted features
    new_df = pd.concat([new_df, sex, chest, resting, angina, st_slope], axis=1)

    return new_df


def get_outlier_indices(df, columns):
    outlier_rows = []
    for column_name, items in df.items():
        if column_name in columns:
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

def remove_outliers_by_indices(outlier_rows, df):
    for column_name, outliers in outlier_rows:
        df = df.drop(outliers)
    return df

def limit_outlier_values(df, column_name):
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])

    return df

def normalize(df, columns, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    else:
        df[columns] = scaler.transform(df[columns])
    return df, scaler
