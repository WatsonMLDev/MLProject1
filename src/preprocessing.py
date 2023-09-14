# this function is to convert categorical feature to numerical (one-hot representation)
def convert_categorical_to_numerical(df):
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

# convert features for training and testing data
my_train_df = convert_categorical_to_numerical(train_df)
my_test_X_df = convert_categorical_to_numerical(test_X_df)

my_train_df.head(n=5)