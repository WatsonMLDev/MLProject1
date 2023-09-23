import pandas as pd
def drop_non_essential_features(model, train_features, processed_train_df, processed_test_df):
    importances = model.feature_importances_

    # Map the feature names to their importance scores
    feature_importances = pd.Series(importances, index=train_features.columns)

    # Determine the threshold for dropping features
    max_importance = max(feature_importances)
    threshold = 0.1 * max_importance

    # Identify features to drop
    features_to_drop = [feature for feature, importance in zip(train_features.columns, feature_importances) if importance < threshold]

    # Print the features being dropped
    print("Dropping the following features:", features_to_drop)

    # Drop these features from both train and test dataframes
    processed_train_df_simplified = processed_train_df.drop(features_to_drop, axis=1)
    processed_test_df_simplified = processed_test_df.drop(features_to_drop, axis=1)

    return processed_train_df_simplified, processed_test_df_simplified
