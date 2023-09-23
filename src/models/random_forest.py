import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real

warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_model(train_x, train_y, tune=True):

    def objective(params):
        n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features = params

        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth != -1 else None,  # -1 means None (no limit)
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=42,
            n_jobs=-1  # Use all available CPU cores
        )

        return -np.mean(cross_val_score(rf, train_x, train_y, cv=3, n_jobs=-1, scoring="f1"))

    if tune:
        space = [
            Integer(50, 200, name="n_estimators"),
            Integer(1, 30, name="max_depth"),  # -1 means None (no limit)
            Integer(2, 10, name="min_samples_split"),
            Integer(1, 4, name="min_samples_leaf"),
            Categorical(['sqrt'], name="max_features")
        ]

        # Run the optimizer
        result = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=True)
        print(f"Best parameters: {result.x}")

        best_rf_model = RandomForestClassifier(
            n_estimators=result.x[0],
            max_depth=result.x[1] if result.x[1] != -1 else None,
            min_samples_split=result.x[2],
            min_samples_leaf=result.x[3],
            max_features=result.x[4],
            random_state=42,
            n_jobs=-1
        )
    else:
        best_rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    best_rf_model.fit(train_x, train_y)

    # Evaluate the model on training data
    train_y_pred = best_rf_model.predict(train_x)
    print("Training Accuracy:", accuracy_score(train_y, train_y_pred))
    print("Training F1 Score:", f1_score(train_y, train_y_pred))

    return best_rf_model
