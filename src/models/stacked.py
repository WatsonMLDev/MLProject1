import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.svm import SVC
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real

warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_knn_model(train_x, train_y):

    # Objective function to optimize
    def objective(params):
        n_neighbors = params[0]

        model = KNeighborsClassifier(n_neighbors=n_neighbors)

        # We use cross_val_score and take the negative mean of the scores since we want to maximize F1-score
        return -np.mean(cross_val_score(model, train_x, train_y, cv=5, n_jobs=-1, scoring="f1"))

    # Define the hyperparameter space to search
    space = [
        Integer(3, 15, name="n_neighbors")
    ]

    # Run the optimizer
    result = gp_minimize(objective, space, n_calls=15, random_state=42, verbose=True)

    optimal_n_neighbors = result.x[0]
    print(f"Optimal number of neighbors: {optimal_n_neighbors}")

    # Train the model with optimal hyperparameters
    final_model = KNeighborsClassifier(n_neighbors=optimal_n_neighbors)
    final_model.fit(train_x, train_y)

    # Evaluate the model
    train_y_pred = final_model.predict(train_x)
    print("Training Accuracy:", accuracy_score(train_y, train_y_pred))
    print("Training F1 Score:", f1_score(train_y, train_y_pred))

    return final_model

def prepare_rf_model(train_x, train_y, tune=True):

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
            Categorical(['auto', 'sqrt'], name="max_features")
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
        best_rf_model = RandomForestClassifier(n_estimators=50,max_depth=30, min_samples_split=10, min_samples_leaf=4, max_features='sqrt', random_state=42, n_jobs=-1)

    best_rf_model.fit(train_x, train_y)
    return best_rf_model

def prepare_svm_model(train_x, train_y, tune=True):

    def objective(params):
        C, degree, kernel, gamma = params

        svm = SVC(
            C=C,
            degree=degree,
            kernel=kernel,
            gamma=gamma,
            random_state=42
        )

        return -np.mean(cross_val_score(svm, train_x, train_y, cv=3, n_jobs=-1, scoring="f1"))

    if tune:
        space = [
            Real(1e-2, 1e+2, name="C"),
            Integer(2, 5, name="degree"),
            Categorical(['linear', 'rbf', 'poly', 'sigmoid'], name="kernel"),
            Categorical(['scale', 'auto'], name="gamma")
        ]

        # Run the optimizer
        result = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=True)
        print(f"Best parameters: {result.x}")

        best_svm_model = SVC(
            C=result.x[0],
            degree=result.x[1],
            kernel=result.x[2],
            gamma=result.x[3],
            random_state=42
        )
    else:
        best_svm_model = SVC(C=1.3363634663750545, degree= 5, kernel='rbf', gamma='auto', random_state=42)

    best_svm_model.fit(train_x, train_y)
    return best_svm_model

def prepare_nn_model(train_x, train_y, tune=True):

    def objective(params):
        hidden_layer_sizes, activation, solver, alpha, learning_rate_init = params

        mlp = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            random_state=42,
            max_iter=1000
        )

        return -np.mean(cross_val_score(mlp, train_x, train_y, cv=3, n_jobs=-1, scoring="f1"))

    if tune:
        space = [
            Integer(10, 200, name="hidden_layer_sizes"),
            Categorical(['relu', 'tanh', 'logistic'], name="activation"),
            Categorical(['adam', 'sgd'], name="solver"),
            Real(10**-5, 10**0, "log-uniform", name="alpha"),
            Real(10**-5, 10**0, "log-uniform", name="learning_rate_init")
        ]

        # Run the optimizer
        result = gp_minimize(objective, space, n_calls=20, random_state=42, verbose=True)
        print(f"Best parameters: {result.x}")

        best_nn_model = MLPClassifier(
            hidden_layer_sizes=result.x[0],
            activation=result.x[1],
            solver=result.x[2],
            alpha=result.x[3],
            learning_rate_init=result.x[4],
            random_state=42,
            max_iter=1000
        )
    else:
        best_nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='tanh', solver='adam', alpha=1e-05, learning_rate_init=0.0006863989086718845, random_state=42)

    best_nn_model.fit(train_x, train_y)
    return best_nn_model
