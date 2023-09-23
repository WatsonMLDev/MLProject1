import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from skopt import gp_minimize
from skopt.space import Integer


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
    final_model = KNeighborsClassifier(n_neighbors=optimal_n_neighbors, weights="distance")
    final_model.fit(train_x, train_y)

    # Evaluate the model
    train_y_pred = final_model.predict(train_x)
    print("Training Accuracy:", accuracy_score(train_y, train_y_pred))
    print("Training F1 Score:", f1_score(train_y, train_y_pred))

    return final_model
