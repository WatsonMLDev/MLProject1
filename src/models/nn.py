import warnings
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real

warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_model(train_x, train_y, tune = True):
    train_X_new, valid_X, train_y_new, valid_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42)

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

        return -np.mean(cross_val_score(mlp, train_X_new, train_y_new, cv=3, n_jobs=-1, scoring="f1"))


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
        print(f"Best accuracy: {-result.fun}")

        best_nn_model = MLPClassifier(
            hidden_layer_sizes=result.x[0],
            activation=result.x[1],
            solver=result.x[2],
            alpha=result.x[3],
            learning_rate_init=result.x[4],
            random_state=42,
            max_iter=1000
        )
        best_nn_model.fit(train_X_new, train_y_new)
    else:
        best_nn_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', random_state=42)
        best_nn_model.fit(train_X_new, train_y_new)

    train_y_pred = best_nn_model.predict(train_x)

    print("Training Accuracy:", accuracy_score(train_y, train_y_pred))
    print("Training F1 Score:", f1_score(train_y, train_y_pred))

    valid_y_pred = best_nn_model.predict(valid_X)

    valid_accuracy = accuracy_score(valid_y, valid_y_pred)
    valid_f1 = f1_score(valid_y, valid_y_pred)

    print(f"Validation Accuracy: {valid_accuracy}")
    print(f"Validation F1 Score: {valid_f1}")

    return best_nn_model