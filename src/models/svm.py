from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV, gp_minimize
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prepare_model(train_x, train_y, tune = True):
    train_X_new, valid_X, train_y_new, valid_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=42)

    def objective(params):
        C, gamma, kernel, degree = params
        svm = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree)
        svm.fit(train_X_new, train_y_new)
        acc = svm.score(valid_X, valid_y)
        return -acc  # We negate because we want to maximize accuracy

    if tune:
        # # Define the hyperparameters and their possible values

        param_space = [
            Real(0.1, 1000, "log-uniform", name='C'),
            Real(0.1, 1000, "log-uniform", name='gamma'),
            Categorical(['linear', 'rbf', 'poly', 'sigmoid'], name='kernel'),
            Integer(1, 8, name='degree')
        ]
        result = gp_minimize(objective, param_space, n_calls=22, random_state=0, verbose=True)
        print(f"Best parameters: {result.x}")
        print(f"Best accuracy: {-result.fun}")

        best_svm_model = SVC(C=result.x[0], gamma=result.x[1], kernel=result.x[2], degree=result.x[3], random_state=42)
        best_svm_model.fit(train_X_new, train_y_new)

    else:
        best_svm_model = SVC(kernel='linear', random_state=42)
        best_svm_model.fit(train_X_new, train_y_new)

    train_y_pred = best_svm_model.predict(train_x)

    print("Training Accuracy:", accuracy_score(train_y, train_y_pred))
    print("Training F1 Score:", f1_score(train_y, train_y_pred))

    valid_y_pred = best_svm_model.predict(valid_X)

    valid_accuracy = accuracy_score(valid_y, valid_y_pred)
    valid_f1 = f1_score(valid_y, valid_y_pred)

    print(f"Validation Accuracy: {valid_accuracy}")
    print(f"Validation F1 Score: {valid_f1}")

    return best_svm_model