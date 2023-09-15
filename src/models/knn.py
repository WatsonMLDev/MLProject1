from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

def prepare_model(train_x, train_y):

    # model selection: hyperparameter tuning
    hyperpara_grid = {'n_neighbors':[3, 5, 10, 15]} # candidate values for the hyperparameter to try
    base_model = KNeighborsClassifier()
    clf = GridSearchCV(base_model, hyperpara_grid, cv=5) # 5-fold cross validation
    clf.fit(train_x, train_y)

    #re-train model after finding the optimal hyper params:
    optimal_k = clf.best_params_['n_neighbors']
    final_model = KNeighborsClassifier(n_neighbors=optimal_k)
    final_model.fit(train_x, train_y)

    train_y_pred = final_model.predict(train_x)
    print(accuracy_score(train_y, train_y_pred))
    print(f1_score(train_y, train_y_pred))

    return final_model
