from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

# prepare features and labels for training/testing
train_X = my_train_df.drop(["HeartDisease", "PatientID"], axis=1)
train_y = my_train_df["HeartDisease"]
test_X = my_test_X_df.drop(["PatientID"], axis=1)

# define and fit your model, with manually set hyperparameter
# e.g., here is an example of KNN classifier, and you may tune the hyperparameter "n_neighbors"
model = KNeighborsClassifier(n_neighbors=10)
model.fit(train_X, train_y)

# evaluate accuracy/f1 score on training data
train_y_pred = model.predict(train_X)
print(accuracy_score(train_y, train_y_pred))
print(f1_score(train_y, train_y_pred))
# model selection: hyperparameter tuning
hyperpara_grid = {'n_neighbors':[3, 5, 10, 15]} # candidate values for the hyperparameter to try
base_model = KNeighborsClassifier()
clf = GridSearchCV(base_model, hyperpara_grid, cv=5) # 5-fold cross validation
clf.fit(train_X, train_y)
print(clf.cv_results_.keys()) # all results for 5-fold cross validation
print(clf.cv_results_['mean_test_score']) # average validation performance for different hyperparameter values
#re-train model after finding the optimal hyper params:

optimal_k = clf.best_params_['n_neighbors']
final_model = KNeighborsClassifier(n_neighbors=optimal_k)
final_model.fit(train_X, train_y)

train_y_pred = final_model.predict(train_X)
print(accuracy_score(train_y, train_y_pred))
print(f1_score(train_y, train_y_pred))
