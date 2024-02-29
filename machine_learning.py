# CODE FOR DATA ANALYSIS ----------------------------------------

# Import packages
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt

# Import data from csv file
data = pd.read_csv("../machine_learning/siren_data_train.csv", sep=",")
print(f"Data size: {data.memory_usage().sum() / 1e6:.2f} MB")
print("The first 10 rows in the data:")
data.head(10)

print("Overview of Numeric Variables")
numeric_columns = [col for col in data.columns]
data[numeric_columns].describe()

# Calculate the distance to nearest horn
list_loc_horn = []
list_loc_person = []
list_distance_to_horn = []

x_cor_horn = data["near_x"]
y_cor_horn = data["near_y"]

x_cor_person = data["xcoor"]
y_cor_person = data["ycoor"]

for row in range(len(x_cor_horn)):
    loc_horn = [x_cor_horn[row], y_cor_horn[row]]
    list_loc_horn.append(loc_horn)

    loc_person = [x_cor_person[row], y_cor_person[row]]
    list_loc_person.append(loc_person)

for i in range(len(x_cor_horn)):
    
    coordinate_horn = list_loc_horn[i] 
    coordinate_person = list_loc_person[i]
    distance_to_horn = math.dist(coordinate_horn,coordinate_person)
    list_distance_to_horn.append(distance_to_horn)
   
data["distance to nearest horn"] = list_distance_to_horn
print("The first 10 rows in the data (distance to nearest horn included):")
data.head(10)

# CODE FOR FIGURE 1 ----------------------------------------

# Create scatter plot which represents the relationship between whether people heard the signal and the distance to the nearest horn
list_heard = []
heard = data["heard"]
for row in range(len(x_cor_horn)):
    heard_yes_or_no = heard[row]
    list_heard.append(heard_yes_or_no)
  
x = list_distance_to_horn
y = list_heard

# compute the entry counts
u,c = np.unique(y, return_counts=True)

# color according to the counts
color = [c[np.where(u==y[i])][0] for i in range(len(y))]
print("Scatter plot of data, where the color bar represents the number of times a certain value of Y is repeated.")
plt.title("The relationship between whether people heard the signal and the distance to the nearest horn")
plt.scatter(x, y, s=100, c=color, cmap='PiYG', marker='o', edgecolors='black', linewidth=1, alpha=0.7)
plt.axis((0, 199000, -0.5, 1.5)) #(xmin, xmsx, ymin, ymax)
plt.colorbar()
plt.ylabel('Heard the siren (YES=1, NO=0)')
plt.xlabel('Distance to nearest horn')
plt.show()

# CODE FOR FIGURE 2 ----------------------------------------

total_heard = data['age'].value_counts()            #Total number of people in each age group

#How many heard and not in each age group and calculate percentage
heard_count = data.groupby('age')['heard'].sum()
not_heard_count = total_heard-heard_count
heard_percentage = (heard_count/total_heard)*100
not_heard_percentage = (not_heard_count/total_heard)*100

#Plot the figure
plt.figure(figsize=(10,6))
plt.bar(heard_percentage.index, heard_percentage, width=0.4, label='Heard')
plt.bar(not_heard_percentage.index + 0.4, not_heard_percentage, width=0.4, label='Not heard')
plt.xlabel('Age')
plt.ylabel('Percentage')
plt.title('The relationship between heard and age')
plt.legend()
plt.show()

# CODE FOR FIGURE 3 ----------------------------------------

# Create scatter plot thet represent the relationship between whether a person hear the siren and the direction towards the nearest horn
list_near_angle = []
agnle  = data["near_angle"]
for row in range(len(x_cor_horn)):
    near_agle_value = agnle[row]
    list_near_angle.append(near_agle_value)

x = list_near_angle
y = list_heard

# compute the entry counts
u,c = np.unique(y, return_counts=True)

# color according to the counts
color = [c[np.where(u==y[i])][0] for i in range(len(y))]
print("Scatter plot of data, where the color bar represents the number of times a certain value of Y is repeated.")
plt.title("The relationship between whether a person hear the siren and the direction towards the nearest horn")
plt.scatter(x, y, s=100, c=color, cmap='PiYG', marker='o', edgecolors='black', linewidth=1, alpha=0.7)
plt.axis((-250, 300, -0.5, 1.5)) #(xmin, xmsx, ymin, ymax)
plt.colorbar()
plt.ylabel('Heard the siren (YES=1, NO=0)')
plt.xlabel('Direction of nearest horn')
plt.show()

# CODE FOR LOGISTIC REGRESSION ------------------------------------

# Import packages
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn . model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,precision_score, recall_score, classification_report)
import seaborn as sns
from sklearn.model_selection import GridSearchCV

RANDOM_SEED = 1

# Importing the data from the csv file
training_data = pd.read_csv("../siren_data_train.csv", sep=",")
training_data['heard'] = training_data['heard'].replace({'hearing': 1, 'not_hearing': 0})
print('Data imported!')

# Adding the parameter 'distance to nearest horn'
training_data["distance_nearest_horn"] = list_distance_to_horn

# Dropping unnecessary features
training_data = training_data.drop(['near_x', 'near_y', 'xcoor', 'ycoor', 'near_fid', 'near_angle'], axis=1)
print(training_data)

# Creating training set and test set

# Creating X - everything about the column 'heard'
X = training_data.drop(['heard'], axis = 1)

# Creating Y - the column 'heard'
y = training_data['heard']

# Splitting the data into 75% for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=RANDOM_SEED, shuffle=True)
print("Data splitted!") 
print(X_train.head(10))

# Creating logisitc regression model
logistic_regression_model = lm.LogisticRegression()

# Parameter tuning
model = logistic_regression_model

# Performing a grid search for performing hyper-parameter optimization, finding the optimal combination of hyper-parameters
params = {'C':[0.001, 0.01, 0.1, 1, 10, 100],'penalty':['l1','l2'], 'solver':['liblinear', 'sag', 'saga']}
grid_search = GridSearchCV(model, params, cv=5)

# Fitting model to training data
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('Best parameters: ', best_params)

f1_scores = grid_search.cv_results_['mean_test_score']
print('f1:',f1_scores )

params = grid_search.cv_results_['params']
print("Params: ", params)

best_lr_model = grid_search.best_estimator_
accuracy = best_lr_model.score(X_test, y_test)
print("Accuracy: ", accuracy)

# Confusion Matrix
y_pred = best_lr_model.predict(X_test)

def print_performance_metrics(y_true, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred), "\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3}")
    print(f"Recall: {recall_score(y_true, y_pred):.3}")
    print(f"Precision: {precision_score(y_true, y_pred):.3}")
    print(f"F1: {f1_score(y_true, y_pred):.3}")

print_performance_metrics(y_test, y_pred)

# Clasification report
print("Logistic regression")
print(classification_report(y_test, y_pred, digits = 3))

# Testing with a naive model
naive = np.ones(y_test.shape[0])
print('Confusion Matrix:')
print(pd.crosstab(naive, y_test),'\n')

print(f"Accuracy: {np.mean(naive == y_test):.3f}")
print(f"F1: {f1_score(y_test,naive):.2}", '\n')

print("Logistic regression")
print(classification_report(y_test, y_pred, digits = 3))

# CODE FOR DISCRIMINANT ANALYSIS -------------------------

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sklearn.discriminant_analysis as skl_da
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import (f1_score, classification_report)

data["distance to nearest horn"] = list_distance_to_horn

# Split data in 75% training data and 25% test data
x = data[['building','noise','asleep','in_vehicle','no_windows','age','distance to nearest horn']]
y = data['heard']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 1, shuffle = True)

# Tuning model with GridSearchCV with parameters solver and shrinkage. Solver 'svd' does not support shrinkage and 'svd' is therefore removed
model = skl_da.LinearDiscriminantAnalysis()
params = {'solver':['lsqr','eigen'],'shrinkage':[None,'auto',0.5]}
grid_search = GridSearchCV(model, params, cv=5, scoring='f1')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
f1_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
best_gridsearch_model = grid_search.best_estimator_
accuracy = best_gridsearch_model.score(X_test, Y_test)
print(accuracy)
print(best_params)
print(params)
print(f1_scores)

# Tuning model with GridSearchCV with parameter solver. Solver 'svd' does not support shrinkage and shrinkage is therefore removed
model = skl_da.LinearDiscriminantAnalysis()
params = {'solver':['svd','lsqr','eigen']}
grid_search = GridSearchCV(model, params, cv=5, scoring='f1')
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
f1_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
best_gridsearch_solver_model = grid_search.best_estimator_
accuracy = best_gridsearch_model.score(X_test, Y_test)
print(accuracy)
print(best_params)
print(params)
print(f1_scores)

# Default model
model = skl_da.LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)
print(accuracy)

# Predict 
predict_prob = model.predict_proba(X_test)
print('The class order in the model:')
print(model.classes_)
print('Examples of predicted probablities for the above classes:')
with np.printoptions(suppress=True, precision=3): 
    print(predict_prob[0:5]) # First 5 predictions

# Evaluate model with confusion matrix
prediction = np.empty(len(X_test), dtype=object)
prediction = np.where(predict_prob[:, 0]>=0.5, 0, 1)  #labeling 0 or 1 depending on probabilities
print("First five predictions:")
print(prediction[0:5], '\n') # First 5 predictions after labeling

# Confusion matrix
print("Consufion matrix:")
print(pd.crosstab(prediction, Y_test),'\n')

# Accuracy
print(f"Accuracy: {np.mean(prediction == Y_test):.3f}")
print(f"F1:{f1_score(Y_test,prediction):.3}")
print(classification_report(Y_test,prediction, digits = 3))

# Naive model
naive = np.ones(Y_test.shape[0])
# Confusion matrix
print("Consufion matrix:")
print(pd.crosstab(naive, Y_test),'\n')
# Accuracy
print(f"Accuracy: {np.mean(naive == Y_test):.3f}")
print(f"F1:{f1_score(Y_test,naive):.2}")

# CODE FOR K_NN ----------------------------------------

# Import packages
import pandas as pd
import numpy as np
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

# Adding 'distance to nearest horn' to data and dropping unneccessary parameters
data['distance to nearest horn'] = list_distance_to_horn
data = data.drop(['xcoor','ycoor','near_x','near_y'],axis=1)

# Create test set
X = data.drop(['heard'], axis=1)
y = data['heard']
#print(y)

# Splitting into 75% training and 25% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1, shuffle=True)
print(X_test)

#Tuning model
params = {'n_neighbors': range(1,50)}
knn = KNeighborsClassifier ()

grid_search = GridSearchCV(knn, params, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(best_params)
f1_scores = grid_search.cv_results_['mean_test_score']
print('f1:',f1_scores )
params = grid_search.cv_results_['params']
print(params)
best_knn = grid_search.best_estimator_
accuracy = best_knn.score(X_test, y_test)
print('Accuracy: =', accuracy)

# Performance metrics
y_pred = best_knn.predict(X_test)

def print_performance_metrics(y_true, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred), "\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3}")
    print(f"Recall: {recall_score(y_true, y_pred):.3}")
    print(f"Precision: {precision_score(y_true, y_pred):.3}")
    print(f"F1: {f1_score(y_true, y_pred):.3}")
    

print_performance_metrics(y_test, y_pred)
print(classification_report(y_test, y_pred, digits=3))

# Naive model
naive = np.ones(y_test.shape[0])
# Confusion matrix
print("Confusion matrix:")
print(pd.crosstab(naive,y_test),'\n')
#Accuracy
print(f"Accuracy: {np.mean(naive == y_test):.3f}")
print(f"F1:{f1_score(y_test,naive):.2}")

# CODE FOR TREE-BASED METHODS ----------------------------------------
import pandas as pd
import numpy as np
import math
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, classification_report)
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

#Set constants
RANDOM_SEED = 1

# Adding distance to nearest horn
data["distance to nearest horn"] = list_distance_to_horn

# Create X and y
#features = [ "near_fid", "near_x", "near_y", "near_angle", "building", "xcoor", "ycoor", "noise", "in_vehicle", "asleep", "no_windows", "age"]
features = ['building','noise','asleep','in_vehicle','no_windows','age','distance to nearest horn']
X = data[features]
y = data[["heard"]]

# Create training data and held-out test data
X_train, X_heldout, y_train, y_heldout = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED, shuffle=True)

# Learn a decision tree
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Plot decision tree
def plot_tree_custom_size(classifier, figsize=(25, 20), fontsize=10):
    fig, ax = plt.subplots(figsize=figsize)
    plot_tree(classifier, fontsize=fontsize, ax=ax)
    ax.set_title("Decision tree")
    fig.tight_layout()

plot_tree_custom_size(decision_tree)

# Performance metrics
y_pred = decision_tree.predict(X_heldout)

def print_performance_metrics(y_true, y_pred):
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred), "\n")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3}")
    print(f"Recall: {recall_score(y_true, y_pred):.2}")
    print(f"Precision: {precision_score(y_true, y_pred):.2}")
    print(f"F1: {f1_score(y_true, y_pred):.2}")

print_performance_metrics(y_heldout, y_pred)
print(classification_report(y_heldout, y_pred, digits= 3))

# Trying parameter value max_depth=3
decision_tree_max_dept_3 = tree.DecisionTreeClassifier(max_depth=3)
decision_tree_max_dept_3.fit(X_train, y_train)

y_pred = decision_tree_max_dept_3.predict(X_heldout)

plot_tree_custom_size(decision_tree_max_dept_3)
print(classification_report(y_heldout, y_pred, digits= 3))

# Parameter tuning with grid search
from sklearn.model_selection import GridSearchCV
model = tree.DecisionTreeClassifier()
params = {"max_depth":np.arange(2, 10),"min_samples_leaf":np.arange(1, 14) ** 2}
grid_search = GridSearchCV(model, params, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(best_params)
f1_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
best_lda_model = grid_search.best_estimator_
accuracy = best_lda_model.score(X_heldout, y_heldout)

#print_performance_metrics(y_heldout, y_pred)
print(classification_report(y_heldout, y_pred, digits= 3))

# Parameter tuning
# Choice of parameters: max_depth, min_samples_leaf
# Using 5-fold cross-validation in order to decide the optimal values of the parameters max_depth and min_samples_leaf.

# Create 5-fold cross-validation
nk = 5
kf = KFold(n_splits=nk, random_state=RANDOM_SEED, shuffle=True)

# Search space of the parameters
max_depth_choices = np.arange(2, 10)
min_samples_leaf_choices = np.arange(1, 14) ** 2

# Learn an optimal decision tree model

param_choices = [
    {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
    for max_depth in max_depth_choices
    for min_samples_leaf in min_samples_leaf_choices
]

accuracy = np.zeros((nk, len(param_choices)))
f1 = np.zeros((nk, len(param_choices)))
recall = np.zeros((nk, len(param_choices)))
precision = np.zeros((nk, len(param_choices)))

for i, (train_index, val_index) in enumerate(kf.split(X_train)):
    X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    for j, param in enumerate(param_choices):
        dt = tree.DecisionTreeClassifier(
            max_depth=param["max_depth"], min_samples_leaf=param["min_samples_leaf"]
        )
        dt.fit(X_t, y_t)
        y_pred = dt.predict(X_val)
        accuracy[i][j] = accuracy_score(y_val, y_pred)
        f1[i][j] = f1_score(y_val, y_pred)
        recall[i][j] = recall_score(y_val, y_pred)
        precision[i][j] = precision_score(y_val, y_pred)

# Finding model with maximum F1 score
best_model = param_choices[np.argmax(np.mean(f1, axis=0))]
print("Selected model for F1:", best_model)

# Computing the performance of the best model 
dt = tree.DecisionTreeClassifier(**best_model)
dt.fit(X_t, y_t)
y_pred = dt.predict(X_heldout)
print_performance_metrics(y_heldout, y_pred)
print(classification_report(y_heldout, y_pred, digits= 3))
# max_depth = 5, min_samples_leaf = 9

# Naive model
print("Confusionmatrics Naive model")
only_heard = np.ones((y_heldout.shape[0]))
print(classification_report(y_heldout, only_heard, digits= 3))

#Random forest - combines the simplicity of decision trees with flexibility resulting in a vast improvement in accuracy

rf_baseline = RandomForestClassifier(random_state=RANDOM_SEED)
rf_baseline.fit(X_train, np.ravel(y_train))
rf_baseline_pred = rf_baseline.predict(X_heldout)

#print_performance_metrics(y_heldout, rf_baseline_pred)
print(classification_report(y_heldout, rf_baseline_pred, digits= 3))

from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameter tuning
# Choice of parameters: N_estimator, max_depth
# Using 5-fold cross-validation in order to decide the optimal values of the parameters N_estimator and max_depth.

# Create 5-fold cross-validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

# Search space of the parameters
param_choices = [
    {'max_depth':max_depth, 'n_estimators':num_est}
    for max_depth in np.arange(2, 10)
    for num_est in np.arange(20,180, 60)
]

accuracy = np.zeros((k, len(param_choices)))
f1 = np.zeros((k, len(param_choices)))
recall = np.zeros((k, len(param_choices)))
precision = np.zeros((k, len(param_choices)))

# Learn an optimal Random Forest model
def train_and_evaluate(classifier, train_index, val_index, param):
    X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
    rf = classifier(**param, random_state=RANDOM_SEED)
    rf.fit(X_t, np.ravel(y_t))
    y_pred = rf.predict(X_val)

    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
    }

def parallel_k_fold_cross_validation(classifier, param_choices):
    with ThreadPoolExecutor() as executor:
        future_to_params = {
            executor.submit(
                train_and_evaluate, classifier, train_index, val_index, param
            ): (i, j)
            for i, (train_index, val_index) in enumerate(kf.split(X_train))
            for j, param in enumerate(param_choices)
        }

        for future in as_completed(future_to_params):
            i, j = future_to_params[future]
            result = future.result()
            accuracy[i][j] = result["accuracy"]
            f1[i][j] = result["f1"]
            recall[i][j] = result["recall"]
            precision[i][j] = result["precision"]
    return {"accuracy": accuracy, "f1": f1, "recall": recall, "precision": precision}


results = parallel_k_fold_cross_validation(RandomForestClassifier, param_choices)
f1 = results["f1"]

# Finding model with maximum F1 score

def select_best_model(measure, results):
    mean_results = np.mean(results, axis=0)
    best_model_index = np.argmax(mean_results)
    return best_model_index

best_model_index = select_best_model("F1", f1)
best_model_params = param_choices[best_model_index]
print("Selected model for F1:", best_model_params)

# Computing the performance of the best model 
best_rf = RandomForestClassifier(
    max_depth=best_model_params["max_depth"],
    n_estimators=best_model_params["n_estimators"],
    random_state=RANDOM_SEED,
)
best_rf.fit(X_train, np.ravel(y_train))
y_best_rf_pred = best_rf.predict(X_heldout)

print_performance_metrics(y_heldout, y_best_rf_pred)
print(classification_report(y_heldout, y_best_rf_pred, digits= 3))

# max_depth = 5, n_estimator = 80

