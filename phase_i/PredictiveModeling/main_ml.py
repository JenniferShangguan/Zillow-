import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn import metrics, svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# general parameter settings
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)
np.random.seed(31415)

# column names
col_names = ['area_sqrt', 'bedroom', 'bathroom', 'type', 'year_built', 'heating', 'cooling', 'parking',
             'elementary_rating', 'middle_school_rating', 'high_school_rating', 'annual_tax_amount', 'zestimate']

# load dataset
rawData = pd.read_csv("input/data.csv", skiprows=1, names=col_names)

# remove heating, cooling, parking and annual_tax_amount columns due to too many NaNs
myData = rawData[rawData.columns[rawData.isnull().mean() < 0.4]]

# remove NaNs
origRecords = myData.shape[0]
myData = myData.dropna()
newRecords = myData.shape[0]
print('Remove records with NaN: {}, remaining records: {}'.format(origRecords, newRecords))

# split dataset in features and target variable
feature_cols = ['area_sqrt', 'bedroom', 'bathroom', 'type', 'year_built',
                'elementary_rating', 'middle_school_rating', 'high_school_rating']

X = myData[feature_cols]
y = myData.zestimate

# deal with nominal 'type' value
X = pd.concat((X, pd.get_dummies(X.type)), 1)
del X['type']

X = X.astype('int')
y = y.astype('int')
X = X.values
y = y.values

# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# pre-process training data, fit the scaler on training data only,
# then standardise both training and test sets with the scaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# ----------------- 1: Decision Tree ---------------------
base = DecisionTreeRegressor()

# tune parameter: max tree depth
scoring = metrics.make_scorer(metrics.mean_squared_log_error, greater_is_better=False)
max_depth_range = np.arange(20) + 1
tuned_params = {'max_depth': max_depth_range}
model = GridSearchCV(base, param_grid=tuned_params, scoring=scoring, cv=5, iid=False)

t_start = time.time()
model.fit(X_train, y_train)
t_end = time.time()
train_time = t_end - t_start
print('Training time: %f seconds' % train_time)

# find best fit parameters
best_dt_parameter = model.best_params_
print("Best max_depth parameter for decision tree: {}".format(best_dt_parameter))

t_start = time.time()
y_pred = model.predict(X_test)
t_end = time.time()
predict_time = t_end - t_start
print('Prediction time: %f seconds' % predict_time)

# evaluation
print("MSLE: {:.3f}".format(metrics.mean_squared_log_error(y_test, y_pred)))

# plot
plt.scatter(y_pred, y_test, c='r', s=10)
plt.xscale('log')
plt.yscale('log')
# plot a diagonal line
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([ymin, ymax], [ymin, ymax], 'b--')
plt.axis('scaled'), plt.xlabel("y_pred"), plt.ylabel("y_test"), plt.title('Decision Tree')
plt.xlim([ymin, ymax]), plt.ylim([ymin, ymax])
# plt.show()
plt.savefig('output/DT_pred_test.png')


# ---------------- 2: Random Forest ---------------------
base = RandomForestRegressor(n_estimators=200)

# tune parameter: max tree depth
scoring = metrics.make_scorer(metrics.mean_squared_log_error, greater_is_better=False)
max_depth_range = np.arange(20) + 1
tuned_params = {'max_depth': max_depth_range}
model = GridSearchCV(base, param_grid=tuned_params, scoring=scoring, cv=5, iid=False)

t_start = time.time()
model.fit(X_train, y_train)
t_end = time.time()
train_time = t_end - t_start
print('Training time: %f seconds' % train_time)

# find best fit parameters
best_dt_parameter = model.best_params_
print("Best max_depth parameter for random forest: {}".format(best_dt_parameter))

t_start = time.time()
y_pred = model.predict(X_test)
t_end = time.time()
predict_time = t_end - t_start
print('Prediction time: %f seconds' % predict_time)

# evaluation
print("MSLE: {:.3f}".format(metrics.mean_squared_log_error(y_test, y_pred)))

# plot
plt.scatter(y_pred, y_test, c='r', s=10)
plt.xscale('log')
plt.yscale('log')
# plot a diagonal line
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([ymin, ymax], [ymin, ymax], 'b--')
plt.axis('scaled'), plt.xlabel("y_pred"), plt.ylabel("y_test"), plt.title('Random Forest')
plt.xlim([ymin, ymax]), plt.ylim([ymin, ymax])
# plt.show()
plt.savefig('output/RF_pred_test.png')


# ---------------------- 3: Boosting -----------------------
base = DecisionTreeRegressor(max_depth=17)
model = AdaBoostRegressor(base_estimator=base, n_estimators=200, random_state=1)

# train
t_start = time.time()
model = model.fit(X_train, y_train)
t_end = time.time()
train_time = t_end - t_start
print('Training time: %f seconds' % train_time)

# predict
t_start = time.time()
y_pred = model.predict(X_test)
t_end = time.time()
predict_time = t_end - t_start
print('Prediction time: %f seconds' % predict_time)

# evaluation
print("MSLE: {:.3f}".format(metrics.mean_squared_log_error(y_test, y_pred)))

# plot
plt.scatter(y_pred, y_test, c='r', s=10)
plt.xscale('log')
plt.yscale('log')
# plot a diagonal line
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([ymin, ymax], [ymin, ymax], 'b--')
plt.axis('scaled'), plt.xlabel("y_pred"), plt.ylabel("y_test"), plt.title('AdaBoost')
plt.xlim([ymin, ymax]), plt.ylim([ymin, ymax])
# plt.show()
plt.savefig('output/AdaBoost_pred_test.png')

# # evaluation: learning curve
# scoring = metrics.make_scorer(metrics.mean_squared_log_error, greater_is_better=False)
# train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, scoring=scoring,
#                                                         train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
#
# train_scores = -train_scores
# test_scores = -test_scores
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
#
# plt.figure()
# plt.plot(train_sizes, train_mean,  label="Training error", color="r")
# plt.plot(train_sizes, test_mean, label="Cross validation test error", color="b")
# plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1)
# plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="b", alpha=0.1)
# plt.title('AdaBoost learning curve')
# plt.xlabel("Training set size"), plt.ylabel("MSLE"), plt.legend(loc="best")
# plt.tight_layout()
# plt.legend(loc="best")
# # plt.grid(b=None)
# plt.grid(axis='y')
# # plt.show()
# plt.savefig('output/AdaBoost_learning_curve.png')


# ---------------------- 4: SVM -----------------------
base = svm.SVR()

# hyperparameter tuning
parameters = {'C': np.logspace(np.log10(10000), np.log10(100000), num=10),
              'gamma': np.logspace(np.log10(0.01), np.log10(1), num=10)}
scoring = metrics.make_scorer(metrics.mean_squared_log_error, greater_is_better=False)
model = GridSearchCV(base, parameters, n_jobs=4, scoring=scoring, cv=5, iid=False)

# training
t_start = time.time()
model.fit(X_train, y_train)
t_end = time.time()
train_time = t_end - t_start
print('Training time: %f seconds' % train_time)

# find best fit parameters
best_parameter = model.best_params_
print("Best size parameter for SVM: {}".format(best_parameter))

# predict
t_start = time.time()
y_pred = model.predict(X_test)
t_end = time.time()
predict_time = t_end - t_start
print('Prediction time: %f seconds' % predict_time)

# evaluation
print("MSLE: {:.3f}".format(metrics.mean_squared_log_error(y_test, y_pred)))

# plot
plt.scatter(y_pred, y_test, c='r', s=10)
plt.xscale('log')
plt.yscale('log')
# plot a diagonal line
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([ymin, ymax], [ymin, ymax], 'b--')
plt.axis('scaled'), plt.xlabel("y_pred"), plt.ylabel("y_test"), plt.title('SVM')
plt.xlim([ymin, ymax]), plt.ylim([ymin, ymax])
# plt.show()
plt.savefig('output/SVM_pred_test.png')


# ---------------------- 5: KNN -----------------------
base = KNeighborsRegressor()

# tune parameter: k size
k_range = np.arange(1, 51)
scoring = metrics.make_scorer(metrics.mean_squared_log_error, greater_is_better=False)
tuned_params = {'n_neighbors': k_range}
model = GridSearchCV(base, param_grid=tuned_params, scoring=scoring, cv=5, iid=False)

# training
t_start = time.time()
model.fit(X_train, y_train)
t_end = time.time()
train_time = t_end - t_start
print('Training time: %f seconds' % train_time)

# find best fit parameters
best_parameter = model.best_params_
print("Best size parameter for KNN: {}".format(best_parameter))

# prediction
t_start = time.time()
y_pred = model.predict(X_test)
t_end = time.time()
predict_time = t_end - t_start
print('Prediction time: %f seconds' % predict_time)

# evaluation
print("MSLE: {:.3f}".format(metrics.mean_squared_log_error(y_test, y_pred)))

# plot
plt.scatter(y_pred, y_test, c='r', s=10)
plt.xscale('log')
plt.yscale('log')
# plot a diagonal line
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([ymin, ymax], [ymin, ymax], 'b--')
plt.axis('scaled'), plt.xlabel("y_pred"), plt.ylabel("y_test"), plt.title('KNN')
plt.xlim([ymin, ymax]), plt.ylim([ymin, ymax])
# plt.show()
plt.savefig('output/KNN_pred_test.png')


# ------------------ 6: Neural network --------------------
model = MLPRegressor(hidden_layer_sizes=(100, 200, 100), random_state=1)

# training
t_start = time.time()
model.fit(X_train, y_train)
t_end = time.time()
train_time = t_end - t_start
print('Training time: %f seconds' % train_time)

# prediction
t_start = time.time()
y_pred = model.predict(X_test)
t_end = time.time()
predict_time = t_end - t_start
print('Prediction time: %f seconds' % predict_time)

# evaluation
print("MSLE: {:.3f}".format(metrics.mean_squared_log_error(y_test, y_pred)))

# plot
plt.scatter(y_pred, y_test, c='r', s=10)
plt.xscale('log')
plt.yscale('log')
# plot a diagonal line
xmin, xmax, ymin, ymax = plt.axis()
plt.plot([ymin, ymax], [ymin, ymax], 'b--')
plt.axis('scaled'), plt.xlabel("y_pred"), plt.ylabel("y_test"), plt.title('NN')
plt.xlim([ymin, ymax]), plt.ylim([ymin, ymax])
# plt.show()
plt.savefig('output/NN_pred_test.png')

