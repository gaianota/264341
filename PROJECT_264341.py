#First step is to import useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
#Import dataset\\
sms = pd.read_csv('social_media_shares.csv')
#Divide data in dependent and independent
X = sms.iloc[:,:-1]
y = sms.iloc[:,-1]
#TASK 1
#We proceed with a Explanatory data analysis (EDA) with visualization.
print(sms.describe())
print(sms.head())
print(sms.info())
print('Social media share dataset shape is', sms.shape)

#Plot heat map to see the correlation among the variables
corr = sms.corr()
print(corr)
plt.figure(figsize = (15,8)) #To set the figure size
sns.heatmap(data=corr, square=True, annot=True, cbar=True)

#We plot the relation between each regressor variable with the response variable "shares"
for col in X.columns:
  sns.scatterplot(data=sms,x=col,y='shares')
  plt.show()
#We can see that there is no linear correlation between the regressor and response variables!


upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))

to_drop = list([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)])

X = X.drop(to_drop, axis=1)
# X is now with 56 variables because we removed the useless ones
print(X.shape)

# we transform outliers in null values

print(X.info())
print(X.isnull().sum())
new_sms = X.assign(shares=y)
new_sms = new_sms.dropna(axis=0)
print(new_sms.info())
X = new_sms.iloc[:, :-1]
y = new_sms.iloc[:, -1]

print('New:', new_sms.shape)
#TASK 2
#Generate a training and test set. The test set should be used only at the end.
#Splitting
from sklearn.model_selection import train_test_split
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.2, random_state=42)
print("X:train shape: ", X_train.shape)
print("X:valid shape: ", X_valid.shape)
print("X:test shape: ", X_test.shape)

# Feature engineerin
#TASK 3
# Preprocess the dataset (remove outliers, encode categorical features with one hot encoding, not necessarily in this order)

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
#We do this in order to avoid scaling columns that have just a binary value (0,1),(as if the One Hot Encoder has done)
#We save the index of the columns to now which columns need to
index_k = X.columns.get_loc('keywords')
index_w = X.columns.get_loc('world')
index_m = X.columns.get_loc('monday')
index_is = X.columns.get_loc('is_weekend')
index_len = len(X.columns)

pipeline = ColumnTransformer([
  ('num', RobustScaler(), list(range(0, index_k + 1))),
  ('num1', RobustScaler(), list(range(index_w + 1, index_m))),
  ('num2', RobustScaler(), list(range(index_is, index_len))),

], remainder='passthrough')
sc = StandardScaler()
X_train = pipeline.fit_transform(X_train)
X_valid = pipeline.transform(X_valid)
X_test = pipeline.transform(X_test)

#Test at least 3 different regressors.
# First, create a validation set from the training set to analyze the behaviour with the default hyperparameters.
#Then use 10-fold cross-validation to find the best set of hyperparameters.
#You must describe every hyperparameter tuned (the more, the better).

from sklearn.metrics import mean_squared_error,mean_absolute_error
# Fitting Random Forest Regression to the dataset
# import the regressor
from sklearn.ensemble import RandomForestRegressor
# create regressor object
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
# fit the regressor with x and y data
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_valid)  # test the output
print(mean_squared_error(y_valid, y_pred, squared=False))
print(mean_absolute_error(y_valid, y_pred))

from sklearn.svm import SVR
svr =  SVR()
svr.fit(X_train,y_train)
y_pred = svr.predict(X_valid)  # test the output
print(mean_squared_error(y_valid, y_pred,squared=False))
print(mean_absolute_error(y_valid, y_pred))

#GRID SEARCH with cross validation
#esplor the parameters to find the best combination
#Cross validation for Random Forest
from sklearn.model_selection import cross_val_score, GridSearchCV
grid ={'max_depth': range(3, 7),'n_estimators': (10, 50, 100)}
rfr_grid = GridSearchCV(regressor,grid,cv=10, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
rfr_grid_result = rfr_grid.fit(X_train, y_train)
rfr_best_params = rfr_grid_result.best_params_
rfr = RandomForestRegressor(criterion=best_params['criterion'],max_features = best_params['max_features'],max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],random_state=False, verbose=False)
# Perform K-Fold CV
scores = cross_val_score(rfr, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
print(scores)

def best_scores_grid_search_df(reg):
    reg.cv_results_
    grid_table = pd.DataFrame(reg.cv_results_)
    colums_wanted = ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    grid_table_rank = grid_table[colums_wanted].sort_values(by='rank_test_score', ascending=True)
    return grid_table_rank
RF_best_scores= best_scores_grid_search_df(rfr_grid)
print(RF_best_scores)

from sklearn.linear_model import LinearRegression
## SVR
#First we find the best combination of hyperparameters
from sklearn.svm import SVR
param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [np.power (10., a) for a in range(-1,4)]}
reg_svr = GridSearchCV(svr, param_grid, cv=10)
reg_svr.fit(X_train,y_train)
SVM_best_scores=best_scores_grid_search_df(reg_svr).head()
print(SVM_best_scores)
