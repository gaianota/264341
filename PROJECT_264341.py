#First step is to import useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
#Import dataset
sms = pd.read_csv('social_media_shares.csv')
#Divide data in dependent and independent
X = sms.iloc[:,:-1]
y = sms.iloc[:,-1]
#We proceed with a Explanatory data analysis (EDA) with visualization.

print(sms.describe())
print(sms.head())
print(sms.info())
print('Social media share dataset shape is', sms.shape)

#Plot heat map to see the correlation among the variables
corr = sms.corr()
plt.figure(figsize = (15,8)) #To set the figure size
sns.heatmap(data=corr, square=True, annot=True, cbar=True)

#We plot the relation between each regressor variable with the response variable "shares"
for col in X.columns:
  sns.scatterplot(data=sms,x=col,y='shares')
  plt.show()
#We can see that there is no linear correlation between the regressor and response variables!

for columns correlation
  upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))

to_drop = list([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)])

X = X.drop(to_drop, axis=1)
# X is now with 56 variables because we removed the useless ones
print(X.shape)

# wetransform outliers in null values

print(X.info())
print(X.isnull().sum())
new_sms = X.assign(shares=y)
new_sms = new_sms.dropna(axis=0)
print(new_sms.info())
X = new_sms.iloc[:, :-1]
y = new_sms.iloc[:, -1]

print('New:', new_sms.shape)

from sklearn.model_selection import train_test_split

X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.8, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=0.2, random_state=42)
print("X:train shape: ", X_train.shape)
print("X:valid shape: ", X_valid.shape)
print("X:test shape: ", X_test.shape)

# Feature engineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

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
# we now scale
# X_train = sc.fit_transform(X_train)
# X_valid = sc.transform(X_valid)
# X_test = sc.transform(X_test)
