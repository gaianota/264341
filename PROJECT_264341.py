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
#Visualizing the dataset info
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


