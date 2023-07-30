import numpy as np
from numpy import math
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

######## read_data #################

path = 'C:\\Users\\hecen\\Desktop\\USA_Housing_data.csv'
data = pd.read_csv(path, header=None, names=[' House Age', 'No.Rooms', 'No.Bedrooms', 'Price'])
print(len(data))

######### Show Data ################
print('_______________________________________________________________________________________________________________')
print('data is : ')
print(data)
print('_______________________________________________________________________________________________________________')
print('data description :')
print(data.describe())
print('_______________________________________________________________________________________________________________')
############
columns=data.shape[1]
x=data.iloc[:,0:columns-1]
y=data.iloc[:,columns-1:columns]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=0)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
math.sqrt(mean_squared_error(y_test,y_pred))
print(r2_score(y_test,y_pred))
