import numpy as np
from numpy import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error




######### read_data #################

path='C:\\Users\\hecen\\Desktop\\USA_Housing_data.csv'
data = pd.read_csv (path,header=None,names=[' House Age','No.Rooms','No.Bedrooms','Price'])
print('data lenth is:', len(data))
data=data[0:1000]
######### Show Data ################
print('_______________________________________________________________________________________________________________')
print('data is : ')
print(data)
print('_______________________________________________________________________________________________________________')
print('data description :')
print(data.describe())
#### add ones column #####
data.insert(0,'ones',1)
######## seperate training data from Target variable ####3
columns=data.shape[1]
### data of independent variables
x=data.iloc[:,0:columns-1]
### data of dependent variable
y=data.iloc[:,columns-1:columns]
m = len(y)
print('Total no of training examples (m) = %s \n' %(m))

###### Compute Cost Func. ######
def compute_cost(x, y, theta):
    predictions = x.dot(theta)
    errors = np.subtract(predictions, y)
    J = 1 / (2 * m) * errors.T.dot(errors)
    return J
###### gradientDescent Func. ######
def gradientDescent(x, y, theta, alpha, iters):
    cost_history = np.zeros(iters)

    for i in range(iters):
        predictions = x.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * x.transpose().dot(errors);
        theta = theta - sum_delta;

        cost_history[i] = compute_cost(x, y, theta)
    return theta, cost_history
### variables for learning rate ####
alpha=0.1
iters=70000
theta=np.zeros(4)
#### Convert to matrices #######
x=np.asarray(x.values)
y=np.asarray(y.values)
### show matrices ###
print('_______________________________________________________________________________________________________________')
print('X Matrix is :')
print(x)
print(x.shape)
print('_______________________________________________________________________________________________________________')
print('Theta Matrix is :')
print(theta)
print(theta.shape)
print('_______________________________________________________________________________________________________________')
print('Y Matrix is :')
print(y)
print(y.shape)
print('_______________________________________________________________________________________________________________')
############ split the data to (Train & Test) #####
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)

########### transform data ############
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)
######## Regression model ##########
reg=LinearRegression()
reg.fit(x_train,y_train)
####### Main ######
def main():
    if __name__== "__main__" :
        ######## predicting test results########
        compute_cost(x,y,theta)
        gradientDescent(x,y,theta,alpha,iters)
        y_pred = reg.predict(x_test)
        print('mean squared eror is : ',math.sqrt(mean_squared_error(y_test, y_pred)))
        print('R_2 Score is : ',r2_score(y_test, y_pred))




main()








