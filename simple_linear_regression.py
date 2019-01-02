# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values #because our matrix of independent features consist of one column, in this
#case its just the first column so we take all the rows except for the last column
y = dataset.iloc[:, 1].values #our dependent variable(s) in this case is just the last column,
#the salary column so we need the 1th index because there's only 2 column but indices start at 0

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#test size is going to be 10 and training set size is going to 20 which is 2/3 of 30

# Feature Scaling
#We dont need to apply feature scaling because the library for simple regression takes care
#of that for us.
#So we'll leave the stuff below commented out
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to the Training set
# need to import linear_model from sci-kit learn and then import LinearRegression class
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #creating an object of the LinearRegression class so we can use its methods
regressor.fit(X_train, y_train) #using the fit method to fit the Regressor object to the 
#training set
#the paramenters for fit are datasets, and in this case the data sets that we want to fit are 
#the X-train and y-train datasets, y-train are the target values

#So we just made the regressor machine, learn on the training set so that the regressor machine will
#be able to predict values based on its experience on the training sets 

# Predicting the Test set results
#y_pred is going to be the vector of predictions for the observations of our test set
y_pred = regressor.predict(X_test) #and we use the predict method specifically on the X_test
#set because we want to TEST our prediction on the TEST set obviously

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')#makes a scatter plot on the observation points(training set)
#on the actual numbers from the actual dataset 
#the parameters are 1st the values for the x-axis, 2nd parameter is values to go along y-axis, and 
#3rd parameter is colors we want the points in


plt.plot(X_train, regressor.predict(X_train), color = 'blue')#now we are going to plot the regression
#line, 1st parameter values to go along the x-axis, 2nd parameter is values to go along y-axis,
#3rd parameter is the color we want for the line
#So what we are plotting here is the prediction for the training set to see how the 
#regressor machine did on the training set. so we should not confuse regressor.predict(X_train)
#with y_pred = regressor.predict(X_test). There's a difference between predicting on the training set
#and predicting on the test set!
plt.title('Salary vs Experience (Training set)') #putting a title on the graph
plt.xlabel('Years of Experience') #labeling the x-axis the # of years of experience
plt.ylabel('Salary') #labeling the y-axis with salary amount
plt.show() #this is how you generate the actual graph: plt.show()

# Visualising the Test set results

#Now we're going to see how the regressor machine compares to the test set
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')#we do not need to change the X_train 
#to X_test because our regressor is already trained on the training set 
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


#on line 35, when we trained our simple linear regression on the training set we obtained one unique model
#equation which is the simple linear equation itself

# we did not need to change the training_sets to test_sets because its
  #going to use the same regressor machine as before..once we find the lm model regressor line its the one thats 
  #going to be used for everything
