# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv') #reading the dataset dataset = pd.read_csv('50_StartUps.csv')
X = dataset.iloc[:, :-1].values#Selecting all rows of the dataset and all columns except the last
#X is our matrix of independent variables needs to be typed in console because its an object
y = dataset.iloc[:, 4].values#selecting the 5th column to be our dependent variable vector

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #importing labelencoder and onehot 
#so we can get change categorical values to numerical values
labelencoder = LabelEncoder() #creating our labelencoder object from LabelEncoder Class
X[:, 3] = labelencoder.fit_transform(X[:, 3]) #Setting the column that we want encoded in this case
# we want to change the values for the States column
onehotencoder = OneHotEncoder(categorical_features = [3]) #now we do not want the model to treat one of the
#states as literally being greater than the other...so we give them binary values 
X = onehotencoder.fit_transform(X).toarray() #then we need to fit_transform those values 
#X = onehotencoder.fit_transform(X).toarray()



# Avoiding the Dummy Variable Trap
X = X[:, 1:] #this means we are taking the entire matrix except for the column with index 0.
#for some libraries, this is done automically, others no, but we can do it anyways
#we do this so that our dataset does not contain any redundant dependencies

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#test size = 0.2 means 20% goes to the test set
# and 80% goes to the training set

# Feature Scaling
#we don't need to apply feature scaling for multiple linear regression
#because the library takes care of that for us
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


"""--------Data Preprocessing Phase Complete----------------"""
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression #exact same thing for linear regression
regressor = LinearRegression() #creating our regressor object from the LinearRegression class
#so that now we can use all the stuff inside LinearRegression class

regressor.fit(X_train, y_train) #now we are going to fit this object, "regressor", to our
#training set by using the "fit" method applied to our training set



# Predicting the Test set results
#Below, we are now going to do the prediction on the test set results
#It is exactly the same for simple linear regression and actually its the same for any type of
# regression
# So we are creating the vector of predictions called y_pred
# and then use the regressor object and the predict method from the LinearRegression class
# to predict the observation of the test set which in this case is X_test

y_pred = regressor.predict(X_test)

"""When we built this model, we used all of the independent variables, but what if among these independent
variables, there are some that are highly statistically significant and vice versa"""

"""We want to find an optimal team of independent variables...we want an OPTIMAL MODEL"""
"""We're going to look into that below; finding a team of optimal independent variables"""


# this is backward elimination

import statsmodels.formula.api as sm 

#in the linear equation where you have some constants plus a bunch of independent variables with 
#some coefficient, you have that constant by itself.
#This is a problem because if thats the case and there is not variable next to that constant
#the model in this case will not take the constant into account
#so the model understands what the actual linear equation is supposed to look like
#so to take care of that, we are going to add this columns of 1's to the matrix of features

X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#So what we just did in the line above: the parameter "arr" (array) is to choose the matrix you want
# to add something to..and the parameter "values" takes in the array that you are going to add to the
# "arr" array. So if you what we are doing in the code above is appending an array with 
#50 rows and 1 column to the beginning of matrix X...we could have appended it to the end by 
# just switching what we currently have equal to "arr" and what we currently have equal to "values"

#We did this because it is required by the stats models library which is why we did this
#We did appended this column of ones to X to proceed with backward elimnation

#Starting backward elimination below

#First thing we're going to do is create our new matrix of features for optimal features called
# X_opt...the independet variables that have high impact

#Backward elimination consists of including all of the independent variables at first
# and then remove one by one the indep. vars. that are not significant

X_opt = X[:, [0, 1, 2, 3, 4, 5]] # taking X with all independent variables

# and we want to specifically write the indicices of X so that we are ready to pop them off
# when we need to

#So now we are going to fit our multiple linear regression model to our future optimal features X
#and y because we will need 'y' to actually make this fit

#new library statsmodels going to be used

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #we just fitted ordinary least squares 
#(multiple linear regression algorithm) to X_opt and y

#Above was step 2 in the algo

#Now step 3, looking for the predictor using the summary function which provides statistical matrix

regressor_OLS.summary()

#So since the p-value for x2 was SO HIGH, we're going to remove by doing what
# is below, just taking it out of X_opt and then fitting it again

X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#removing the 1th index because its p-value is SO HIGH now
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

#removing the 2nd index because its p-value is SO HIGH
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 

#the dummy variables for X will NOT be part of the final optimal team is going to be just one
#independent variable which in this case, is the R&D Spending for the Start up!!!

#FYI if we wanted to plot a visual graph it would be difficult because we would need five
# dimensions.

#Below are some automatic implementations of Backward Elimination in Python
"""
Backward Elimination with p-values only:

import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
Backward Elimination with p-values and Adjusted R Squared:

import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)

"""
