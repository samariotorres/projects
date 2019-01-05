# ----------------------------SVR INTUTION--------------------------
#SVR performs linear regression in a higher dimensional space
#You can think of SVR as if each data point in the training set, represents its own 
# dimension. When you are evaluating your kernel between a test point and a point in the
# training set, the resulting value gives you the coordinate of your test point in that 
# dimension.

#The vector we get when we evaluate the test point for all points in the training set,
#Typically denoted as the, k_vector, is the representation of the test point in the higher
# dimensional space

#You're pretty much trying to create k_vectors inside a certain space,
#where that space is constrained by some thresholds, epsilon_i and zeta_i parameters.

#The work of the SVR is to approximate the function we used to generate the training set.

#In a classification problem, the set of vectors X_vectorAbove are used to define the a hyperplane
#that separates the two different classes in your solution 
# These vectors perform linear regression. The vectors closest to the test point are refererred
# to as SUPPORT VECTORS. We can evaluate our function, ANYWHERE, so any vectors could 
# be closest to our test evaluation location.

#-------------------------------Building an SVR---------------------------------
# 1.) Collect a training set 'tao' = [X_vectorAbove,Y_vectorAbove]
# 2.) Choose a kernel and its parameters as well as any regularization needed
# 3.) Form the correlation matrix, K_vectorAbove 
# 4.) Train the machine, exactly or approximately, to get the contraction coefficients,
#       alpha_vectorAbove = {alpha_i}
# 5.) Use those coefficients to create your estimator, 
#               f(X_vectorAbove, alpha_vectorAbove, x^*) = y^*
# 6.) Choose a kernel: 
#       Gaussian kernel: A kernel that approaches zero as the distance between the arugments
#                       grow as you move away from the training data, the machine
#                       will return the mean value of the training data.
# In addition to choosing the kernel, regularization is also important.
# Because due to the training sets with noise, the regularizer will help prevent wild
# fluctuations between the data points by smoothing out the prior.

# We evaluate our kernel at the correlation matrix, for all pairs of points in the 
# training set and adding the regulizer resulting in the matrix
# Main part of alogrithm: K_vectorAbove * alpha_vectorAbove = y_vectorAbove
# Then you left-hand multiply both sides of the equation and get 
# alpha_vectorAbove = (K_vectorAbove)^-1 * y_vectorAbove

# Once we know the alpha parameters, we form the estimators
# Then use the coefficients we found during the optimization step and the kernel we started off
# with.
# To estimate the value y^* for a test point, x_vectorAbove^*, then compute the correlation
# vector k_vectorAbove and you get, y^* = alpha_vectorAbove * k_vectorAbove 

#Overall: SVR has a different regression goal compared to linear regression. In linear regression,
# we are trying to minimize the error between the prediction and data.

# In SVR, our goal is to make sure the errors do not exceed the threshold. SVR classifies all
# of the linear predictions into two types; the predictor lines that pass through the error
# lines drawn, and the predictor lines that do not. The lines that do not pass through those
# error lines are not acceptable because it means the difference, or the distance between the
# prediction points and the actual values at those points have exceeded the error threshold.


# Once you have that k_vector, you then use the k_vector to performa linear regression
# Importing the libraries
import numpy as np #import numpy as np
import matplotlib.pyplot as plt #import matplotlib.pyplot as plt
import pandas as pd #import pandas pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling #SVR class is a less common class so perhaps thats why the class
# does not include feature scaling..which is why we include this piece of code below..
# If we do not do this, our SVR is a straight line and incorrect
# Note there is no test set here because we are not using one because of the smaller size
# of data, but if the data was larger we would be doing this with a test set
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y) 

# Fitting SVR to the dataset, creating our SVR regressor
from sklearn.svm import SVR # SVR is actually an SVM but for regression from the SVM class
regressor = SVR(kernel = 'rbf') #the actual regressor object, using the gaussian kernel, rbf
# the default kernel is rbf. also our data is non linear, so good idea to use rbf
regressor.fit(X, y) #fitting the regressor to our matrix of features X, and y the dependent
#variable vector

# Predicting a new result
y_pred = regressor.predict(sc_X.transform(np.array([[6.5]]))) #if you only put single brackets,
# it will be a vector, you need double brackets to make it a matrix
y_pred = sc_y.inverse_transform(y_pred)# use this because we want the original scale of 
# the salary so we apply the inverse transform method to the scalar object,  

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
