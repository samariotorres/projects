# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression #from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures #from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #create an object of this class
# the degree on the polynomial specifys the degree of the independent variable 

X_poly = poly_reg.fit_transform(X) #poly_reg is a tool that will transform the matrix of features
#X into this matrix of features, X_poly, by adding the additional polynomial terms into the matrix X,
#and since the degree equals 4 we are adding 3 extra polynomial terms! We use fit_transform
# because we are FIRST fitting our object to X AND THEN transforming X into X_poly
# It included the column of ones just in case we need it to represent our constant b_0
poly_reg.fit(X_poly, y) #now we're going to actually include this FIT into a multiple linear
#regression model ...
lin_reg_2 = LinearRegression() #here we are creating a new linearRegression object to not make any
# confusion with the non_Poly lin_reg object
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1) #creates resolution of 0.1 and also gives us a vector
# but we need a matrix so we use the reshape method below
X_grid = X_grid.reshape((len(X_grid), 1)) #we use the reshape method from numpy
#So now X_grid is a matrix that contains the values from 0 to 1 incrementing by 0.1 every time

#the two lines above just make the curve look a little more continuous and pretty by changing
#the scale at which the grid or graph is drawn :)

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') 
#so on line 69, we changed the first parameter, X, to X_grid, so that our model now predicts all of the 
# salaries of this imaginary 90 levels from 1 to 10.
#We are predicting on the FITTED TRANSFORM 
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5) #we can put an actual value we want to compute into this lin_reg.predict

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
