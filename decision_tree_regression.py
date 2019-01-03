# Decision Tree REGRESSION (Decision Tree CLASSIFICATION found in decision_tree_classification.py)

# -----------------------------------Decision Tree Intuition------------------------------------------
#Example) Consider a scatter plot that represents some dataset with two indepedent variables x_1 and x_2
# predicting the third variable y (where y is the third dimension). We do not need to be able to visualize y at first.
# We can begin by building our decision tree using the scatter plot, then after we build it, return to y.
# Once we run the decision tree regression algorithm, the scatterplot will be split up into segments, how and where
# the splits are conducted is determined by the algorithm (by mathematical information entropy....complex)
# Boiled down: When the algorithm performs the split, its asking, is the split increasing the amount of information we have
# about our points. Are the splits adding value? how do we want to group our points?
# The algorithm knows when to stop when there is a certain minimum for the information that needs to be added and once
# it cannot add any more information (its reached the minimum), it stops SPLITTING the LEAVES (each split is called
# a leaf). For example, a decision_tree_regression would stop if WHEN we conducted a split,
# that split or leaf would have less than 5% of the total scattered points, then that leaf wouldn't be created.
# Final leaves are called TERMINAL leaves.
# By adding these leaves, we've added information into our system. how does that help us predict the value of y?
# You just take the averages of each of the terminal leaves; you take the average of the y-values for all of the 
# points in a particular leaf and thats your y_pred which is assigned to any new data point in that leaf and is done
# so using a decision tree, hence the name. :) 
# 


#Information entropy is the average rate at which information is produced 
#by a stochastic source of data. The measure of information entropy associated
#with each possible data value is the negative logarithm of the probability 
#mass function for the value...see wikipedia


# Importing the libraries
import numpy as np #import standard data libraries
import matplotlib.pyplot as plt #import matplotlib.pylot as plt
import pandas as pd #import pandas as pd

# Importing the dataset make sure the current directory is switched
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree Regression to the dataset
#Creating our decision tree regressor
from sklearn.tree import DecisionTreeRegressor #importing DecisionTreeRegressor class from sklearn.tree
regressor = DecisionTreeRegressor(random_state = 0) #creating our DecisionTreeRegressor object called regressor
# default criterion = mse or mean squared error (so we are taking the squared difference between the prediction
# and the actual result, and then taking the sum of those distances to measure the error. very good and common criterion)
# then you have some other parameters like splitter and max_features which is for a more advanced approach of how
# to build the decision tree. (there's a lot of parameters you can choose)
# we let the random state = 0 so that we all get the same result..for example purposes
regressor.fit(X, y) #final step is to FIT the regressor object to our dataset
#regressor.fitMethod(X our matrix of features, y our dependent variable vector)

# Predicting a new result
y_pred = regressor.predict(6.5) 

#Now saving the best part for last...the regression results VISUALIZED
#It should have the appearance of a piecewise function...model is non-continuous
#the value for a prediction should be a constant average for each prediction on each interval
# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01) #you need to do this so that we get a higher resolution picture
X_grid = X_grid.reshape((len(X_grid), 1)) # if you graph using basic technique, you'll get something that looks like
plt.scatter(X, y, color = 'red') # a linear regression model...so we "PLOT" more points
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show() 
#As you can see, the decision tree regression model is predicting the average

#Check out random forest next. They're pretty much a team of decision trees! :)