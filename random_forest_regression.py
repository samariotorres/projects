# Random Forest Regression
# Ensemble Learning: Random forest is a version of ensemble learning, there are other version such as gradient 
# boosting. Ensemble learning is when you take multiple algorithms or the same algoirthm mutliple times and you put
# them together to make something much more powerful than the original.
# 
#
# --------------------------------------Random Forest Intuition--------------------------------
# Step 1.) Pick at random from the data points, 'K' data points from the Training set
# Step 2.) Then build a Decision Tree associated to these 'K' data points,
# Rather than building a decision tree based on everything in your dataset, we're just building a decision tree
# based on THAT SUBSET of data points
# Step 3.) Then you choose the number of trees you want to build and then repeat steps 1 and 2...
# so you're building lots of regression trees. building building building trees.
# Step 4.) And then finally we're going to use each one of the N-trees to predict the value of 'y' for the data
# point in question. and then assign the "new data point or the "data point being predicted"", 
# the average across all of the predicted y-values. So instead of
# just getting one prediction, you're getting lots of predictions. by default these algorithms are set to about
# 500 trees at least...so you're getting 500 predictions for the value of 'y', and then you're taking the 
# average across those. So now, instead of predicting just on one tree, we're going to be predicting based
# on a forest of trees; which improves the accuracy of the prediction; its because you're taking the average
# of many predictions and therefore even if one of the predictions is somehow bad, its less likely to get 
# a bad prediction. Also, ensemble algorithms are more stable, because any changes in your dataset,
# could really impact one tree, but for them to REALLY impact a FOREST of trees is much harder, which is way where
# ensemble is strong.
#  

# Random forest regression is a non-continuous regression because its a combination of some non-continuous,
# regression models that are the decision trees themselves.
# Importing the libraries
import numpy as np #import numpy as np
import matplotlib.pyplot as plt #import matplotlib.pylot as plt
import pandas as pd #import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv') #dataset = pd.read_csv('Position_Salaries')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

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

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor # importing the right class for the job, RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0) #creating the object for the class
#where n_estimators is the number of trees you want in your forest. by default is 10 trees. second parameter is
# criterion, but rememeber by default it is MSE, the sum of the squared differences of the predicted values
# and the actual values, it will work great here. And other parameters. Again, we use random_state = 0
# so we generate similar values for example purposes.

regressor.fit(X, y) # and finally we to FIT the regressor to the dataset using the FIT METHOD

# Predicting a new result
y_pred = regressor.predict(6.5)

#something important to point out: If we add a lot more trees in our random forest, it doesn't mean we'll
# get a lot more steps in the stairs. Because the more trees you add, the more the average of the different
# predictions made by the trees is converging to the same average...based on the same technique, 
# entropy and information gain. So the more you add trees, the more that the average of these votes will converge
# to the same ultimate average. So visually, it will converge to some shape of stairs.
# However, just because you have a specific shape of stairs, does not mean the prediction is stuck in one form.
# If you continue to increase the number of trees, dont forget, the steps might look the same, but maybe now,
# they're better chosen; better aligned along with axes.

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01) #something import
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#So to sum it up, what we did with random forest regression was we formed a team of machine learning
#models which were DecisionTreeRegression models. You can also make a team of different machine learning models.