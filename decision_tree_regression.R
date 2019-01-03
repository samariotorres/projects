# Decision Tree Regression, see decision_tree_regression.py for intuition behind decision_tree_REGRESSION (not classification)
# Dataset is nonlinear, which why we are making a nonlinear regression model
# No need to split dataset into training and test set in this example because of test size
# DecisionTreeRegression model is not very interesting in 1-D, but CAN be a very interest and powerful in more dimensions.
# Importing the dataset
dataset = read.csv('Position_Salaries.csv') #dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3] #dataset = dataset[2:3] setting the dataset to be equal to columns 2 and 3

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# We do not need to apply feature scaling to decision trees because DecisionTreeRegression models are based on
# conditions on the independent variables that have nothing to do with Euclidean distances. When we need to apply
# feature scaling its because the machine learning models are based on Euclidean distances and we need to put all of the
# independent variables on the same scale so that one of the independent variables are not dominating another one.
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Decision Tree Regression to the dataset
# install.packages('rpart')
library(rpart) # using a function from rpart to use a DecisionTreeRegression regressor
regressor = rpart(formula = Salary ~ ., #regressor object = rpart = ( formula = dependentVariable ~ .(dot represents that
                  data = dataset, #setting our data being used = dataset     #we are using all independent variables
                  control = rpart.control(minsplit = 1)) #Remember the way the decisionTree model is made is by making 
# some splits based on different conditions. The more conditions you have on your independent variables, the more 
# you have splits, and if you do not include "control = rpart.control(minsplit = 1)" then the regressor will
#directly take the average because of the no split....so we do some basic "MODEL PERFORMANCE IMPROVEMENT" using line 29

#DecisionTreeRegression model

# Predicting a new result with Decision Tree Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Decision Tree Regression results (higher resolution)
# install.packages('ggplot2')
library(ggplot2)
#notice the line of code below is changing the scale of the graph 
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01) #for same reason as in python we need to use this 0.01
ggplot() +                                                # to plot more predictions to have a nicer visualization
  geom_point(aes(x = dataset$Level, y = dataset$Salary), # remember the shape of a decisionTreeRegression model
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression)') +
  xlab('Level') +
  ylab('Salary')

# Plotting the tree
plot(regressor) #Here you can actually plot the regressor
text(regressor) # and the text that goes along with it to make a nice picture :)