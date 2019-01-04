# Random Forest Regression

# Importing the dataset, dont forget to select the correct working directory
dataset = read.csv('Position_Salaries.csv') #importing the dataset dataset = read.csv('Position_Salaries')
dataset = dataset[2:3]

# Splitting the dataset into the Training set and Test set
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Random Forest Regression to the dataset
# install.packages('randomForest') needed to install this package
library(randomForest) #importing the library randomForest library (you can also check it in the packages section)
set.seed(1234) #set.seed(1234) so that we generate the same values for example purposes
regressor = randomForest(x = dataset[-2], # this will give us a dataframe here..a subdataframe of our
                                          #original data frame..all but the second
                         y = dataset$Salary, #second parameter is, 'y', our response VECTOR.
                                            # dataset$Salary is a column vector, in this case
                                            # its also our response vector.
                         ntree = 500) # our number of trees in the forest

# Don't forget, the more trees we add, the more the avaerage of the different predictions is converging
# to the same average..."entropy and informational gain" 

# Predicting a new result with Random Forest Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Random Forest Regression results (higher resolution)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01) #we increase 0.1 to 0.01 to increase our
                                                          #resolution
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Random Forest Regression)') +
  xlab('Level') +
  ylab('Salary')

# Ensemble Learning: Random forest is a version of ensemble learning, there are other version such as gradient 
# boosting. Ensemble learning is when you take multiple algorithms or the same algoirthm mutliple times and you put
# them together to make something much more powerful than the original.

#Note: The model that was winning out of all of the regressions was the polynomial regression,
# but now, the trees are winning. They're getting closer to the actual value! 

#When you have a team of several machine learning models, they can actually make awesome predictions!
#There is no Einstein Machine learning model....yet