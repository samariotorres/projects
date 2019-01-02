# Polynomial Regression
#Problems with NonLinear Regression in the future
# Importing the dataset
dataset = read.csv('Position_Salaries.csv') #importing the csv file, dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3] #redefining the dataset as the with the indexes of the 1st and 2nd column....


#----------------------------------> INDEXES IN R START AT 1!!!!!!! <--------------------------

# Splitting the dataset into the Training set and Test set
#We do not need to split the data set because the data is so small only 10 observations 
# # install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Salary, SplitRatio = 2/3)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Linear Regression to the dataset
#Making the linear model to compare to the polynomial regression model 
lin_reg = lm(formula = Salary ~ ., #Creating our regressor lin_reg using the lm formula, lm(formula = Salary ~ .,)
             data = dataset) #second argument is our data 

# Fitting Polynomial Regression to the dataset
#here below, we are adding the polynomial features, which are some additional independent variables
# which are the level squared, the level cubed, and the level to the 4th power
dataset$Level2 = dataset$Level^2 #this will build a model with the original indep. variable plus the squared version 
dataset$Level3 = dataset$Level^3 # etc
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., #creating the poly_reg regressor
              data = dataset) #still training our polynomial regression model on the same dataset

# Visualising the Linear Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)), #graphing our lin_reg predictor on the dataset
            colour = 'blue') +
  ggtitle('Truth or Bluff (Linear Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Polynomial Regression results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), #x-coordinates are the independent variable values
             colour = 'red') + # y-coordinates are the dependendent variable(s), add a second argument for the color
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the Regression Model results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(poly_reg,
                                        newdata = data.frame(Level = x_grid,
                                                             Level2 = x_grid^2,
                                                             Level3 = x_grid^3,
                                                             Level4 = x_grid^4))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (Polynomial Regression)') +
  xlab('Level') +
  ylab('Salary')

# Predicting a new result with Linear Regression
predict(lin_reg, data.frame(Level = 6.5)) #just made a single prediction with lin_reg

# Predicting a new result with Polynomial Regression
predict(poly_reg, data.frame(Level = 6.5,
                             Level2 = 6.5^2,
                             Level3 = 6.5^3,
                             Level4 = 6.5^4))
