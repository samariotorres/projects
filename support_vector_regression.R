# SVR
#Support Vector Regression from SVM
# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3] #not using a training set or test set here..we are going to use the entire data set

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

# Fitting SVR to the dataset
install.packages('e1071')# this is where the the SVM function is located
library(e1071)
regressor = svm(formula = Salary ~ .,   #using the dot for more than one independent variable
                data = dataset, #setting data = our dataset
                type = 'eps-regression', #using eps-regression because we are building a non-linear regression model
                kernel = 'radial') #the gaussian kernel for our SVR model in R..see index
#If we were using an SVM model for classication, we would type = 'C-classification' see index
# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the SVR results
# install.packages('ggplot2')
#Creating our model visualizations
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')

# Visualising the SVR results (for higher resolution and smoother curve)
# install.packages('ggplot2')
library(ggplot2) #library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1) #x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), #So its like you're making 100 predictions and plotting..
             colour = 'red') +                            # so its going to be a smoother curve
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR)') +
  xlab('Level') +
  ylab('Salary')
