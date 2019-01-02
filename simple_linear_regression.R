# Simple Linear Regression

# Importing the dataset
dataset = read.csv('Salary_Data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Salary, SplitRatio = 2/3) #dataset contains 30 observations so that test set gets 10 and 
#training set gets 20, splitRatio favors the training set
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Simple Linear Regression to the Training set
#We are going to call a new variable, regressor, and thats going to be the simple linear regressor itself
#lm stands for linear model,

regressor = lm(formula = Salary ~ YearsExperience,#which means salary is proportional to years of experience
               data = training_set) #second argument is the data we want to create the regressor machine on 

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set) #so we already trained our model on the training set,
#now we want to see what is going to happen with the prediction on the test_set
#y_pred is going to be the vector that contains the predicited values of the test set observations
#and we use the predict function, which takes two arguemtns, the simple linear regressor, and the "newdata", which is the
#data that we want to predict on which is going to be the test set


# Visualising the Training set results
#install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Salary vs Experience (Training set)') + #creating the title for graph
  xlab('Years of experience') + #creating the label for the x-axis
  ylab('Salary') #creating the label for the y-axis

# Visualising the Test set results
library(ggplot2) #importing ggplot2 using library(ggplot2)
ggplot() + #begin the ggplot function using + to continue editing
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),#we need to change the "training_set" spots in this one
  #test set because we actually want to see the test set points)
             #aes stands for aesthetics to create pretty shit
             colour = 'red') + #x-values will be YearsofExperience from the test set, y-values will be salary from test set
  #put obersvation points in red, observation points meaning the actual data goes in red
  #geom_POINT plots POINTS
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + #geom_LINE plots LINES # we did not need to change the training_sets to test_sets because its
  #going to use the same regressor machine as before..once we find the lm model regressor line its the one thats 
  #going to be used for everything...OUR REGRESSOR IS ALREADY TRAINED ON THE TRAINING SET!!!!!!
  ggtitle('Salary vs Experience (Test set)') + 
  xlab('Years of experience') +
  ylab('Salary')


#When you run summary(regressor), the coefficients with *** means that the linear relationship
#between both variables is strong
#a very small p-value implies a high-level of statisticsl
