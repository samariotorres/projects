# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('50_Startups.csv') #Importing a csv file dataset = read.csv('50_StartUps.csv')

# Encoding categorical data
#we want to encode our categorical variables in R, do so doing dataset$ColumnWantToEncode
#we use the factor function to encode here, first parameter is the column we want to encode
# second parameter is the vector of the levels or names of the variables in that column that we want to replace
#third parameter is the vector we want to replace the categorical variables with all done respectively
#We're doing this because of the CATEGORICAL VARIABLE
dataset$State = factor(dataset$State, 
                       levels = c('New York', 'California', 'Florida'), 
                       labels = c(1, 2, 3)) #1,2, and 3 are the numeric FACTORS that are going to replace the categorical

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools) 
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8) #we are splitting over the dependent variable
#0.8 means 80% of the dataset goes to the training set
training_set = subset(dataset, split == TRUE) #80% should be true
test_set = subset(dataset, split == FALSE) #20% should be false

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

# Fitting Multiple Linear Regression to the Training set
# first thing is to introduce the MULTIPLE LINEAR REGRESSOR,called REGRESSOR below, which we will fit to TRAINING set 
#and then we will apply the predict function on it
regressor = lm(formula = Profit ~ ., #we also use the lm function ---- in simple linear regression we only had one other 
               #independent variable, so we did y ~ x_n with just one thing...but now we have to consider the
               #other independent variables so we use a "."...
               #we could have also listed out all independent variables separated by a "+"
               data = training_set) #second arugement is the training set obviously we need to TRAIN the PREDICTOR on the 
                                      #TRAINING set
#when you run summary(regressor) in the console, you can see State 2 and State 3 which are the dummy variables
# that R already knew how to create....R and Python take care of dummy variables
#p-value tells us the level of significance of the independent variable on the dependent variable...
#normally a good threshold is 5%....0.05 :)
#remember the level of stars tells you how high the level of statistical significance is

#as you can see from summary(regressor)..we see that R&D is the most significant..so if you think about it we could change
# "." in the equation y ~ "." to Profit ~ R&D_Spend

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set) #here we are applying the predict function, which takes two arguments in 
# case, one if the regressor object, you have to specificy with which linear regressor you want to specify the results
#and of course it is with our multiple linear regressor that we have expressed on line 32
# and the second arguement is the newdata that we want to use to predict a profit...which is going to be the 
# test set obviously because we want to PREDICT on the TEST set

"""Did we make the optimal model just now? By using all of the independent variables? Could we have used some
of the more significant variables and left out insignificant variables?""" 

#This will be a little more simple in R than it was Python...we're going to use the "summary" function for sure

#We are going to use the same model to implement backwards elimination
#Except we are going to actually list the independent variables so that when we copy paste we just need to remove
#the non-statistically significant independent random variables
#We put a "." in between the names because thats how R handles "spaces"
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset) # We are taking the whole dataset so that we can have complete information on the 
#independent variables which are statistically significant 
summary(regressor)




#----------------------------------------------------------------------

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)
summary(regressor)            # these two lines of code are the ones that continue to get copy, pasted, deleted,
                              #until we have our optimal team


regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend,
               data = dataset)
summary(regressor)  


regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend,
               data = dataset)
summary(regressor)  


regressor = lm(formula = Profit ~ R.D.Spend,
               data = dataset)
summary(regressor)  #we removed the marketing spend even though it was very close because we are strictly following the 
                    # backward elimination , this was the final OPTIMAL TEAM OF 1


#Automatic Implementation of Backward Elimination Algorithm 

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)
