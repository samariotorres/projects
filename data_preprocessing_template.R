# Data Preprocessing

#Importing the dataset...set your working directory, navigate to the directory you want, then click more,
#then set as working directory

dataset = read.csv('Data.csv')
#-------->       dataset = dataset[, 2:3]      <------------- 
#this is how you create a subset of a dataset in this case all of the rows and only columns 2 and 3
#Taking care of missing data

#First doing it with the "Age" column: so we say "dataset$Age" to start
#is.na(dataset$Age) means if the value in the dataset for the specified column is missing or not
#it will return true if a value is missing and false if there are no values missing
#na.rm means that we are asking "R" to include the missing values when "R" goes through the "Age" column 
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                      dataset$Age)
#so in this ifelse (3 parameters), there's 3 things, first thing is the condition that says if something is missing in that
#specific column.
#, second parameter is what we want to do if there is something in that column, and we're going to replace what's missing 
#with the mean of the of the values in that column, third parameter is what we want when there is nothing missing and in 
# that case we just want the column itself so we just say datset@Age.

#Then we do the exact same thing with the Salary column
dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

#Machine learning models are based on mathematical equations, so 
#intuitively it makes sense that we need to encode our categorical 
#random variables into numbers

#Encoding categorical data

#We're going to use the "factor" function which will transform our categorical values into numerical values
#In other words, we're going to transform the Country column into a column of factors

#factor function has three parameters, the column of row you're messing with,
#the levels, which are the actual values you're trying to encode
#and the labels, which are what you want to redefine the levels as

dataset$Country = factor(dataset$Country,
                         levels = c('France','Spain','Germany'), 
                         labels = c(1, 2, 3))
#c is a vector in R, so we just created a vector of 3 elements
dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No','Yes'), 
                         labels = c(0, 1))

#Splitting the dataset into the training set and test set 

library(caTools)
set.seed(123) #To generate the same values as instructor

split = sample.split(dataset$Purchased, SplitRatio = 0.8)

#When you do SplitRatio , TRUE means that the observation goes to the training set (80% since 0.8) and
#false means that the observation goes to the test set
#Now what we're doing below is actually splitting the sets into their corresponding subsets
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Let's do some feature scaling
#When we used the factor function above, all we did was change what the "Country" looked like, we didn't actually change
#what the value of the country was, so if you try to run:
#training_set = scale(training_set)
#test_set = scale(test_set)
#you'll get an error that says 'x' must be numeric. Because even though the training test and test set looks numeric,
#it actually is not, the country column and the purchased column are actually not numeric.
#The FACTOR in R is not a numeric number
#so what we are going to do is EXCLUDE those pseudo-numeric columns from being scaled, and only scale the actual numeric
#columns as shown below: Remember indices start at 1 in R
training_set[,2:3] = scale(training_set[,2:3])
test_set[,2:3] = scale(test_set[,2:3]) #taking all rows and only columns 2 and 3...now the machine learning model
#will converge rapidly
