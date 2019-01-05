#Logistic Regression in R

#need to import the dataset and set as current directory 

dataset = read.csv('Social_Network_Ads_for_Classification.csv')
dataset = dataset[, 3:5] #Choosing the columns that we want in our dataset

#Splitting the dataset into the Training Set and Test set
library(caTools) #library used to split into training and test
set.seed(123) #setting the seed so that we get the same numbers for example purposes
split = sample.split(dataset$Purchased, SplitRatio = 0.75) # 75% going to the training set
training_set = subset(dataset, split == TRUE) 
test_set = subset(dataset, split == FALSE)

#now we need to take care feature scaling..for the same reason why we did it
#it in Python, so that in this case, the Estimated Salary and Age are not directly compared
#numerically..it doesn't make sense to actually compare the number 40 and the number 60,000...
# and its easy in R to do this we use the scale function
training_set[,1:2] = scale(training_set[,1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

#End of datapreprocessing; now we are going to build the Logistic Regression Model
classifier = glm(formula = Purchased ~ ., #using glm because our LogReg Classifier is linear
                 family = binomial, # for Logistic regression you have to specify the binomial family
                 data = training_set) 
#Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])     
#Line 24: We are predicting the test set observations using our classifier which is the logistic
# regression classifier from Line 20. Type = 'response' to get back a vector. newdata=test_set[-3]
# because we want to prediict on our test set...prob_pred is going to return the predicted 
# probabilities that the user will buy the product.
#But we dont want the probabilities, we want the 0 and 1 results.

#so we'll do the following
y_pred = ifelse(prob_pred > 0.5, 1, 0) #changing the probabilities to ifelse.

#Now we are creating the confusion matrix
cm = table( test_set[, 3], y_pred) #first argument is the vector of real values, 
#second is the vector of predictions

#Now we are going to visualize the training set results.
#--------> for information on the graph, see logistic_regression.py <-------------
install.packages('ElemStatLearn')
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 1]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(classifier, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Logistic Regression (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))


#Why are we making classifiers?
#The goal here is to classify the right users into the right catgories
#trying to make a classifier that will catch the right users into the right category

set = test_set #shortcut so we dont have to change training_set to test_set everywhere
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01) #doing this so that our points are not 
X2 = seq(min(set[, 2]) - 1, max(set[, 1]) + 1, by = 0.01) #squeezed in the graph. we're building the 
grid_set = expand.grid(X1, X2) #grid here. We're doing this for the age&salary columns
#since grid_set is actually a matrix of the two columns, age and salary,
colnames(grid_set) = c('Age', 'EstimatedSalary') #so here we give a name to the columns of the matrix
#remember c is a function that combines values into a vector or list
prob_set = predict(classifier, type = 'response', newdata = grid_set)

#Now we use the classifier to predict the result of each of the pixel observation points
#the imaginary pixel users
y_grid = ifelse(prob_set > 0.5, 1, 0) #since the predict function returns probabilities, we transform
plot(set[, -3], #it into 0 or 1 results. which returns a vector of predictions of all the points in the
     # the grid
     main = 'Logistic Regression (Test set)', #labeling
     xlab = 'Age', ylab = 'Estimated Salary', #labeling
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato')) #choosing colors
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#from confusion matrix: 10 + 7 = 17 incorrect predictions
#and 57 + 26 = 83 correct predictions
#type in cm into the console to view the confusion matrix
