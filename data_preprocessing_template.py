import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #used for importing datasets

#Importing the dataset...first thing to do is set your working directory
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values #We are removing the last column of our dataset because that means
#we are taking all of the feature variables(all of the independent variables) so it should be ok

y = dataset.iloc[:, 3].values #sometime we will have to change the "3" because thats the index for the
#"y", which is the dependent variable, because we might have more than 3 independent variables,
#or less than that

#Taking care of missing data using sci-kit-learn
#Imputer class allows us to take care of missing data
from sklearn.preprocessing import Imputer

#create an object of the Imputer class
#NaN because thats the default for missing values
#mean because we want to replace the missing values with the mean
# axis is 0 because we're taking the average of the values that are in the columns
# axis is 1 if we want to take the average along the rows
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

#using "fit" because we want to "fit" the imputer on the matrix X
#but we dont want to "fit" the entire matrix, we just want to fit the columns
#that are missing values, so we take all the rows with :, and specify the columns
#we only want columns 1 through 2, but we use 1:3 because the Python syntax to do it
#fitted the imputer object to matrix X
imputer = imputer.fit(X[:, 1:3])

#now we need to actually replace the missing data of the matrix X by the mean of
#the column
#X[:, 1:3] selecting the columns with missing data
#the "transform" method replaces the missing data with the mean of the column
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Machine learning models are based on mathematical equations, so 
#intuitively it makes sense that we need to encode our categorical 
#random variables into numbers

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Label encoder is a class for sklearn preprocessing library
#first thing to do is to create an object from the LabelEncoder class
labelencoder_X = LabelEncoder()
labelencoder_X.fit_transform(X[:, 0]) #this is going to return the first column,
#"Country" of the matrix X, ENCODED"


#Now we want the ACTUAL COLUMN to be equal to that encoded vector we just created so we set
#it equal to that line of code i.e, line 45
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

#So right now, we're good with having values for everything
#But we need to be careful with how the machine learning model actually treats the 0,1,and 2's
#We dont want the machine learning model to think that Spain is actually greater than
#Germany or whatever so we need to fix that using OneHotEncoder
#What's going to happen is we're going to change the values 0,1,2,etc, to a table of binary values
#where its going to fire a 1 if its that actual country in the corresponding spot, and it will be
#zero when its not the other two countries..ex) 1 0 0 would mean we are talking about Spain if Spain was
#the first column

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Doing the same thing with the Yes or No column
#We don't need to use OneHot because since this column is the dependent variable,
#the machine learning model will know that its a category and therefore there is no order
#between the two possible values

labelencoder_y = LabelEncoder()
labelencoder_y.fit_transform(y) #so this encodes the vector y


#Now were going to split the training set and the test set 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#A lot of maachine learning models are based on the Euclidean distance...
#The salary is going from 0 to 100k, so the Euclidean distance is going to
#be dominated by the salary...so when you do the square root of the sum of the squares,
#you're not really going to be changing the values between the Age and the Salary
#You dont want dominating values
#There are several ways of scaling the data
#Common approach is standardization: x_stand = ((x - mean(x)) / std.dev.(x) )
#and also Normalization; x_norm = ((x - min(x)/ (max(x)- min(x)) ))
#where 'x' is the observation feature value

#So we are going to feature scale

#When you're applying your object standardScaler object to your training set 
#you have to FIT the object of the training set and THEN transform it
from sklearn.preprocessing import StandardScaler
#scaling the X matrix of features
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

#However, we dont need to fit the test set, we just need to transform it
X_test = sc_X.transform(X_test)

#Question: Do we need to fit and transform the dummy variables?
#Answer: It depends on the context of the problem, if we scale the dummy variables,
#then everything will be on the same scale and we'll be happy and it will
#be good for our predictions, but we will lose the interpretation of knowing
#which observation belong to which country, in this case we would be scaling the 1's amd 0's from
#the X_train

#Even if the machine learning models are not based on Euclidean distances, we will still need 
#to do feature scaling because the algorithm will converge much faster.
#an example is decision trees.

#Question: Do we need to apply feature scaling to y? the dependent variable vector y_train and y_test?
#Answer: No, we do not because this is a classification problem with a categorical dependent variable
#But we will see that for regression, the dependent variable will take a huge range of values
#we will need to apply feature scaling to the dependent variable 'y' as well

