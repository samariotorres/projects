#Logistic Regression 

#You want to highlight and run the code bits at a time
#import the libraries we need
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#make sure in the correct directory
#import the dataset

dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values #Matrix of current independent variables
y = dataset.iloc[:, -1].values # dependent variable...also could have used y = dataset.iloc[:, 4].values

from sklearn.cross_validation import train_test_split #i call the line of code below, the wendys 4for4
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0) 
#Are we going to apply feature scaling? Yes because we want to predict which users are going to buy the 
# SUV so that we can target these users as well as possible! Data Science!

# Remember, what we are doing is making sure that nothing off scale...meaning; 
# we dont want independent variables outweighing
# other independent variables in a way that they are not supposed to.
# To do that:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #let the training set be equal to sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) # let the test set be equal to sc_X.fit_transform(X_test)

#this two lines above really looks like StandardScaler().sc_X.fit_transform(X_train)
#Notice we go from class --> object of that class --> instance of that class


#-----Now we are going to fit the logistic regression models-----
from sklearn.linear_model import LogisticRegression #importing our LogisticRegression class
classifier = LogisticRegression(random_state = 0) #creating our -->>CLASSIFIER OBJECT<<--- from the class
# LogisticRegression using sklearn.linear_model
classifier.fit(X_train, y_train)   # fitting the classifier to our TRAINING set...
#so we are taking our LogisticRegression classifier object and we are FITTING it to the TRAINING set, X_train, y _train

#Now we are going to predict the test set results
y_pred = classifier.predict(X_test) # so we are going to take our FITTED classifier object from the line of code above 

#We are going to make something new here called the CONFUSION MATRIX
#We are going to import a function from the sklearn.metrics library to compute the CONFUSION MATRIX faster!
from sklearn.metrics import confusion_matrix 
#notice the lower cases here because we are importing a function not a class. A class contains CAPITALS

cm = confusion_matrix(y_test, y_pred) #we used to the confusion matrix to evaluate the 
# predictive power of our logistic regression model

#first parameter is y_true, that is the REAL values that happended in reality
#second parameter is y_pred, that is the vector of predictions that our logistic regression model predicted
#other two parameters don't need for now


# Now we are going to have a graphic visualization in our results
# This graph is going to be pretty good.
%matplotlib qt
from matplotlib.colors import ListedColormap #helps colorize all of the data points
X_set, y_set = X_train, y_train #create some local values so we can create a shortcut so that we dont have to go around and 
#replace every single X_test and y_test. ;) 

#Line X1,X2 we take the minimum ages and estimatedsalaries - 1 and + 1 because we dont want the data points to be squeezed in
#the graph and make it look ugly...we want some border room pretty much
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), 
                    

                     np.arange(start = X_set[:,1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01)) 
                            #  #We took all of the pixel points available, and then we applied our classifier
                            # on it...so we just made a bunch of fake predictions
                            # and then applied our classifier on it...so its like each of the pixel points in     
                            # the graph is a user of the social network (like the solid red and green)
                            #Its like an observation, only its not one of the observations we had in our dataset


plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  #this line here is where we 
             #actually apply the classifier to all of the "pixel observation points" and by doing that, 
             #it colorizes everything.. use contour function to actually make the contour between the two prediction regions.
            
             alpha = 0.75, cmap = ListedColormap(('red','green'))) #------>if the pixel point belongs to class zero  
                                                                    # its gets red, else it gets green

plt.xlim(X1.min(), X1.max()) #here we are plotting the limits of                               

plt.ylim(X2.min(), X2.max() #x, the age, and y, the estimatedsalary                              

for i, j in enumerate(np.unique(y_set)): #then with this for loop, we are plotting all of the data points that are the real 
                                        #values. (unique, separated)

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], #using a scatter plot for the REAL value

                c = ListedColormap(('red','green'))(i), label = j) #green they bought the product,  red they did not

#Just labeling the axes, providing a legend, and showing the graph plt.show()
plt.title('Logistic Regression ( Training Set )')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualizing the Test set results below
%matplotlib qt
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = X_set[:,1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red','green'))(i), label = j)

#Just labeling the axes, providing a legend, and showing the graph plt.show()
plt.title('Logistic Regression ( Test Set )')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
""" 
We have some red points and green points.

All of the points you see on the graph are the observations from the TRAINING set.
That is, these are all of the users of the social network that were selected to
go to the training set. And each of these users is characterized by its Age (X-axis)
and its EstimatedSalary (y-axis).

--------------> The RED points are the TRAINING set observations for 
which the DEPENDENT variable, "Purchased" is EQUAL TO ZERO.

The GREEN points are the TRAINING set observations for which the DEPENDENT variable,
"Purchased" is EQUAL TO ONE. <---------------

We represented the way our classifier catches these users by plotting what are
called, THE PREDICTION REGIONS. 

RED region is the one that catches all of the user who do not buy the SUV.
Green region is the region where the classifier catches all of the user that do
buy the SUV.

This is the classifier predictioned COMPARED to the TRUTH.

THE POINT is THE TRUTH, THE REGION is the PREDICTION, hence PREDICTION REGION

The straight line that separates the prediction regions is called the

--------------------------> PREDICTION BOUNDARY <-----------------

AND AND AND the fact that its a straight line is not a coincidence...

its because THATS THE ESSENSE OF LOGISTIC REGRESSION.

If the prediction boundary is a straight line its because our logisticRegression 
classisfer is a LINEAR classifier. 

That means that since we are in 2-Dimensions, because we have two indepdent variables
Then SINCE the logistic regression classifier is a linear classifer,
then the prediction boundary separator can only be a straight line.


If we were in 3-dimensions, it would be straight plane.
Separting two SPACES.

Its always going to be a straight line if your classifier is linear classifier.

However, not so fast, as you can see the green SPOTS in the red REGION and the 
red SPOTS in the green REGION, are the SPOTS where our classifier 
INCORRECTLY PREDICTS.

These incorrect predictions are due specifically to the fact that our classifier is 
a linear classifier and also because our users are not linearly distriubted.

Later on, when you build non-linear classifiers, the prediction boundary 
separator will no longer be a straight line.





