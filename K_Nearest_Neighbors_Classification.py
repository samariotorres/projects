# K-Nearest Neighbors (K-NN) Classification
# =============================================================================
# #----------------------------K-NN Intuition---------------------
# K-NN Step by Step Guide (Very simple algorithm)
# 
# Step 1.) Choose the number 'K' of neighbors (common default value = 5)
# 
# Step 2.) Take the K-nearest neighbors of the new data point, according to their
#         Euclidean distance...or Manhattan distance...in most cases its Euclidean
#         You could use other distances like Minkowski distances and take a p-norm
#         In this example we use the l2 norm (basic 2-D Euclidean norm)
# 
# Step 3.) Once we have done this, then among these K-nearest neighbors, count 
#         the number of data points in each category.
# 
# Step 4.) Then we assign the new data point to the category where we counted 
#         the most neighbors. Simple, "K-NEAREST NEIGHBORS"
# 
# 
# =============================================================================



#importing the dataset and libraries making sure you have the correct directory
# hit the little folder top right if you're in spyder to do this

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv') 
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

#splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split #function because its lowercase
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#now we need to do feature scaling because its the same dataset that we are using and we do not want our
#independent variables outweighing others irrelevently, i.e, our 

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#Now we need to fit the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier #from sklearn.NEIGHBORS
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2) #setting the neighbors equal to
# 5. then our metric is the Minkowki metric with p = 2. The minkowski metric with p = 2
# is just the regular euclidean distance...like the one learned in middle school..its not hard just google it
classifier.fit(X_train, y_train) #Now we fit the classifier to the training data X_train and y_train

#Now we need to predict the test set results using the classifier object
y_pred = classifier.predict(X_test)

#Lets use the confusion matrix to see how the prediction did
from sklearn.metrics import confusion_matrix #from the sklearn.metrics class import the confusion_matrix function
cm = confusion_matrix(y_test, y_pred)
print(cm) #because we want to compare the values for the confusion_matrix based
# on the dependent variable we are testing and the predicted dependent variable


# Visualising the Training set results
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
             #it colorizes everything.. use contour function to actually make the contour between the two prediction regions..
            
             alpha = 0.75, cmap = ListedColormap(('red','green'))) #------>if the pixel point belongs to class zero its gets red, 
                                                                    # else it gets green

plt.xlim(X1.min(), X1.max()) #here we are plotting the limits of                               

plt.ylim(X2.min(), X2.max()) #x, the age, and y, the estimatedsalary                              

for i, j in enumerate(np.unique(y_set)): #then with this for loop, we are plotting all of the data points that are the real 
                                        #values. (unique, separated)

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], #using a scatter plot for the REAL value

                c = ListedColormap(('red','green'))(i), label = j) #green they bought the product,  red they did not

#Just labeling the axes, providing a legend, and showing the graph plt.show()
plt.title('KNN ( Training Set )')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# =============================================================================
# As you can see, the K-NN mode is non-linear classifier and is very useful
# =============================================================================
