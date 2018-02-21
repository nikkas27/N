################################### K NEAREST NEIGHBORS ALGO #########################################

######################################## Load Libraries #########################################

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

##################################### Read Data ################################################

#df = pd.read_csv('D:/University Selection System/Datasets/Dataset_1690.csv')
df = pd.read_csv("D:/University Selection System/Datasets/Fall 2014 Results.csv",)
test_data = pd.read_csv('D:/University Selection System/Datasets/TestFor1690.csv')
df=df.astype('int')
# df.drop(['IELTS'], 1, inplace=True)
# test_data.drop(['IELTS'], 1, inplace=True)
df.dropna(inplace=True)
print (df.isnull().any())

test_data.dropna(inplace=True)
print (test_data.isnull().any())
#df = df.fillna(lambda x: x.median())
print (df.columns)

df[df.isnull().any(axis=1)]
cleanup_nums = {"Result":{"Accept": 1, "Reject": 0}}
df.replace(cleanup_nums, inplace=True)
test_data.replace(cleanup_nums, inplace=True)
df.head()
################################### Data Manipulations #########################################

 # Drop a column
 # Doing df.drop returns a new dataframe with our chosen column(s) dropped.
 # The labels, y, are just the class column.
df.drop(['ID'],1, inplace=True)
df.drop(['Timestamp'], 1, inplace=True)
df.drop(['Name'], 1, inplace=True)
df.drop(['Date of Application'], 1, inplace=True)
df.drop(['Date of Decision'], 1, inplace=True)
df.drop(['Major'], 1, inplace=True)
df.drop(['Work-Ex'], 1, inplace=True)
df.drop(['International Papers'], 1, inplace=True)
df.drop(['Scale'], 1, inplace=True)
df.drop(['Undergrad Univesity'], 1, inplace=True)
# df.drop(['IELTS'], 1, inplace=True)
########################## Define our features (X) and labels (y) ##############################

X = df.values[:,:4]
y = np.array(df.values[:,4])
y=y.astype('int')


Test_TargetVar = test_data.values[:, 1:5]
Estimated = test_data.values[:, 5]
print ("Train_IndependentVars:\n",X)
print ("Train_TargetVars:\n",y)
print ("Test Values:\n",Test_TargetVar)


 #Checking whether the lengths of both our X and Y s are same..
 #print (len(X),len(y))

############################## Training and Testing samples #####################################
######################### Using Scikit-Learn's Cross_Validation #################################

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

X_train[np.isnan(X_train)] = np.median(X_train[~np.isnan(X_train)])
X_test[np.isnan(X_test)] = np.median(X_test[~np.isnan(X_test)])
#openingPriceTest[np.isnan(openingPriceTest)] = np.median(openingPriceTest[~np.isnan(openingPriceTest)])
 #Printing the lengths of both our X_Train and y_Train s are ..
print (len(X_train) ,len(y_train))

################################# Define the classifier: ########################################

clf = neighbors.KNeighborsClassifier()

################################# Train the classifier: #########################################

clf.fit(X_train, y_train)

####################################### Test: ###################################################

accuracy = clf.score(X_test, y_test)
print("Accuracy:",accuracy*100,"%")

Test_TargetVar = Test_TargetVar.reshape(len(Test_TargetVar),-1)

predictions= clf.predict(Test_TargetVar)
print("Prediction:",predictions)

print("Estimated Output:",Estimated)

Accuracy_Score_rf=accuracy_score(Estimated, predictions)
print("Accuracy Score of Random Forest:",(Accuracy_Score_rf)*100,"%")

############################# This code is purely for graphing only #############################

slices = [(accuracy*100),((1-accuracy)*100)]
activities = ['Accurate','Not Accurate']
cols = ['c','lightgrey']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow= True,
        explode=(0.1,0),
        autopct='%1.1f%%' )

plt.title('University Selection System\n\nK Nearest Neighbors Accuracy')
 #For saving the plot
plt.savefig('D:/University Selection System/Results of Algos/KNearestNeighborsAccuracy.png')
plt.show()
