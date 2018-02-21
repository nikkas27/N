###################### K NEAREST NEIGHBORS ALGO ######################

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import matplotlib.pyplot as plt

 # Read Data
df = pd.read_csv("D:/University Selection System/Datasets/TrialDataset_2_CSV.csv")
test_data = pd.read_excel('D:/University Selection System/Datasets/TestData_1.xlsx')
#df.replace('nan',-99999, inplace=True)
#df.replace('NaN',-99999, inplace=True)

 ### Data Manipulations
 # Columns
 # Drop a column
  # Doing df.drop returns a new dataframe with our chosen column(s) dropped.
  # The labels, y, are just the class column.
df.drop(['Applicant ID '], 1, inplace=True)

 #we define our features (X) and labels (y):
X = np.array(df.values[:,:3])
y = np.array(df.values[:,3])
y=y.astype('int')

Test_TargetVar = np.array(test_data.values[:, 1:4])

print ("Train_IndependentVars:\n",X)
print ("Train_TargetVars:\n",y)

 #Checking whether the lengths of both our X and Y s are same..
#print (len(X),len(y))

 #creating training and testing samples, using Scikit-Learn's cross_validation.train_test_split:
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

 #Printing the lengths of both our X_Train and y_Train s are ..
print (len(X_train) ,len(y_train))

 #Define the classifier:
clf = neighbors.KNeighborsClassifier()

 #Train the classifier:
clf.fit(X_train, y_train)

 #Test:
accuracy = clf.score(X_test, y_test)
print("Accuracy:",accuracy)

Test_TargetVar = Test_TargetVar.reshape(len(Test_TargetVar),-1)

predictions= clf.predict(Test_TargetVar)

print("Prediction:",predictions)


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
plt.savefig('D:/University Selection System/Results of Algos/KNearestNeighborsAccuracy.png')
plt.show()
