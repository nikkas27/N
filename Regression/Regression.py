###################### REGRESSION ALGO ######################



#import Quandl, math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


#assigning predictor and target variables
df = pd.read_csv("D:/University Selection System/Datasets/TrialDataset_2_CSV.csv")
test_data = pd.read_excel('D:/University Selection System/Datasets/TestData_1.xlsx')

#df.dropna(inplace=True)
X = np.array(df.values[:,1:4])
y = np.array(df.values[:,4])
print ("Train Data:",X)
print ("Train Output:",y)

Test_TargetVar = test_data.values[:, 1:4]
print ("Test Data Values:",Test_TargetVar)

#X = preprocessing.scale(X)
print (len(X),len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#clf = LinearRegression(n_jobs=10)
clf = svm.SVR()
clf.fit(X_train, y_train)

#confidence = clf.score(x,y)
Accuracy = clf.score(X_test, y_test)
print("Accuracy:",Accuracy)

predicted = clf.predict(Test_TargetVar)
print("Predicted:",predicted)

