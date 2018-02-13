###################### GAUSSIAN NAIVE BAYES ALGO ######################


from sklearn.naive_bayes import GaussianNB
from textblob.classifiers import NaiveBayesClassifier
from sklearn import preprocessing, cross_validation, svm
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
from textblob import TextBlob

#assigning predictor and target variables
df = pd.read_csv("D:/University Selection System/Datasets/TrialDataset_2_CSV.csv")
test_data = pd.read_excel('D:/University Selection System/Datasets/TestData_1.xlsx')

#df.dropna(inplace=True)
x = (df.values[:,1:4])
y = (df.values[:,4])
y=y.astype('int')
print ("Train Data:",x)
print ("Train Output:",y)

Test_TargetVar = test_data.values[:, 1:4]
print ("Test Data Values:",Test_TargetVar)
#Create a Gaussian Classifier
model = GaussianNB()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.4)
# Train the model using the training sets
#model.fit(x, y)
#model.fit(x,y)
model.fit(X_train,y_train)

#Predict Output
predicted= model.predict(Test_TargetVar)
print (predicted)

xpos = df.values[:, 1]
ypos = df.values[:, 2]
zpos = df.values[:, 3]

xpos1 = test_data.values[:, 1]
ypos1 = test_data.values[:, 2]
zpos1 = test_data.values[:, 3]
#x, y, z = axes3d.get_test_data()
print (type(xpos), type(ypos), type(zpos))
print (xpos.shape, ypos.shape, zpos.shape)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xpos,ypos,zpos, c='b',marker='o',label='TrainData')
ax.scatter(xpos1,ypos1,zpos1, c=('#00ceaa'if(predicted == 2)else('r')),marker='X',label='TestData')
#ax.plot_wireframe(xpos,ypos,zpos,rstride=5,cstride=15)
#ax.plot_wireframe(xpos,ypos,zpos)
ax.set_xlabel(' GRE ')
ax.set_ylabel(' TOEFL ')
ax.set_zlabel(' GPA ')
plt.title('University Selection System\nGaussian Naiive Bayes Algo')
plt.legend()
plt.savefig('D:/University Selection System/Results of Algos/GaussianNaiiveBayesAlgo.png')
plt.show()

# cl = NaiveBayesClassifier()
#
# print(cl.accuracy(Test_TargetVar))

