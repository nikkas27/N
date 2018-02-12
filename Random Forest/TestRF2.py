# Load Libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
import numpy as np

 # Read Data
 # binary = pd.read_csv('D:/University Selection System/Datasets/TrialDataset_2.csv')
binary = pd.read_excel('D:/University Selection System/Datasets/TrialDataset_1.xlsx')
test_data = pd.read_excel('D:/University Selection System/Datasets/TestData_1.xlsx')
 # print the column names
print ("TrainData Columns:\n",binary.columns)
print ("TestData Columns:\n",test_data.columns)
 # Explore Data
binary.describe()
test_data.describe()

 ### Data Manipulations
 # Columns
binary.dtypes.index

 # Drop a column
 #binary.drop('Year Applying', axis=1, inplace=True)
binary.drop('Letter of Recommendation', axis=1, inplace=True)
binary.drop('SOP', axis=1, inplace=True)
binary.drop('Experience', axis=1, inplace=True)
binary.drop('Employer', axis=1, inplace=True)
binary.drop('Finance', axis=1, inplace=True)
binary.drop('Attendence', axis=1, inplace=True)
binary.drop('ServedUSMilitary', axis=1, inplace=True)
binary.drop('Extra Ciricular Activities', axis=1, inplace=True)
binary.drop('Unnamed: 13', axis=1, inplace=True)
binary.drop('Unnamed: 14', axis=1, inplace=True)
binary.drop('Unnamed: 15', axis=1, inplace=True)
binary.drop('Unnamed: 16', axis=1, inplace=True)
binary.drop('Unnamed: 17', axis=1, inplace=True)
 # Print a few rows
print ("TrainData:\n",binary.head())
print ("TestData:\n",test_data)


 ### Split Target and Feature Set
 # Keep Target and Independent Variable into different array
Train_IndepentVars = binary.values[:, 1:4]
Train_TargetVars = binary.values[:, 4]
Test_TargetVar = test_data.values[:, 1:4]

print("Train_IndepentVars:\n",Train_IndepentVars)
print("Train_TargetVar:\n",Train_TargetVars)
print("Test_TargetVar:\n",Test_TargetVar)
 ### Random Forest Model
rf_model = RandomForestClassifier(max_depth=5, n_estimators=200)
rf_model.fit(Train_IndepentVars, Train_TargetVars)


 # Scoring based on the train RF Model
predictions = rf_model.predict(Test_TargetVar)
print ("Prediction:\n",predictions)

  # Confusion Matrix
#print(" Confusion matrix\n", confusion_matrix(Train_TargetVars, predictions))


importance = rf_model.feature_importances_
importance = pd.DataFrame(importance, index=binary.columns[1:4],
                          columns=["Importance"])

print(importance)

# style.use('ggplot')
# import numpy as np
# xpos = binary.values[:, 1]
# ypos = binary.values[:, 2]
# zpos = binary.values[:, 3]
# dx = np.ones(20)
# dy = np.ones(20)
# dz = np.ones(20)
#
# xpos1 = test_data.values[:, 1]
# ypos1 = test_data.values[:, 2]
# zpos1 = test_data.values[:, 3]
# #dz = (20 * np.random.rand(20) + 1).astype(int)
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(xpos1,ypos1,zpos1, c=('#00ceaa'if(predictions == 2)else('r')),marker='X',label='TestData')
# ax.bar3d(xpos,ypos,zpos,dx,dy,dz,color='#00ceaa')
# ax.set_xlabel(' GRE ')
# ax.set_ylabel(' TOEFL ')
# ax.set_zlabel(' GPA ')
# plt.title('University Selection System\nRandom Forest Algo')
# plt.legend()
# plt.subplots_adjust(bottom=0.18,right=0.94,top=0.90,wspace=0.2,hspace=0)
# plt.show()


# x, y are a np.meshgrid de deux np.linspace(-30,30,120,endpoint=False)
xpos = binary.values[:, 1]
ypos = binary.values[:, 2]
zpos = binary.values[:, 3]

xpos1 = test_data.values[:, 1]
ypos1 = test_data.values[:, 2]
zpos1 = test_data.values[:, 3]
#x, y, z = axes3d.get_test_data()
print (type(xpos), type(ypos), type(zpos))
print (xpos.shape, ypos.shape, zpos.shape)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(xpos,ypos,zpos, c='b',marker='o',label='TrainData')
ax.scatter(xpos1,ypos1,zpos1, c=('#00ceaa'if(predictions == 2)else('r')),marker='X',label='TestData')
#ax.plot_wireframe(xpos,ypos,zpos,rstride=5,cstride=15)
#ax.plot_wireframe(xpos,ypos,zpos)
ax.set_xlabel(' GRE ')
ax.set_ylabel(' TOEFL ')
ax.set_zlabel(' GPA ')
plt.title('University Selection System\nRandom Forest Algo')
plt.legend()
plt.show()
