import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy.stats import skew
#%matplotlib inline

 
#importing Dataset
data=pd.read_csv("heart.csv")

    
# # Analysing the Data of Heart dataset

 
data.head()

 
#Checking Datatype 
data.info()

 
data.shape

 
#statistical data
data.describe()

 
sns.pairplot(data,x_vars=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca'],y_vars='target',height=2,aspect=0.5)

 
# In the above figures we can see the squiggle relationships.Hence we have to use the Sigmoid function to predict the heart disease

 
sns.heatmap(data.corr(),annot=True)

 
#The above figure represents the Correlations. It is evident that chest Pain has high correlation to the heart disease(target)

# #  Splitting the attributes and the target

 
X= data.drop("target",axis=1)
Y=data["target"]
X.head()

 
Y.head()

    
# # Splitting Training and Test data

 
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.3,stratify=Y,random_state=1)

    
# # Model training

 
model=LogisticRegression()

 
model.fit(X_train,Y_train)

    
# #  Model evaluation

 
#accuracy on training data
XtrainPrediction=model.predict(X_train)
traininDataAccuracy=accuracy_score(XtrainPrediction,Y_train)
print("Accuracy = ",traininDataAccuracy)

 
#accuracy on test data
XtestPrediction=model.predict(X_test)
testDataAccuracy=accuracy_score(XtestPrediction,Y_test)
print("Accuracy = ",testDataAccuracy)

 
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, XtestPrediction)
print(confusion_matrix)

 
from sklearn.metrics import (classification_report)
print(classification_report(Y_test, XtestPrediction))

 
# As the difference between the Train accuracy and Test accuracy is small, our Model is not biased and Overfitted.

    
# # Building a user-friendly interface to predict whether the patient's heart is fine or has some disease

 
user_input=(59,1,0,110,239,0,0,142,1,1.2,1,1,3

)
#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=model.predict(userInputReshaped)

if(prediction[0]==1):
  print("This person has a heart disease")
else:
  print("This Person Doesn't have a heart disease")

 



