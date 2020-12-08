#IBRAHIM OMAR
#ICS 352 ASSIGNMENT 1

import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

#load spambase.data into the same directory as your code
df = pd.read_csv("spambase.data",header=None,names=range(58))
model = GaussianNB() #Load data into a dataframe
X= df.loc[:,:56] #Separate X from data
y= df.loc[:,57] #Separate y values(class) from data

#Split data into train(70%) and test(30%)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
model.fit(X_train,y_train) #Train the model

#Predict output on test data
pred = model.predict(X_test) 
#Calculate and print accuracy
print("Accuracy: ",round(metrics.accuracy_score(y_test,pred),3))
