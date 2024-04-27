import pandas as pd
import numpy as np
import math
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot
from pandas.plotting import scatter_matrix
mydata = pd.read_csv("C:\cancer_n1.csv")
mydata.info()
print(mydata)
mydata.hist()
pyplot.show()

X=mydata.iloc[:,0:-1]
print(X)
y=mydata.iloc[:,-1]
print(y)

# 20% svm
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=400)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(C=0.3, random_state=350, kernel='linear')
svc.fit(X_train_std, y_train)
y_predict = svc.predict(X_test_std)
b=accuracy_score(y_test, y_predict)
print("SVM Accuracy score  is ",b)

# 20% rf
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=400)
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
a=accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy is:",a )

# 20% KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy:", accuracy)

#20% PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=400)
pca = PCA(n_components=5)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
log_reg = LogisticRegression(max_iter=1000) 
log_reg.fit(X_train_pca, y_train)
y_pred_pca = log_reg.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("PCA + Logistic Regression Model Accuracy:", accuracy_pca)

# 30% svm
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=350)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(C=0.3, random_state=350, kernel='linear')
svc.fit(X_train_std, y_train)
y_predict = svc.predict(X_test_std)
b=accuracy_score(y_test, y_predict)
print("SVM Accuracy score  is ",b)


#30% rf
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.3, random_state=350)
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
p=accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy is:",p )

# 30% KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=350)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy:", accuracy)

#30% PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=350)
pca = PCA(n_components=5)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
log_reg = LogisticRegression(max_iter=1000) 
log_reg.fit(X_train_pca, y_train)
y_pred_pca = log_reg.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("PCA + Logistic Regression Model Accuracy:", accuracy_pca)

# 40% svm
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.4, random_state=300)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(C=0.3, random_state=350, kernel='linear')
svc.fit(X_train_std, y_train)
y_predict = svc.predict(X_test_std)
b=accuracy_score(y_test, y_predict)
print("SVM Accuracy score  is ",b)


#40% rf
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.4, random_state=300)
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
p=accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy is:",p )

# 40% KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=300)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy:", accuracy)

# 40% PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=300)
pca = PCA(n_components=5)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
log_reg = LogisticRegression(max_iter=1000) 
log_reg.fit(X_train_pca, y_train)
y_pred_pca = log_reg.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("PCA + Logistic Regression Model Accuracy:", accuracy_pca)

# 50% svm
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.5, random_state=250)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
svc = SVC(C=0.3, random_state=350, kernel='linear')
svc.fit(X_train_std, y_train)
y_predict = svc.predict(X_test_std)
b=accuracy_score(y_test, y_predict)
print("SVM Accuracy score  is ",b)

#50% rf
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.5, random_state=250)
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
p=accuracy_score(y_test, y_pred)
print("Random Forest Model Accuracy is:",p )

# 50% KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=250)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)  
knn.fit(X_train_std, y_train)
y_pred = knn.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
print("KNN Model Accuracy:", accuracy)

#50% PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=250)
pca = PCA(n_components=5)  
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
log_reg = LogisticRegression(max_iter=1000) 
log_reg.fit(X_train_pca, y_train)
y_pred_pca = log_reg.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)
print("PCA + Logistic Regression Model Accuracy:", accuracy_pca)

X=mydata.iloc[:,0:-1]
Y=mydata.iloc[:,-1]
from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3, random_state=1540)
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100, random_state=42)
RF.fit(X_train,Y_train)
Y_pred = RF.predict(X_test)
X = metrics.accuracy_score(Y_test, Y_pred)
acc_list = []
acc_list.append(X)
model=[]
model.append('RF')
Q=RF.score(X_train,Y_train)
R=RF.score(X_test,Y_test)
from sklearn import metrics
from sklearn.metrics import classification_report
print('Accuracy:',metrics.accuracy_score(y_test, y_pred))

mydata= mydata.iloc[: , :-1]
mydata = mydata.sample()
prediction = RF.predict(mydata)
print(prediction)
print(mydata)

X=mydata.iloc[:,0:-1]
mydata= mydata.iloc[: , :-1]
mydata=np.array([[400135,98.8,42,51.5,4500,123,13.21,11.2]])
prediction=RF.predict(mydata)
print(prediction)