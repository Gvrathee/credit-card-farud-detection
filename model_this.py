import pandas as pd
data = pd.read_csv('creditcard.csv')
#pd.options.display.max_columns = None
print("number of rows",data.shape[0])
print("number of columns",data.shape[1])
data.info()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))
data = data.drop(['Time'],axis=1)
print(data.shape)
print(data.duplicated().any())
data = data.drop_duplicates()
print(data.shape)
print(data['Class'].value_counts())
import seaborn as sns
x = data.drop(['Class'],axis=1)
y = data['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)

#undersampling
normal = data[data['Class']==0]
fraud = data[data['Class']==1]
print(normal.shape)
print(fraud.shape)
normal_sample = normal.sample(n=473)
new_data = pd.concat([normal_sample,fraud])
print(new_data['Class'].value_counts())
x = new_data.drop('Class',axis=1)
y = new_data['Class']
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=42)


#logistic regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
print(log.fit(X_train,y_train))

y_pred1 = log.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred1))

from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))

#decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
print(dt.fit(X_train,y_train))
y_pred2 = dt.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
print(rf.fit(X_train,y_train))
y_pred3 = rf.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))
final_data = pd.DataFrame({'Models':['LR','DT','RF'],
"ACC":[accuracy_score(y_test,y_pred1)*100,
accuracy_score(y_test,y_pred2)*100,
accuracy_score(y_test,y_pred3)*100,
]})
print(final_data)

#oversampling
X = data.drop('Class',axis=1)
y = data['Class']
print(X.shape)
print(y.shape)
from imblearn.over_sampling import SMOTE
X_res,y_res = SMOTE().fit_resample(X,y)
print(y_res.shape)
print(y_res.value_counts())
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,random_state=42)

#logistic regression
log = LogisticRegression()
print(log.fit(X_train,y_train))
y_pred1 = log.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(precision_score(y_test,y_pred1))
print(recall_score(y_test,y_pred1))
print(f1_score(y_test,y_pred1))

#decision tree classifier
dt = DecisionTreeClassifier()
print(dt.fit(X_train,y_train))
y_pred2 = dt.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(precision_score(y_test,y_pred2))
print(recall_score(y_test,y_pred2))
print(f1_score(y_test,y_pred2))

#random forest classifier
rf = RandomForestClassifier()
print(rf.fit(X_train,y_train))
y_pred3 = rf.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))

final_data = pd.DataFrame({'Models':['LR','DT','RF'],
"ACC":[accuracy_score(y_test,y_pred1)*100,
accuracy_score(y_test,y_pred2)*100,
accuracy_score(y_test,y_pred3)*100,
]})
print(final_data)

#final RandomForestClassifier model
rf1 = RandomForestClassifier()
print(rf1.fit(X_train,y_train))
y_pred4 = rf1.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(precision_score(y_test,y_pred3))
print(recall_score(y_test,y_pred3))
print(f1_score(y_test,y_pred3))
import joblib
print(joblib.dump(rf1,'modelfreshvikram'))
