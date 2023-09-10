import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

df=pd.read_csv("/content/drive/MyDrive/data.csv")

print('Shape of Diabetes dataset is :',df.shape)
print('Size of Diabetes dataset is  :',df.size)

df

df.columns

df.info()

df = df.drop_duplicates()

df.shape

import matplotlib
matplotlib.rcParams['figure.figsize'] = (30, 17)
corrmat=df.corr()
sns.heatmap(corrmat, annot=True)

df.drop(["Unnamed: 32"],axis=1,inplace=True)

df.isnull().sum()

df.info()

df['diagnosis'].unique()

df['diagnosis']=df['diagnosis'].map({'M':'1','B':'0'})

#Label Encoding
from sklearn.preprocessing import LabelEncoder
Encode = LabelEncoder()
df['diagnosis'] = Encode.fit_transform(df['diagnosis'])

df['diagnosis'].unique()

df['diagnosis']=df['diagnosis'].astype(int)

df.info()

df.drop(["id"],axis=1,inplace=True)

df

import matplotlib

matplotlib.rcParams['figure.figsize'] = (300, 200)
sns.barplot(x='radius_mean',y='perimeter_mean',data=df)

matplotlib.rcParams['figure.figsize'] = (300, 200)
sns.barplot(x='area_worst',y='area_mean',data=df)

#matplotlib.rcParams['figure.figsize'] = (300, 200)
sns.barplot(x='compactness_se',y='smoothness_se',data=df)

df1 = df[['radius_mean','area_mean','concave points_mean']]
x = pd.DataFrame(df1['radius_mean'].unique())
heatmap_pt = pd.pivot_table(df1,values ='concave points_mean', index=['area_mean'], columns='radius_mean')
sns.heatmap(heatmap_pt,cmap='twilight')

df

df.columns

matplotlib.rcParams['figure.figsize'] = (12, 6)
x = df.drop(['diagnosis'],axis = 1) # drop dependent feature and plot the outliers.
for i in x.columns:
    sns.boxplot(x = i, data = x,color = 'blue')
    plt.xlabel(i)
    plt.show()

from sklearn.preprocessing import QuantileTransformer
x=df
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
df_new=pd.DataFrame(X)
df_new.columns =['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

df_new

df_new.info()

matplotlib.rcParams['figure.figsize'] = (12, 6)
corrmat=df_new.corr()
sns.heatmap(corrmat, annot=True)

x = df_new.drop(['diagnosis'],axis = 1)
for i in x.columns:
    sns.boxplot(x = i, data = x,color = 'blue')
    plt.xlabel(i)
    plt.show()

X = df_new.drop(columns='diagnosis', axis=1)
Y = df_new['diagnosis']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

X_train.shape,Y_train.shape

X_test.shape, Y_test.shape

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
logic = LogisticRegression()
logic.fit(X_train, Y_train)
Y_pred_lr = logic.predict(X_test)

log_train = round(logic.score(X_train, Y_train) * 100, 2)
log_accuracy = round(accuracy_score(Y_pred_lr, Y_test) * 100, 2)

print("Training Accuracy    :",log_train ,"%")
print("Model Accuracy Score :",log_accuracy ,"%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n",classification_report(Y_test,Y_pred_lr))
print("\033[1m--------------------------------------------------------\033[0m")

df_new.iloc[5,:]

# input_data = (2,95,90,40,150,24,0.727000,20)
input_data=(17.99,10.38,.8122,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)
# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data) # converting this list into numpy array

# reshape the numpy array as we are predicting for one instance

input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)
print(input_data_reshaped)

predictions = logic.predict(input_data_reshaped)
print(predictions)
if predictions[0] == 1:
    print("Malignant")
else:
    print("Benign")

a=confusion_matrix(Y_test,Y_pred_lr)
a

matplotlib.rcParams['figure.figsize']=(12,6)
sns.heatmap(a,annot=True,cmap='YlOrRd')

from sklearn.model_selection import GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],                  # Regularization parameter
    'penalty': ['l1', 'l2'],            # Regularization type
    'solver': ['liblinear', 'saga']     # Optimization algorithm
}

# Create the grid search object
grid_search = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid, cv=5)

# Perform grid search on the training data
grid_search.fit(X_train, Y_train)

# Get the best hyperparameters found by grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

logic_tuned = LogisticRegression(**best_params)
logic_tuned.fit(X_train, Y_train)
Y_pred_lr_tuned = logic_tuned.predict(X_test)
log_accuracy_tuned = round(accuracy_score(Y_pred_lr_tuned, Y_test) * 100, 2)
print("Tuned Model Accuracy:", log_accuracy_tuned)

predictions = logic_tuned.predict(input_data_reshaped)
print(predictions)
if predictions[0] == 1:
    print("Malignant")
else:
    print("Benign")

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)

svc_train = round(svc.score(X_train, Y_train) * 100, 2)
svc_accuracy = round(accuracy_score(Y_pred_svc, Y_test) * 100, 2)

print("Training Accuracy    :",svc_train ,"%")
print("Model Accuracy Score :",svc_accuracy ,"%")
print("\033[1m--------------------------------------------------------\033[0m")
print("Classification_Report: \n",classification_report(Y_test,Y_pred_svc))
print("\033[1m--------------------------------------------------------\033[0m")

predictions = svc.predict(input_data_reshaped)
print(predictions)
if predictions[0] == 1:
    print("Malignant")
else:
    print("Benign")

b=confusion_matrix(Y_test,Y_pred_svc)
b

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)
Y_pred_knn= knn_model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred_knn)
conf_matrix = confusion_matrix(Y_test, Y_pred_knn)
classification_rep = classification_report(Y_test, Y_pred_knn)
print("Accuracy:", accuracy*100)
print("Classification Report:\n", classification_rep)

print("Confusion Matrix:\n", conf_matrix)

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)

accuracy_rf=accuracy_score(Y_test, Y_pred_rf)
conf_matrix = confusion_matrix(Y_test, Y_pred_rf)
classification_rep = classification_report(Y_test, Y_pred_rf)
print("Accuracy:", accuracy_rf*100)
print("Classification Report:\n", classification_rep)

print("Confusion Matrix:\n", conf_matrix)

import xgboost as xgb

boost= xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

boost.fit(X_train, Y_train)
Y_pred_xgb=boost.predict(X_test)
print("Accuracy Score:",accuracy_score(Y_pred_xgb,Y_test)*100)
print("Classification Report:",classification_report(Y_pred_xgb,Y_test))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

accuracy_xgb = accuracy_score(Y_test, Y_pred_xgb)*100
accuracy_lr = accuracy_score(Y_test, Y_pred_lr)*100
accuracy_rf = accuracy_score(Y_test, Y_pred_rf)*100
accuracy_knn = accuracy_score(Y_test, Y_pred_knn)*100
accuracy_svm = accuracy_score(Y_test,Y_pred_svc)*100
ocean_colors = ['#00A5DB', '#0097B5', '#007FAE', '#00689D', '#005588']
models = ['XGBoost', 'Logistic Regression', 'RandomForest', 'KNN', 'SVM']
accuracy_scores = [accuracy_xgb, accuracy_lr, accuracy_rf, accuracy_knn, accuracy_svm]

for i, score in enumerate(accuracy_scores):
    plt.text(i, score + 2, f'{score:.2f}%', ha='center', fontsize=12, fontweight='bold')

plt.bar(models, accuracy_scores, color=ocean_colors)
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Accuracy Score Comparison')
plt.ylim(0, 120)

import pickle
model_name="breast_logic.pkl"
with open(model_name,'wb') as file:
  pickle.dump(logic,file)