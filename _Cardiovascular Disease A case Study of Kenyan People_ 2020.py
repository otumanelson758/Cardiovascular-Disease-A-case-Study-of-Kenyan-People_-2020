#!/usr/bin/env python
# coding: utf-8

# # Loading Data Set Cardiovascular Disease dataset_Kenya 2020

# # 1.#Introduction
# We need to understand the data set in detail.We develop a brief understanding of the data set of which we will be working with. For example how many features are there in the data set, how many unique labels, How are they distributed or how are the labels distributed, different data types and quantities.

# In[1]:


#importing the libraries for data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score,KFold,StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,average_precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder,MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest,chi2

from keras.models import Sequential
from keras.layers import Activation,BatchNormalization
from keras.layers.core import Dense,Dropout
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau,EarlyStopping


# In[2]:


#loading the data set for preprocessing
mydata=pd.read_csv("//Users//nelsonotumaongaya//Documents//DataScience Projects//cardio.csv",header="infer")


# In[3]:


mydata# we explore the headers in the data set


# In[4]:


mydata.columns.values# we check for the number of columns in our dataset.


# In[5]:


#Dropping the columns that we wont use to create the model.
mydata = mydata.drop(['id'], axis=1)


# In[6]:


mydata.head()# we have already dropped the id which is not relevant for our data.


# In[7]:


mydata.tail()# We explore the bottom of the data set to understand it.


# In[8]:


mydata.sample(30)#We sample out any number of the data set.


# # 2. Lets Check for the Missing values and Duplicate Values in the data set and Clean it

# In[9]:


mydata.isnull()# we now check for the missing values- True means the data set has missing values


# In[10]:


mydata.isnull().sum()# We check for the sum of the missing values in our data set.


# Our data set is clean therefore we do not need to clean it again.

# In[11]:


mydata.duplicated().sum()# we check for the sum of the duplicate values in our data set.


# our duplicate values in the data set is 24 therefore we need to drop the duplicate values

# In[12]:


mydata = mydata.drop_duplicates()# we now remove the duplicate values in our data set.


# In[13]:


mydata


# # Note

# we need to convert the age column into years. How we need to divide it by the number of days in a year.
# 

# In[14]:


mydata['age'] = mydata['age']/365# converting the age into years


# In[15]:


mydata.head()# we view of data again to confirm the conversion of the number of age if correctly done.


# In[16]:


#Checking for the data information
mydata.info()


# In[17]:


#Checking if there are NaNs
mydata.isna().sum()# our data does not have any missing values therefore its clean.


# We need to  BMI features since this is an important feature for heart diseases.
# ody Mass Index is a simple calculation using a person's height and weight. The formula is BMI = kg/m2 where kg is a person's weight in kilograms and m2 is their height in metres squared. A BMI of 25.0 or more is overweight, while the healthy range is 18.5 to 24.9. 
# BMI applies to most adults 18-65 years.

# In[18]:


mydata['bmi'] = mydata["weight"]/(mydata["height"]/100)**2


# In[19]:


#number of columns
num_colunas = mydata.shape[1]


# # Checking the correlations

# In[20]:


corr = mydata.corr()#we now check for the correlations in our data set


# In[21]:


f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# As expected there are correlations among gender and height and a small correlation among smoke and alcohol

# It's known that blood pressures higher than 250 for high and 200 for low are outliers. These data will be removed.

# # 2. Understanding the Data set on Blood pressure categories

# Normal
# Blood pressure numbers of less than 120/80 mm Hg are considered within the normal range. If your results fall into this category, stick with heart-healthy habits like following a balanced diet and getting regular exercise.
# 
# Elevated
# Elevated blood pressure is when readings consistently range from 120-129 systolic and less than 80 mm Hg diastolic. People with elevated blood pressure are likely to develop high blood pressure unless steps are taken to control the condition.
# 
# Hypertension Stage 1
# Hypertension Stage 1 is when blood pressure consistently ranges from 130-139 systolic or 80-89 mm Hg diastolic. At this stage of high blood pressure, doctors are likely to prescribe lifestyle changes and may consider adding blood pressure medication based on your risk of atherosclerotic cardiovascular disease (ASCVD), such as heart attack or stroke.
# 
# Learn more about your risk with our Check. Change. Control. Calculator™.
# 
# Hypertension Stage 2
# Hypertension Stage 2 is when blood pressure consistently ranges at 140/90 mm Hg or higher. At this stage of high blood pressure, doctors are likely to prescribe a combination of blood pressure medications and lifestyle changes.
# 
# Hypertensive crisis
# This stage of high blood pressure requires medical attention. If your blood pressure readings suddenly exceed 180/120 mm Hg, wait five minutes and then test your blood pressure again. If your readings are still unusually high, contact your doctor immediately. You could be experiencing a hypertensive crisis.

# In[22]:


#t's known that blood pressures higher than 250 for high and 200 for low are outliers. These data will be removed.
mydata = mydata[(mydata["ap_hi"]<=250) & (mydata["ap_lo"]<=200)]
mydata = mydata[(mydata["ap_hi"] >= 0) & (mydata["ap_lo"] >= 0)]


# Checking the new correlations

# In[23]:


#we now check for the new correlations
corr = mydata.corr()
f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.1f',ax=ax)


# Now the correlations look better and now there is correlations between high and low pressures.

# Cardio is out output data. Checking the amount for each classes

# In[24]:


sns.countplot(x='cardio',data=mydata)


# # The data is balanced now we are going to check for the outliers in other columns.

# In[25]:


fig,ax = plt.subplots(2,3,figsize=(13,5))
sns.countplot(x='gender',data=mydata,ax=ax[0][0])
sns.countplot(x='cholesterol',data=mydata,ax=ax[0][1])
sns.countplot(x='smoke',data=mydata,ax=ax[0][2])
sns.countplot(x='gluc',data=mydata,ax=ax[1][0])
sns.countplot(x='alco',data=mydata,ax=ax[1][1])
sns.countplot(x='active',data=mydata,ax=ax[1][2])
plt.tight_layout()


# In[26]:


print("Maximum age = ",mydata['age'].max())
print("Minimum age = ",mydata['age'].min())


# In[27]:


print("Maximum height = ",mydata['height'].max())
print("Minimum height = ",mydata['height'].min())


# In[28]:


print("Maximum ap_high = ",mydata['ap_hi'].max())
print("Minimum ap_high = ",mydata['ap_hi'].min())


# In[29]:


print("Maximum ap_low = ",mydata['ap_lo'].max())
print("Minimum ap_low = ",mydata['ap_lo'].min())


# Based on the values, columns 'ap_hi','ap_lo','age','height','weight' must be normalized. This will be done with MinMaxScaler

# In[30]:


mydata_norm = mydata.copy()


# In[31]:


colunas_normalizar = ['ap_hi','ap_lo','age','height','weight']

tipo_scaler = 'MinMax'
if(tipo_scaler=='Standard'):
    scaler = StandardScaler((0,1))
elif(tipo_scaler=='Robust'):
    scaler = RobustScaler()
elif(tipo_scaler=='MinMax'):
    scaler = MinMaxScaler(feature_range=(0, 1))

for col in colunas_normalizar:
    mydata_norm[col] = scaler.fit_transform(mydata_norm[col].values.reshape(-1,1))


# In[32]:


# We now check for normalization in our data set.
mydata_norm.head()


# In[33]:


#We now Check for the existence of outliers using boxplots
fig,ax = plt.subplots(1,2,figsize=(13,5))
sns.boxplot(y=mydata_norm['ap_hi'],x=mydata_norm['cardio'],ax=ax[0])
sns.boxplot(y=mydata_norm['ap_lo'],x=mydata_norm['cardio'],ax=ax[1])
plt.tight_layout()


# In[34]:


fig,ax = plt.subplots(1,3,figsize=(13,5))
sns.boxplot(y=mydata_norm['age'],x=mydata_norm['cardio'],ax=ax[0])
sns.boxplot(y=mydata_norm['height'],x=mydata_norm['cardio'],ax=ax[1])
sns.boxplot(y=mydata_norm['weight'],x=mydata_norm['cardio'],ax=ax[2])
plt.tight_layout()


# In[35]:


#Function to remove outliers
def remover_outlier(mydata,coluna_input,coluna_output,tipo):
    mydata_tmp = mydata[mydata[coluna_output]==tipo]
    q25, q75 = np.percentile(mydata_tmp[coluna_input], 25), np.percentile(mydata_tmp[coluna_input], 75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    x_inferior, x_superior = q25 - cut_off, q75 + cut_off
    outliers = [x for x in mydata_tmp[coluna_input] if x < x_inferior or x > x_superior]
    mydata_novo = mydata.drop(mydata[(mydata[coluna_input] > x_superior) | (mydata[coluna_input] < x_inferior)].index)
    return mydata_novo


# In[36]:


mydata_norm = remover_outlier(mydata_norm,'ap_hi','cardio',1)


# In[37]:


fig,ax = plt.subplots(1,2,figsize=(13,5))
sns.boxplot(y=mydata_norm['ap_hi'],x=mydata_norm['cardio'],ax=ax[0])
sns.boxplot(y=mydata_norm['ap_lo'],x=mydata_norm['cardio'],ax=ax[1])
plt.tight_layout()


# In[38]:


mydata_norm = remover_outlier(mydata_norm,'ap_lo','cardio',0)


# In[39]:


fig,ax = plt.subplots(1,2,figsize=(13,5))
sns.boxplot(y=mydata_norm['ap_hi'],x=mydata_norm['cardio'],ax=ax[0])
sns.boxplot(y=mydata_norm['ap_lo'],x=mydata_norm['cardio'],ax=ax[1])
plt.tight_layout()


# In[40]:


mydata_norm = remover_outlier(mydata_norm,'age','cardio',0)
mydata_norm = remover_outlier(mydata_norm,'height','cardio',0)
mydata_norm = remover_outlier(mydata_norm,'weight','cardio',0)


# In[41]:


fig,ax = plt.subplots(1,3,figsize=(13,5))
sns.boxplot(y=mydata_norm['age'],x=mydata_norm['cardio'],ax=ax[0])
sns.boxplot(y=mydata_norm['height'],x=mydata_norm['cardio'],ax=ax[1])
sns.boxplot(y=mydata_norm['weight'],x=mydata_norm['cardio'],ax=ax[2])
plt.tight_layout()


# # 3.Outliers

# Now our data values seems more reasonable than before.

# In[42]:


sns.countplot(x='cardio',data=mydata_norm)# we recheck again for the outliers 
plt.tight_layout()


# The data is still balanced after removing outliers

# # 4.Defining the variables X and Y for our data set.

# In[43]:


X = mydata_norm.drop('cardio',axis=1).values
Y = mydata_norm['cardio'].values


# # 5.We now split the data set into 80:20

# When you’re working on a model and want to train it, you obviously have a dataset. But after training, we have to test the model on some test dataset. For this, you’ll a dataset which is different from the training set you used earlier. But it might not always be possible to have so much data during the development phase. In such cases, the obviously solution is to split the dataset you have into two sets, one for training and the other for testing; and you do this before you start training your mode

# In[44]:


#Defining the train and test samples.
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20,random_state=42)


# In[45]:


#List to compute metrics
accuracy = []
precision =[]
recall = []
f1 = []
roc = []


# The data will be modeled using Logistic regression, KNN, Random Forest, AdaBoost and Gradient Boosting Classifiers

# # 6.Logistic Regression

# In[46]:


#print("Logistic Regression")
#log_reg_params = {"penalty": ['l1', 'l2','elasticnet'], 'C': [1, 10], 
#                  'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}
#grid_log_reg = GridSearchCV(LogisticRegression(max_iter=10000), log_reg_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')
#grid_log_reg.fit(X_train, y_train)
#logreg = grid_log_reg.best_estimator_
#print(logreg)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.linear_model import LogisticRegression
import wandb
import time


# In[47]:


#Parameters have been choosing based on GridSearchCV
logreg = LogisticRegression(C=1,max_iter=10000,penalty='l1',solver='liblinear')
logreg.fit(X_train,y_train)


# In[48]:


log_reg_score = cross_val_score(logreg, X_train, y_train, cv=10,scoring='roc_auc_ovo')
log_reg_score_teste = cross_val_score(logreg, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score Regressao Logistica Treino: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')
print('Score Regressao Logistica Teste: ', round(log_reg_score_teste.mean() * 100, 2).astype(str) + '%')


# In[49]:


Y_pred_logreg = logreg.predict(X_test)


# In[50]:


cm_logreg = confusion_matrix(y_test,Y_pred_logreg)


# In[51]:


acc_score_logreg = accuracy_score(y_test,Y_pred_logreg)
f1_score_logreg = f1_score(y_test,Y_pred_logreg)
precisao_logreg = average_precision_score(y_test,Y_pred_logreg)
recall_logreg = recall_score(y_test,Y_pred_logreg)
roc_logreg = roc_auc_score(y_test,Y_pred_logreg,multi_class='ovo')
print('Acuracia Regressão Logistica ',round(acc_score_logreg*100,2).astype(str)+'%')
print('Precião média Regressão Logistica ',round(precisao_logreg*100,2).astype(str)+'%')
print('F1 Regressão Logistica ',round(f1_score_logreg*100,2).astype(str)+'%')
print('Recall Regressão Logistica ',round(recall_logreg*100,2).astype(str)+'%')
print('ROC Regressão Logistica ',round(roc_logreg*100,2).astype(str)+'%')


# In[52]:


accuracy.append(acc_score_logreg)
precision.append(precisao_logreg)
recall.append(recall_logreg)
f1.append(f1_score_logreg)
roc.append(roc_logreg)


# In[53]:


def train_eval_pipeline(model, train_data,test_data,name):
    #initialize weights and biases
    wandb.init(project="Cardiovascular Disease dataset_Kenya 2020",name=name)
    #segragate the datasets
    (X_train, y_train)=train_data
    (X_test,y_test)=test_data
    #Train the model and keep the log of all the necessary metrics
    start=time.time()
    model.fit(X_train,y_train)
    end=time.time()-start
    prediction=model.predict(X_test)
    wandb.log({"accuracy": accuracy_score(y_test,prediction)*100.0,"precision": precision_recall_fscore_support(y_test,prediction, average="macro")[0],"recall":precision_recall_fscore_support(y_test,prediction,average='macro')[1],"training_time":end})
    print("Accuracy score of the Logistic Regression classifier with default hyperparameter values {0:.2f}%".format(accuracy_score(y_test,prediction)*100.))
    print("\n")
    print("---Classificatin Report of the Logistic Regression classifier with default hyperparameter values ----")
    print("\n")
    print(classification_report(y_test,prediction,target_names=[" Cardiovascular Disease dataset_Kenya 2020","Normal Human being"]))
logreg=LogisticRegression()
train_eval_pipeline(logreg,(X_train,y_train),(X_test,y_test),"logistic_regression")


# # Improving the Model

# Can we improve this model? Agood way to start approaching this idea is tune the hyperparameters of the model. We want to look at which is the best parameter for our model. We define the grid of values for the hyperparameter we would like to tune. In this case we use random search for hyperparameters tuning.

# In[54]:


#import GridSearchCV if something goes outside the region we penalize it
from sklearn.model_selection import RandomizedSearchCV


# In[55]:


#We define the grid
penalty=["l1","l2"]
C=[0.8,0.9,1.0]
tol=[0.01,0.001,0.0001]#what we can tolerate-tolerant values
max_iter=[100,150,200,250]# maximum iteration


# In[56]:


#we create key value dist
param_grid=dict(penalty=penalty,C=C,tol=tol,max_iter=max_iter)


# Now with the grid, we work to find the best set of values of hyperparameters values.

# In[57]:


#Instanstiate RandomizedSearchCV with the required parameters.
param_grid=dict(penalty=penalty,C=C,tol=tol,max_iter=max_iter)
random_model=RandomizedSearchCV(estimator=logreg,param_distributions=param_grid, cv=5)


# In[58]:


#Instanstiate RandomizedSearchCV with the required parameters.
random_model=RandomizedSearchCV(estimator=logreg,param_distributions=param_grid, cv=5)
random_model_result=random_model.fit(X_train,y_train)


# In[59]:


#summary of the results
best_score, best_params=random_model_result.best_score_,random_model_result.best_params_


# In[60]:


#summary of the results
best_score, best_params=random_model_result.best_score_,random_model_result.best_params_
print("Best Score: %.2f using %s" %(best_score*100, best_params))

Random search did not help much in boosting up the accuracy score. Just to ensure that lets take the hyperparameter values and train another logistic regression model with the same values
# In[61]:


#log the hyperparameter values with which we are going to train our model.
config=wandb.config
config.tol=0.01
config.penalty="12"
config.C=1.0


# In[62]:


#Train the model
logreg=LogisticRegression(tol=config.tol,penalty=config.penalty,max_iter=150,C=config.C)
train_eval_pipeline(logreg,(X_train,y_train),(X_test,y_test),'Logistic-regression -random-search')


# # 7.Building The Model Using K-Nearest Neighbor Classifier. KNN

# We have already explored the same data in Logistic Regression Analysis therefore the data is clean

# In[63]:


#print("KNN")
#knears_params = {"n_neighbors": list(range(20,30,1)),'leaf_size' : list(range(5,11,1)), 'weights': ['uniform', 'distance']}
#grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
#grid_knears.fit(X_train, y_train)
#knn = grid_knears.best_estimator_
#print("Best Estimator")
#print(knn)


# In[64]:


#Parameters have been choosing based on GridSearchCV
knn = KNeighborsClassifier(weights='uniform',n_neighbors=27,leaf_size=6)
knn.fit(X_train,y_train)


# In[67]:


knears_score = cross_val_score(knn, X_train, y_train, cv=10,scoring='roc_auc_ovo')
knears_score_teste = cross_val_score(knn, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score KNN Treino: ', round(knears_score.mean() * 100, 2).astype(str) + '%')
print('Score KNN Teste: ', round(knears_score_teste.mean() * 100, 2).astype(str) + '%')


# In[68]:


Y_pred_knn = knn.predict(X_test)


# In[69]:


cm_knn = confusion_matrix(y_test,Y_pred_knn)


# In[70]:


acc_score_knn = accuracy_score(y_test,Y_pred_knn)
f1_score_knn = f1_score(y_test,Y_pred_knn)
precisao_knn = average_precision_score(y_test,Y_pred_knn)
recall_knn = recall_score(y_test,Y_pred_knn)
roc_knn = roc_auc_score(y_test,Y_pred_knn,multi_class='ovo')
print('Acuracy KNN ',round(acc_score_knn*100,2).astype(str)+'%')
print('Pred media KNN ',round(precisao_knn*100,2).astype(str)+'%')
print('F1 KNN ',round(f1_score_knn*100,2).astype(str)+'%')
print('Recall KNN ',round(recall_knn*100,2).astype(str)+'%')
print('ROC KNN ',round(roc_knn*100,2).astype(str)+'%')


# In[71]:


accuracy.append(acc_score_knn)
precision.append(precisao_knn)
recall.append(recall_knn)
f1.append(f1_score_knn)
roc.append(roc_knn)


# In[72]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_knn, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("KNN \n Matriz de Confusão", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# # Ada Boost Classifier

# In[73]:


#print("Ada Boost Classifier")
#ada_params = {'n_estimators' : list(range(100,200))}
#grid_ada = GridSearchCV(AdaBoostClassifier(), ada_params,n_jobs=8,cv=10,scoring='roc_auc_ovo')
#grid_ada.fit(X_train, y_train)
#ada = grid_ada.best_estimator_
#print("Best Estimator")
#print(ada)


# In[74]:


#Parameters have been choosing based on GridSearchCV
ada = AdaBoostClassifier(n_estimators=102)
ada.fit(X_train,y_train)


# In[75]:


ada_score = cross_val_score(ada, X_train, y_train, cv=10,scoring='roc_auc_ovo')
ada_score_teste = cross_val_score(ada, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score AdaBoost Treino: ', round(ada_score.mean() * 100, 2).astype(str) + '%')
print('Score AdaBoost Teste: ', round(ada_score_teste.mean() * 100, 2).astype(str) + '%')


# In[76]:


Y_pred_ada = ada.predict(X_test)


# In[77]:


cm_ada = confusion_matrix(y_test,Y_pred_ada)


# In[78]:


acc_score_ada = accuracy_score(y_test,Y_pred_ada)
f1_score_ada = f1_score(y_test,Y_pred_ada)
precisao_ada = average_precision_score(y_test,Y_pred_ada)
recall_ada = recall_score(y_test,Y_pred_ada)
roc_ada = roc_auc_score(y_test,Y_pred_ada,multi_class='ovo')
print('Acuracia ADA Boost ',round(acc_score_ada*100,2).astype(str)+'%')
print('Precião média Ada Boost ',round(precisao_ada*100,2).astype(str)+'%')
print('F1 Ada Boost ',round(f1_score_ada*100,2).astype(str)+'%')
print('Recall Ada Boost ',round(recall_ada*100,2).astype(str)+'%')
print('ROC Ada Boost ',round(roc_ada*100,2).astype(str)+'%')


# In[79]:


accuracy.append(acc_score_ada)
precision.append(precisao_ada)
recall.append(recall_ada)
f1.append(f1_score_ada)
roc.append(roc_ada)


# In[80]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_ada, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Ada Boost \n Matriz de Confusão", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# # 8. Random Forest Classifier-RFC

# In[81]:


#print("Random Forest Classifier")
#forest_params = {"max_depth": list(range(10,50,1)),"n_estimators" : [350,400,450]}
#forest = GridSearchCV(RandomForestClassifier(), forest_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')
#forest.fit(X_train, y_train)
#random_forest = forest.best_estimator_
#print("Best Estimator")
#print(random_forest)


# In[82]:


#Parameters have been choosing based on GridSearchCV
random_forest = RandomForestClassifier(max_depth=10,n_estimators=350)
random_forest.fit(X_train,y_train)


# In[ ]:


forest_score -= cross_val_score(random_forest, X_train, y_train, cv=10,scoring='roc_auc_ovo')
forest_score_teste = cross_val_score(random_forest, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score RFC Treino: ', round(forest_score.mean() * 100, 2).astype(str) + '%')
print('Score RFC Teste: ', round(forest_score_teste.mean() * 100, 2).astype(str) + '%')


# In[ ]:


Y_pred_rf = random_forest.predict(X_test)


# In[ ]:


cm_rf = confusion_matrix(y_test,Y_pred_rf)


# In[ ]:


acc_score_rf = accuracy_score(y_test,Y_pred_rf)
f1_score_rf = f1_score(y_test,Y_pred_rf)
precisao_rf = average_precision_score(y_test,Y_pred_rf)
recall_rf = recall_score(y_test,Y_pred_rf)
roc_rf = roc_auc_score(y_test,Y_pred_rf,multi_class='ovo')
print('Acuracia Random Forest ',round(acc_score_rf*100,2).astype(str)+'%')
print('Precião média Random Forest ',round(precisao_rf*100,2).astype(str)+'%')
print('F1 Random Forest ',round(f1_score_rf*100,2).astype(str)+'%')
print('Recall Random Forest ',round(recall_rf*100,2).astype(str)+'%')
print('ROC Random Forest ',round(roc_rf*100,2).astype(str)+'%')


# In[ ]:


accuracy.append(acc_score_rf)
precision.append(precisao_rf)
recall.append(recall_rf)
f1.append(f1_score_rf)
roc.append(roc_rf)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_rf, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Random Forest \n Matriz de Confusão", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# # 9. Gradient Boost Classifier- GBC

# In[ ]:


#print("Gradient Boost Classifier")
#grad_params = {'n_estimators' : [50,55,60,65,70,75,80,85,90],'max_depth' : list(range(3,11,1))}
#grad = GridSearchCV(GradientBoostingClassifier(), grad_params,n_jobs=-1,cv=10,scoring='roc_auc_ovo')
#grad.fit(X_train, y_train)
#grad_boost = grad.best_estimator_
#print("Best Estimator")
#print(grad_boost)


# In[ ]:


#Parameters have been choosing based on GridSearchCV
grad_boost = GradientBoostingClassifier(n_estimators=65,max_depth=4)
grad_boost.fit(X_train, y_train)


# In[ ]:


grad_score = cross_val_score(grad_boost, X_train, y_train, cv=10,scoring='roc_auc_ovo')
grad_score_teste = cross_val_score(grad_boost, X_test, y_test, cv=10,scoring='roc_auc_ovo')
print('Score GradBoost Treino: ', round(grad_score.mean() * 100, 2).astype(str) + '%')
print('Score GradBoost Teste: ', round(grad_score_teste.mean() * 100, 2).astype(str) + '%')


# In[ ]:


Y_pred_gb = grad_boost.predict(X_test)


# In[ ]:


cm_gb = confusion_matrix(y_test,Y_pred_gb)


# In[ ]:


acc_score_gb = accuracy_score(y_test,Y_pred_gb)
f1_score_gb = f1_score(y_test,Y_pred_gb)
precisao_gb = average_precision_score(y_test,Y_pred_gb)
recall_gb = recall_score(y_test,Y_pred_gb)
roc_gb = roc_auc_score(y_test,Y_pred_gb,multi_class='ovo')
print('Acuracia Gradient Boosting ',round(acc_score_gb*100,2).astype(str)+'%')
print('Precião média Gradient Boosting  ',round(precisao_gb*100,2).astype(str)+'%')
print('F1 Gradient Boosting  ',round(f1_score_gb*100,2).astype(str)+'%')
print('Recall Gradient Boosting  ',round(recall_gb*100,2).astype(str)+'%')
print('ROC Gradient Boosting ',round(roc_gb*100,2).astype(str)+'%')


# In[ ]:


accuracy.append(acc_score_gb)
precision.append(precisao_gb)
recall.append(recall_gb)
f1.append(f1_score_gb)
roc.append(roc_gb)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_gb, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Gradient Boosting  \n Matriz de Confusão", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# In[ ]:


resultados = [log_reg_score,knears_score,ada_score,forest_score,grad_score]
resultados_teste = [log_reg_score_teste,knears_score_teste,ada_score_teste,forest_score_teste,grad_score_teste]
nome_modelo = ["Logistic Regression","KNN","AdaBoost","RFC","GradBoost"]


# In[ ]:


fig,ax=plt.subplots(figsize=(10,5))
ax.boxplot(resultados)
ax.set_xticklabels(nome_modelo)
plt.tight_layout()


# In[ ]:


fig,ax=plt.subplots(figsize=(10,5))
ax.boxplot(resultados_teste)
ax.set_xticklabels(nome_modelo)
plt.tight_layout()


# # 10. Deep Learning Model.

# In[ ]:


n_inputs = X_train.shape[1]


# In[ ]:


model = Sequential()
model.add(Dense(128, input_shape=(n_inputs, ), activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax', kernel_initializer='glorot_uniform',bias_initializer='zeros'))


# In[ ]:


reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, mode='auto', min_delta=0.0001)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
callbacks_list = [reduce_lr,es]
bsize = 2000


# In[ ]:


model.compile(Adam(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=bsize, epochs=200, verbose=2, validation_data=(X_test,y_test),callbacks=callbacks_list)


# In[ ]:


Y_pred_keras = model.predict_classes(X_test, batch_size=bsize, verbose=0)


# In[ ]:


cm_keras = confusion_matrix(y_test,Y_pred_keras)
acc_score_keras = accuracy_score(y_test,Y_pred_keras)
f1_score_keras = f1_score(y_test,Y_pred_keras)
precisao_keras = average_precision_score(y_test,Y_pred_keras)
recall_keras = recall_score(y_test,Y_pred_keras)
roc_keras = roc_auc_score(y_test,Y_pred_keras,multi_class='ovo')
print('Acuracia Keras ',round(acc_score_keras*100,2).astype(str)+'%')
print('Precião média Keras  ',round(precisao_keras*100,2).astype(str)+'%')
print('F1 Keras  ',round(f1_score_keras*100,2).astype(str)+'%')
print('Recall Keras  ',round(recall_keras*100,2).astype(str)+'%')
print('ROC Keras ',round(roc_keras*100,2).astype(str)+'%')


# In[ ]:


accuracy.append(acc_score_keras)
precision.append(precisao_keras)
recall.append(recall_keras)
f1.append(f1_score_keras)
roc.append(roc_keras)


# In[ ]:


fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(cm_keras, ax=ax, annot=True, cmap=plt.cm.copper)
ax.set_title("Keras  \n Matriz de Confusão", fontsize=14)
ax.set_xticklabels(['B', 'M'], fontsize=14, rotation=0)
ax.set_yticklabels(['B', 'M'], fontsize=14, rotation=360)


# In[ ]:


name_model = ["Logistic Regression","KNN","AdaBoost","RFC","GradBoost","Keras"]
dic_metrics = {'Model' : name_modelo, 'Accuracy' : accuracy, 'Precision' : precision, 'Recall' : recall, 'F1' : f1, 'ROC' : roc}
dataframe = pd.DataFrame(dic_metrics)


# In[ ]:


dataframe_sorted =  dataframe.sort_values(by=['ROC','Accuracy','Recall','F1','Precision'],ascending=False).reset_index().drop('index',axis=1)


# In[ ]:


dataframe_sorted


# From all models that have been testes Gradient Boosting Classifier had the best performance. RFC had a very similar behavior. Excepted KNN all other models had very similar accuracies and ROC score.

# In[ ]:


### Nelson Data Doyen.....######


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




