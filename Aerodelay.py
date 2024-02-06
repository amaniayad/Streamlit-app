#!/usr/bin/env python
# coding: utf-8

# # Visualization:

# ### Reading datasets

# In[35]:


import pandas as pd
df = pd.read_csv("./flight_delays_train/flight_delays_train.csv") #df us he labeled data
ud = pd.read_csv("./flight_delays_test/flight_delays_test.csv") #ud is the unlabeled data


# In[2]:


df


# In[3]:


ud


# ## Statistics:

# In[4]:


df.shape,ud.shape


# In[5]:


print(df.info(),'\n')
print(ud.info())


# In[6]:


print(df.describe(),'\n')
print(ud.describe())


# In[7]:


df.hist()


# In[8]:


df.plot(subplots=True, figsize=(8, 8))


# In[9]:


import seaborn as sns
sns.displot(df, x='dep_delayed_15min')


# ### Verify Independency between features:

# In[10]:


import matplotlib.pyplot as plt
import copy
d=copy.deepcopy(df)
for column in d.columns:
    d[column] = d[column].astype('category').cat.codes
plt.figure(figsize=(16,8))
corr=d.corr()
sns.heatmap(corr,annot=True)


# In[11]:


fig = df.plot(kind ="box")


# In[12]:


fig = ud.plot(kind ="box")


# ### Gaussian Distribution:

# In[13]:


sns.distplot(df['DepTime'])


# In[14]:


sns.distplot(df['Distance'])


# ## Handling Outliers

# In[36]:


import numpy as np 
a=np.percentile(df['Distance'],75)
b=np.percentile(df['Distance'],25)
IQRA=a-b
UpperA=(a+1.5*IQRA)
LowerA=(b-1.5*IQRA)
print(UpperA,LowerA)

for i in range(len(df)):
    if (df.at[i,'Distance']>UpperA):
         df.at[i,'Distance']=UpperA  
    elif (df.at[i,'Distance']<LowerA):
        df.at[i,'Distance']=LowerA


# In[37]:


a=np.percentile(ud['Distance'],75)
b=np.percentile(ud['Distance'],25)
IQRA=a-b
UpperA=int((a+1.5*IQRA))
LowerA=int((b-1.5*IQRA))
print(UpperA,LowerA)

for i in range(len(ud)):
    if (ud.at[i,'Distance']>UpperA):
         ud.at[i,'Distance']=UpperA  
    elif (ud.at[i,'Distance']<LowerA):
        ud.at[i,'Distance']=LowerA


# In[17]:


import plotly.express as px
fig = px.box(df, y="Distance")
fig.show()
fig = px.box(ud, y="Distance")
fig.show()


# ## Encoding

# In[38]:


from sklearn.preprocessing import OrdinalEncoder
import joblib
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_columns = ['UniqueCarrier', 'Origin', 'Dest']

df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])
ud[categorical_columns] = ordinal_encoder.transform(ud[categorical_columns])
joblib.dump(ordinal_encoder,'ordinal_encoder.pkl')


# In[39]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['dep_delayed_15min']=label_encoder.fit_transform(df['dep_delayed_15min'])


# In[40]:


x=df.drop(columns=['dep_delayed_15min'])
y=df['dep_delayed_15min']


# In[41]:


cols=['Month','DayofMonth','DayOfWeek']
for i in range(len(cols)):
    df[cols[i]] = df[cols[i]].str.replace('c-', '').astype(float)
    x[cols[i]] = x[cols[i]].str.replace('c-', '').astype(float)
    ud[cols[i]] = ud[cols[i]].str.replace('c-', '').astype(float)


# In[22]:


x


# In[23]:


ud


# In[24]:


y


# ## Split the data

# In[42]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)


# ### Verify linear separability:

# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns

class_column = 'dep_delayed_15min'
feature_columns = x.columns
for feature_column in feature_columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df[feature_column], y=df[class_column],hue=df['dep_delayed_15min'], palette='viridis', marker='o')
    plt.xlabel(feature_column)
    plt.ylabel(class_column)
    plt.title(f'Scatter Plot of {feature_column} with Class Coloring')
    plt.show()


# ## Normalization

# In[44]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
for column in x.columns:
    x_train[column] = scaler.fit_transform(pd.DataFrame(x_train[column]))
    x_test[column] = scaler.transform(pd.DataFrame(x_test[column]))
    ud[column] = scaler.transform(pd.DataFrame(ud[column])) 
joblib.dump(scaler,'minmax_scaler.pkl')


# # Modeling:

# # Self training

# In[45]:


import copy
x_train_knn=copy.deepcopy(x_train)
x_train_nb=copy.deepcopy(x_train)
x_train_svm=copy.deepcopy(x_train)
x_train_lr=copy.deepcopy(x_train)
x_train_clf=copy.deepcopy(x_train)

ud_knn=copy.deepcopy(ud)
ud_nb=copy.deepcopy(ud)
ud_svm=copy.deepcopy(ud)
ud_lr=copy.deepcopy(ud)
ud_clf=copy.deepcopy(ud)

y_train_knn=copy.deepcopy(y_train)
y_train_nb=copy.deepcopy(y_train)
y_train_svm=copy.deepcopy(y_train)
y_train_lr=copy.deepcopy(y_train)
y_train_clf=copy.deepcopy(y_train)


# ### Knn:

# In[126]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier()

max_iterations = 20

for iteration in range(max_iterations):
    knn.fit(x_train_knn,y_train_knn)
    knn_pred_unlabeled = knn.predict(ud_knn)

    confidence_threshold = 0.5
    confident_knn=knn.predict_proba(ud_knn)[:, 1] >= confidence_threshold
    
    y_train_knn = pd.concat([y_train_knn, pd.Series(knn_pred_unlabeled[confident_knn])], ignore_index=True)
    x_train_knn=pd.concat([x_train_knn,ud_knn[confident_knn]])
    
    for i in range(len(confident_knn)):
        if (confident_knn[i]==True):
            ud_knn=ud_knn.drop(index=i)
            
    ud_knn = ud_knn.reset_index(drop=True)   
    print(ud_knn.shape)
knn.fit(x_train_knn,y_train_knn)    


# ### Naive Bayes:

# In[68]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()

for iteration in range(max_iterations):
    nb.fit(x_train_nb,y_train_nb)
    nb_pred_unlabeled = nb.predict(ud_nb)
    
    confidence_threshold = 0.5
    confident_nb=nb.predict_proba(ud_nb)[:, 1] >= confidence_threshold

    y_train_nb = pd.concat([y_train_nb, pd.Series(nb_pred_unlabeled[confident_nb])], ignore_index=True)
    x_train_nb=pd.concat([x_train_nb,ud_nb[confident_nb]])
    
    for i in range(len(confident_nb)):
        if (confident_nb[i]==True):
            ud_nb=ud_nb.drop(index=i)        
            
    ud_nb = ud_nb.reset_index(drop=True)
    print(ud_nb.shape)
nb.fit(x_train_nb,y_train_nb)    


# ### Logistic Regression:

# In[46]:


from sklearn.linear_model import LogisticRegression
logistic_reg=LogisticRegression()
max_iterations = 20
for iteration in range(max_iterations):
    logistic_reg.fit(x_train_lr,y_train_lr)
    lr_pred_unlabeled = logistic_reg.predict(ud_lr)

    confidence_threshold = 0.5
    confident_lr=logistic_reg.predict_proba(ud_lr)[:, 1] >= confidence_threshold
    
    y_train_lr = pd.concat([y_train_lr, pd.Series(lr_pred_unlabeled[confident_lr])], ignore_index=True)
    x_train_lr=pd.concat([x_train_lr,ud_lr[confident_lr]])
    
    for i in range(len(confident_lr)):
        if (confident_lr[i]==True):
            ud_lr=ud_lr.drop(index=i)
    ud_lr = ud_lr.reset_index(drop=True)        
    print(ud_lr.shape)
logistic_reg.fit(x_train_lr,y_train_lr)    


# ### SVM:

# In[66]:


from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd

svml = LinearSVC(C=1.0, max_iter=1000, dual=False)
svm = CalibratedClassifierCV(svml, method='sigmoid')

for iteration in range(max_iterations):
    svm.fit(x_train_svm, y_train_svm)
    svm_pred_unlabeled = svm.predict(ud_svm)
    
    confidence_threshold = 0.5
    confidence_svm = svm.predict_proba(ud_svm)[:, 1] >= confidence_threshold
    y_train_svm = pd.concat([y_train_svm, pd.Series(svm_pred_unlabeled[confidence_svm])], ignore_index=True)
    x_train_svm = pd.concat([x_train_svm, ud_svm[confidence_svm]])

    confident_indices = [i for i, is_confident in enumerate(confidence_svm) if is_confident]
    ud_svm = ud_svm.drop(index=confident_indices).reset_index(drop=True)
    print(ud_svm.shape)

svm.fit(x_train_svm, y_train_svm)


# ### Decision Tree:

# In[72]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42)

for iteration in range(max_iterations):
    clf.fit(x_train_clf,y_train_clf)
    clf_pred_unlabeled = clf.predict(ud_clf)
    
    confidence_threshold = 0.5
    confident_clf=clf.predict_proba(ud_clf)[:, 1] >= confidence_threshold
    y_train_clf = pd.concat([y_train_clf, pd.Series(clf_pred_unlabeled[confident_clf])], ignore_index=True)
    x_train_clf=pd.concat([x_train_clf,ud_clf[confident_clf]])

    for i in range(len(confident_clf)):
        if (confident_clf[i]==True):
            ud_clf=ud_clf.drop(index=i)       
    ud_clf= ud_clf.reset_index(drop=True)
    print(ud_clf.shape)
clf.fit(x_train_clf,y_train_clf)    


# # Evaluation:

# ### KNN:

# In[73]:


from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score
final_pred =knn.predict(x_test)
accuracy = accuracy_score(y_test, final_pred)
score=f1_score(y_test, final_pred,average='weighted')
precision = precision_score(y_test, final_pred, average='weighted')
recall = recall_score(y_test, final_pred, average='weighted')
print(f'Precision of Knn: {precision:.4f}')
print(f'Recall of Knn: {recall:.4f}')
print(f'Final Accuracy of knn: {accuracy}')
print(f'Final f1_score of knn: {score}')


# ### Naive bayes:

# In[74]:


final_pred =nb.predict(x_test)
accuracy = accuracy_score(y_test, final_pred)
score=f1_score(y_test,final_pred,average='weighted')
precision = precision_score(y_test, final_pred, average='weighted')
recall = recall_score(y_test, final_pred, average='weighted')
print(f'Precision of nb: {precision:.4f}')
print(f'Recall of nb : {recall:.4f}')
print(f'Final Accuracy of nb: {accuracy}')
print(f'Final f1_score of nb: {score}')


# ### Logistic Regression:

# In[108]:


final_pred =logistic_reg.predict(x_test)
accuracy = accuracy_score(y_test, final_pred)
score=f1_score(y_test, final_pred,average='weighted')
precision = precision_score(y_test, final_pred, average='weighted')
recall = recall_score(y_test, final_pred, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Final Accuracy of lr: {accuracy}')
print(f'Final f1_score of lr: {score}')


# ### SVM:

# In[67]:


final_pred =svm.predict(x_test)
accuracy = accuracy_score(y_test, final_pred)
score=f1_score(y_test,final_pred,average='weighted')
precision = precision_score(y_test, final_pred, average='weighted')
recall = recall_score(y_test, final_pred, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Final Accuracy of svm: {accuracy}')
print(f'Final f1_score of svm: {score}')


# ### Decision Tree:

# In[76]:


final_pred =clf.predict(x_test)
accuracy = accuracy_score(y_test, final_pred)
score=f1_score(y_test,final_pred,average='weighted')
precision = precision_score(y_test, final_pred, average='weighted')
recall = recall_score(y_test, final_pred, average='weighted')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'Final Accuracy of decision tree: {accuracy}')
print(f'Final f1_score of decision tree: {score}')


# # Label Proparagtion

# In[89]:


from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

pseudo_labels = np.full(len(ud), -1)

combined_X = np.concatenate((x_train, ud), axis=0)
combined_y = np.concatenate((y_train, pseudo_labels), axis=0)
label_propagation_model = LabelPropagation(kernel='knn', n_neighbors=5, gamma=30, max_iter=2000)
label_propagation_model.fit(combined_X, combined_y)

predicted_labels_labeled_data = label_propagation_model.predict(x_test)

predicted_labels_unlabeled_data = label_propagation_model.predict(ud)

predicted_labels_combined = np.concatenate((predicted_labels_labeled_data, predicted_labels_unlabeled_data), axis=0)

accuracy = accuracy_score(y_test, predicted_labels_labeled_data)
precision = precision_score(y_test, predicted_labels_labeled_data, average='weighted')
recall = recall_score(y_test, predicted_labels_labeled_data, average='weighted')
f1 = f1_score(y_test, predicted_labels_labeled_data, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score : {f1}")


# In[78]:


print(np.unique(predicted_labels_unlabeled_data))


# ### Exporting the best model:

# In[36]:


import joblib
joblib.dump(svm, 'svm_model.pkl')


# In[27]:


loaded_ordinal_encoder = joblib.load('ordinal_encoder.pkl')
loaded_scaler = joblib.load('minmax_scaler.pkl')
loaded_svm = joblib.load('svm_model.pkl')
lpm = joblib.load('label_propagation_model.pkl')


# In[117]:


new_data = pd.DataFrame({
    'Month': [2],
    'DayofMonth': [22],
    'DayOfWeek': [3],
    'DepTime': [1551],
    'UniqueCarrier': ['XE'],
    'Origin': ['IND'],
    'Dest': ['IAH'],
    'Distance': [845]
})
categorical_columns = ['UniqueCarrier', 'Origin', 'Dest']
new_data[categorical_columns] = loaded_ordinal_encoder.transform(new_data[categorical_columns])
print(new_data['DepTime'])
for col in new_data.columns:
    new_data[col] = loaded_scaler.transform(new_data[col].values.reshape(-1, 1))
prediction_input = new_data.values.reshape(1, -1)
prediction = loaded_svm.predict(prediction_input)
print("Predicted Class:", prediction)


# In[ ]:




