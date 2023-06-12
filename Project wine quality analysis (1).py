#!/usr/bin/env python
# coding: utf-8

# # Importing the Library

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import random

random.seed(100)


# # Importing the Dataset

# In[6]:


wine = pd.read_csv('winequality.csv')


# In[7]:


wine.head()


# # Making binary classificaion for the response variable.

# In[8]:


from sklearn.preprocessing import LabelEncoder
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()


# # Plotting the response variable

# In[9]:


sns.countplot(wine['quality'])


# In[10]:


wine.columns


# In[15]:


sns.pairplot(wine)


# In[17]:


wine[wine.columns[:11]].describe()


# # Histogram

# In[18]:


fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(wine.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(wine.columns.values[i])

    vals = np.size(wine.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(wine.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[16]:


wine.isna().any()


# # Correlation with Quality with respect to attributes

# In[19]:


wine.corrwith(wine.quality).plot.bar(
        figsize = (20, 10), title = "Correlation with quality", fontsize = 15,
        rot = 45, grid = True)


# # Correlation Matrix

# In[21]:


sns.set(style="white")

#Compute the correlation matrix
corr = wine.corr()


# In[22]:


corr.head()


# In[23]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(18, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# # Feature Enggineering

# In[24]:


#Assigning and dividing the dataset
X = wine.drop('quality',axis=1)
y=wine['quality']


# In[25]:


X.head()


# In[26]:


y.head()


# In[27]:


wine.columns[:11]


# In[28]:


features_label = wine.columns[:11]


# # Random Forest
# The model we choose is random forest.
# 
# Set number of estimators as 100, which is the best amongst 10, 50 and 100 Set splitting criteria as Gini Index Set min samples required to split: 5% Set min samples required at leaf as 0.1% We use 10-fold corss validation to exam the model and gain an average accuracy. The figure shows ROC curve given by each iteration, the grey area stands for average ROC curve.
# 

# In[29]:


#Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X, y)
importances = classifier.feature_importances_
indices = np. argsort(importances)[::-1]
for i in range(X.shape[1]):
    print ("%2d) %-*s %f" % (i + 1, 30, features_label[i],importances[indices[i]]))


# In[30]:


plt.title('Feature Importances')
plt.bar(range(X.shape[1]),importances[indices], color="green", align="center")
plt.xticks(range(X.shape[1]),features_label, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()


# In[31]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 5)


# # Feature Scaling

# In[32]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train2 = pd.DataFrame(sc.fit_transform(X_train))
X_test2 = pd.DataFrame(sc.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


# In[33]:


#Using Principal Dimensional Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(pd.DataFrame(explained_variance))


# # Models
# We firstly define a function which returns accuracy, true positive rate, false positive rate, precision, and F-score in binary classification. In this case, we use 10-fold cross validation, so there is no need to split our dataset.

# # Logistic Regression
# The model we choose is logistic regression.
# 
# Use default L2 penalties, which outperforms L1 with higher accuracy We can not see very obvious improvement on accuracy by increasing times of iteration, therefore, we set iteration times as its default value 100. After training this model using training dataset We use We use 10-fold corss validation to exam the model and gain an average accuracy. Following table shows the accuracy, TPR, FPR, precision, F-score on validation dataset. The figure shows ROC curve given by each iteration, the grey area stands for average ROC curve.
# 
# 

# In[34]:


#### Model Building 

### Comparing Models

## Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results = pd.DataFrame([['Logistic Regression', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results)


# # Support Vector Machine
# The second model we choose is support vector machine.
# 
# Set kernel as rbf, wich outperformes 'linear, 'poly' and 'sigmoid' kernels Set C as 1, which gives the highest accuracy amongst 0.1, 0.5, 1 and 10 Set tolerance for stopping criteria as 1e-3, which gives the highest accuracy amongst 1e-1, 1e-2, 1e-3, 1e-4 and 1e-5 We use 10-fold corss validation to exam the model and gain an average accuracy. The figure shows ROC curve given by each iteration, the grey area stands for average ROC curve.

# In[35]:


from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'linear')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)


# In[36]:


## SVM (rbf)
from sklearn.svm import SVC
classifier = SVC(random_state = 0, kernel = 'rbf')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)


# In[37]:


## Randomforest
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'entropy')
classifier.fit(X_train, y_train)

# Predicting Test Set
y_pred = classifier.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)


# In[2]:


from tkinter import *
import numpy as np

def showQuality():
    new = np.array([[float(e1.get()),float(e2.get()),float(e3.get()),float(e4.get()),float(e5.get()),float(e6.get()),float(e7.get()),float(e8.get()),float(e9.get()),float(e10.get()),float(e11.get())]])
    Ans = RF_clf.predict(new)
    fin=str(Ans)[1:-1]#IT WILL remove[ ]
    quality.insert(0, fin)
#------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# For this kernel, I amm only using the red wine dataset
data = pd.read_csv('winequality.csv')
data.head()

#Summary statistics
data.describe()

#All columns has the same number of data points
extra = data[data.duplicated()]
extra.shape

...
# Let's proceed to separate 'quality' as the target variable and the rest as features.
y = data.quality                  # set 'quality' as target
X = data.drop('quality', axis=1)  # rest are features
print(y.shape, X.shape)

#Let's look at the correlation among the variables using Correlation chart
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, 
            linecolor='black', annot=True)

#Use Random Forest Classifier to train a prediction model

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix

#Split data into training and test datasets
seed = 8 # set seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=seed)


#Train and evaluate the Random Forest Classifier with Cross Validation
# Instantiate the Random Forest Classifier
RF_clf = RandomForestClassifier(random_state=seed)


# Compute k-fold cross validation on training dataset and see mean accuracy score
cv_scores = cross_val_score(RF_clf,X_train, y_train, cv=10, scoring='accuracy')

#Perform predictions
RF_clf.fit(X_train, y_train) 
pred_RF = RF_clf.predict(X_test)


#------------------------------------------------------------------------
master =Tk()
master.title("Wine Quality Analysis")

Label(master, text="Fixed Acidity", anchor="nw",font=10, width=50).grid(row=0)
Label(master, text="Volatile Acidity", anchor="nw",font=10, width=50).grid(row=5)
Label(master, text="Citric Acid", anchor="nw",font=10, width=50).grid(row=10)
Label(master, text="Residual Sugar", anchor="nw",font=10, width=50).grid(row=15)
Label(master, text="Chlorides", anchor="nw",font=10, width=50).grid(row=20)
Label(master, text="Sulfur Dioxide", anchor="nw",font=10, width=50).grid(row=25)
Label(master, text="Total Sulfur Dioxide", anchor="nw",font=10, width=50).grid(row=30)
Label(master, text="Density", anchor="nw",font=10, width=50).grid(row=35)
Label(master, text="pH", anchor="nw",font=10, width=50).grid(row=40)
Label(master, text="Sulphates", anchor="nw",font=10, width=50).grid(row=45)
Label(master, text="Alcohol", anchor="nw",font=10, width=50).grid(row=50)
Label(master, text = "Quality", anchor="nw",font=10, width=50).grid(row=55)
Label(master, text = "from 0.0 to 15.0", anchor="nw",font=10, width=50).grid(row=0, column=3)
Label(master, text = "from 0.0 to 1.00", anchor="nw",font=10, width=50).grid(row=5, column=3)
Label(master, text = "from 0.0 to 1.0", anchor="nw",font=10, width=50).grid(row=10, column=3)
Label(master, text = "from 1.0 to 3.0", anchor="nw",font=10, width=50).grid(row=15, column=3)
Label(master, text = "from 0.0 to 1.0", anchor="nw",font=10, width=50).grid(row=20, column=3)
Label(master, text = "from 5.0 to 15.0", anchor="nw",font=10, width=50).grid(row=25, column=3)
Label(master, text = "from 15.0 to 60.0", anchor="nw",font=10, width=50).grid(row=30, column=3)
Label(master, text = "from 0.0 to 1.0", anchor="nw",font=10, width=50).grid(row=35, column=3)
Label(master, text = "from 0 to 14", anchor="nw",font=10, width=50).grid(row=40, column=3)
Label(master, text = "from 0.0 to 1.0", anchor="nw",font=10, width=50).grid(row=45, column=3)
Label(master, text = "from 1.0 to 12.0", anchor="nw",font=10, width=50).grid(row=50, column=3)

e1 = Entry(master)
e2 = Entry(master)
e3 = Entry(master)
e4 = Entry(master)
e5 = Entry(master)
e6 = Entry(master)
e7 = Entry(master)
e8 = Entry(master)
e9 = Entry(master)
e10 = Entry(master)
e11 = Entry(master)
e12 = Entry(master)
e13 = Entry(master)
e14 = Entry(master)
e15 = Entry(master)
e16 = Entry(master)
e17 = Entry(master)
e18 = Entry(master)
e19 = Entry(master)
e20 = Entry(master)
e21 = Entry(master)
e22 = Entry(master)
quality = Entry(master)

e1.grid(row=0, column=2)
e2.grid(row=5, column=2)
e3.grid(row=10, column=2)
e4.grid(row=15, column=2)
e5.grid(row=20, column=2)
e6.grid(row=25, column=2)
e7.grid(row=30, column=2)
e8.grid(row=35, column=2)
e9.grid(row=40, column=2)
e10.grid(row=45, column=2)
e11.grid(row=50, column=2)
quality.grid(row=55, column=2)
Button(master, text='Quit', command=master.destroy, font=10, width=40).grid(row=52, column=0, sticky=W, pady=4)
Button(master, text='Find Quality', command=showQuality, font=10, width=40).grid(row=52, column=2, sticky=W, pady=4)
mainloop( )


# In[ ]:




