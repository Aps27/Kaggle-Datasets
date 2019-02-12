# Kaggle's titanic Problem : Complete analysis of what sort of people were likely to survive.
'''
Independent variables : Pclass, Age, Sex, Cabin, Embarked (feature set X)
Dependent variables : Survived (target / Y)
'''
# Import the needed referances
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

''' --------------- Data preprocessing ------------------ '''

#loading the data sets from the csv files
train = pd.read_csv('train.csv').iloc[:,[0,1,2,3,4,5,10,11]]
test = pd.read_csv('test.csv').iloc[:,[0,1,2,3,4,9,10]]

# recheck if we have loaded all files correctly.
print('train dataset: %s, test dataset %s' %(str(train.shape), str(test.shape)) )
train.head()
# train.info()

# check if there are null values
test.isnull().sum() # if there is a number other than 0 for a feature, then it means there are null values within the coloumn

# initial visualisation
import seaborn as sns
sns.set()  # seaborn default for plots

# we can use this function for all cetgorical variables to depict relationship 
def barchart(feature):
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived', 'Dead']
    df.plot(kind='bar', stacked = 'True')
    return

barchart('Sex') # more women survived
barchart('Pclass') # class 1 most survived

# Feature Engineering 

# Missing Age
train = train.fillna(train.mean()) # missing values
test = test.fillna(test.mean()) # missing values

#Sex
train['Sex']=pd.get_dummies(train['Sex'])
test['Sex']=pd.get_dummies(test['Sex'])

# Embarked
pc1 = train[train['Pclass']==1]['Embarked'].value_counts()
pc2 = train[train['Pclass']==2]['Embarked'].value_counts()
pc3 = train[train['Pclass']==3]['Embarked'].value_counts()
df = pd.DataFrame([pc1,pc2,pc3])
df.index = ['1st', '2nd', '3rd']
df.plot(kind='bar', stacked=True)

# Missing Embarked
for dataset in train:
    if dataset == "Embarked":
        train['Embarked'] = train['Embarked'].fillna("S")
for dataset in test:
    if dataset == "Embarked":
        test['Embarked'] = test['Embarked'].fillna("S")
        
#train['Embarked']=pd.get_dummies(train['Embarked'])
#test['Embarked']=pd.get_dummies(test['Embarked'])

em_mapping={"S" : 0,"C" : 1, "Q" : 2}

train['Embarked'] = train['Embarked'].map(em_mapping)
test['Embarked'] = test['Embarked'].map(em_mapping)
                
# Cabin
train.Cabin.value_counts()
for dataset in train:
    if dataset == "Cabin":
        train['Cabin'] = train['Cabin'].str[:1]
        
for dataset in test:
    if dataset == "Cabin":
        test['Cabin'] = test['Cabin'].str[:1]

pc1 = train[train['Pclass']==1]['Cabin'].value_counts()
pc2 = train[train['Pclass']==2]['Cabin'].value_counts()
pc3 = train[train['Pclass']==3]['Cabin'].value_counts()
df = pd.DataFrame([pc1,pc2,pc3])
df.index = ['1st', '2nd', '3rd']
df.plot(kind='bar', stacked=True)

cabin_map={"A" : 0,"B" : 0.4,"C" : 0.8,"D" : 1.2,"E" : 1.6,"F" : 2,"G" : 2.4,"T" : 2.8}
train['Cabin'] = train['Cabin'].map(cabin_map)
test['Cabin'] = test['Cabin'].map(cabin_map)

# missing cabin values
train['Cabin']= train.groupby(['Pclass'])['Cabin'].apply(lambda x : x.fillna(x.median()))
test['Cabin']= test.groupby(['Pclass'])['Cabin'].apply(lambda x : x.fillna(x.median()))

# Names

test['Name'] = test['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
train['Name'] = train['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())
title = np.concatenate((train['Name'].unique(),test['Name'].unique()))
title = np.unique(title)
#title

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2,"Master": 1, "Dr": 3, "Rev": 3, "Col": 3, "Major": 3, "Mlle": 3,"the Countess": 3,"Ms": 1, "Lady": 3, "Jonkheer": 3, "Don": 3, "Dona" : 3, "Mme": 3,"Capt": 3,"Sir": 3}
train['Name'] = train['Name'].map(title_mapping)
test['Name'] = test['Name'].map(title_mapping)

''' -------------- Used for training ---------------- '''

x = train.iloc[:,[0,2,3,4,5,6,7]] # feature matrix
y = train['Survived'] # dep variables

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.7, random_state=42) 

''' --------------- Algorithm ------------------ '''

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import KFold

kfold = KFold(n_splits = 10, shuffle = True, random_state = 0 )

# Try for each method and take one with max accuracy
# Eg:
clf = knn(n_neighbors = 13)
score = cross_val_score(clf, xtrain, ytrain, cv=kfold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # 59%

clf = DecisionTreeClassifier()
score = cross_val_score(clf, xtrain, ytrain, cv=kfold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # 72%

clf = RandomForestClassifier(n_estimators=100)
score = cross_val_score(clf, xtrain, ytrain, cv=kfold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # 78%

clf = GaussianNB()
score = cross_val_score(clf, xtrain, ytrain, cv=kfold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # 75%

clf = SVC()
score = cross_val_score(clf, xtrain, ytrain, cv=kfold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # 62%

clf = MLPClassifier()
score = cross_val_score(clf, xtrain, ytrain, cv=kfold, n_jobs=1, scoring='accuracy')
round(np.mean(score)*100, 2) # 58%

''' --------------- Testing ------------------ '''

# When you test, we wont have target variable coz we have already trained it

clf = RandomForestClassifier(n_estimators=100)
clf.fit(xtrain,ytrain)

# rechecking
ypred = clf.predict(xtest) # has to give values similar to ytest

# Fitting the Confusion Matrix : evaluate model
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(ytest, ypred) # (0,0) and (1,1) correct pred

# final prediction
prediction = clf.predict(test)

''' --------------- Kaggle Submission ------------------ '''

submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission.head()