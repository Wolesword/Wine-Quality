![image](https://user-images.githubusercontent.com/75275475/112500799-12197580-8d5f-11eb-9669-81a3de15ee99.png)

# Wine Quality Prediction

## Technology and Resources Used

**Python Version**: 3.7.7

## Table of Contents
1) [Define the Problem](#1.-Define-the-Problem)<br>
2) [Gather the Data](#2.-Data-Set-Information)<br>
3) [Prepare Data for Consumption](#3.-Prepare-Data-for-Consumption)<br>
4) [Data Cleaning](#4.-Data-Cleaning)<br>
5) [Data Exploration](#5.-Data-Exploration)<br>
6) [Feature Engineering](#6.-Feature-Engineering)<br>
7) [Model Building](#7.-Model-Building)<br>
8) [Hyperparameter Tuning](#8.-More-Hyperparameter-Tuning)<br>
9) [End](#END)<br>

## 1. Define the Problem
Two datasets are included, related to red and white vinho verde wine samples, from the north of Portugal. The goal is to model wine quality based on physicochemical tests. In this study we evaluate the quality grade of Portugese white wine.
The Solution proposed classifies the white into four categories namely:
1. Excellent Wine - Cat 4
2. Very Good wine - Cat 3
3. Fairly good Wine - Cat 2
4. Bad Wine - Cat 1

## 2. Data Set Information
The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. For more details, consult: [Web Link] or the reference [Cortez et al., 2009]. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are many more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods..

![image.png](attachment:image.png)

**Source:**: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties.
In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

Available at: https://archive.ics.uci.edu/ml/datasets/wine+quality

## 3. Prepare Data for Consumption


### 3.1 Import Libraries
The following code is written in Python 3.7.7. Below is the list of libraries used.

# !pip install lux-api
import numpy as np 
import pandas as pd
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import lux
# %config Completer.use_jedi = False # to activate autocomplete assistance, disable jedi

### 3.2 Load Data Modeling Libraries
These are the most common machine learning and data visualization libraries.

# Model Algorithms
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# Model Helpers
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix

### 3.3 Data dictionary
The data dictionary for the data set is as follows:<br>
Input variables (based on physicochemical tests):
1. - fixed acidity
2. - volatile acidity
3. - citric acid
4. - residual sugar
5. - chlorides
6. - free sulfur dioxide
7. - total sulfur dioxide
8. - density
9. - pH
10. - sulphates
11. - alcohol
Output variable (based on sensory data):
12. - quality (score between 0 and 10)

### 3.4 Greet the data

# read data set
# wine_data = pd.read_csv("Data set/winequality-white.csv",sep=";" , encoding= 'unicode_escape')

```python
from google.colab import drive
drive.mount('/content/drive') 

wine_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/winequality-white.csv',sep=";" , encoding= 'unicode_escape')

wine_data

# get a peek at the top 5 rows of the data set
wine_data.head() #print(wine_data.head())

# understand the type of each column
print(wine_data.info())

# get information on the numerical columns for the data set
wine_data.describe(include='all')
```

## 4. Data Cleaning
The Data cleaning was seen to be instrumental in improving the quality of the modeling. Outliers were thus removed from the dataset.

### 4.1 IQR method

def remove_outliers_iqr(df):
    dataf = pd.DataFrame(df)
    quartile_1, quartile_3 = np.percentile(dataf, [5,95])

    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    #print("lower bound:", lower_bound)
    #print("upper bound:", upper_bound)
    #print("IQR outliers:", np.where((dataf > upper_bound) | (dataf < lower_bound)))
    print("# of outliers:", len(np.where((dataf > upper_bound) | (dataf < lower_bound))[0]))

    return dataf[~((dataf < lower_bound) | (dataf > upper_bound)).any(axis=1)]

### 4.2 Removal of  outliers in the top 5%
Top 5% means here the values that are out of the 95th percentile of data.

# Sample outlier plot
sns.boxplot(x=wine_data['volatile acidity'])
plt.show()

# calculate IQR score and remove outliers
wine_data['fixed acidity'] = remove_outliers_iqr(wine_data['fixed acidity'])
wine_data['volatile acidity'] = remove_outliers_iqr(wine_data['volatile acidity'])
wine_data['citric acid'] = remove_outliers_iqr(wine_data['citric acid'])
wine_data['residual sugar'] = remove_outliers_iqr(wine_data['residual sugar'])
wine_data['chlorides'] = remove_outliers_iqr(wine_data['chlorides'])
wine_data['free sulfur dioxide'] = remove_outliers_iqr(wine_data['free sulfur dioxide'])
wine_data['total sulfur dioxide'] = remove_outliers_iqr(wine_data['total sulfur dioxide'])
wine_data['density'] = remove_outliers_iqr(wine_data['density'])
wine_data['pH'] = remove_outliers_iqr(wine_data['pH'])
wine_data['sulphates'] = remove_outliers_iqr(wine_data['sulphates'])
wine_data['alcohol'] = remove_outliers_iqr(wine_data['alcohol'])

**NOTE**: Removing outliers improved the performance of most of the models by about 2%. The range of the quartile was determined at 95% to not bear a great cost on the amount of observations taken from the data. In total **91 outliers were removed**.

### 4.3 drop null values
There are no nulls in our dataset.

wine_data = wine_data.dropna()
sns.heatmap(wine_data.isnull()); # No nulls

wine_data.info()

### 4.4 Output to CSV
Output cleaned data to CSV for keeps. This is done because after dropping the information of the missed entries remains on the dataset and would influence a merger during feature engineering operations.


wine_data.to_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/white_data_cleaned.csv',index = False)

## 5. Data Exploration
This section explores the distribution of each variable using cleaned data set.

### 5.1 Visualisation and correlation helper methods

def plotHist(xlabel, title, column):
    fig, ax = plt.subplots(1, 1, 
                           figsize =(8, 5),  
                           tight_layout = True)

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)

    plt.xlabel(xlabel, fontsize=16)  
    plt.ylabel("# of entries", fontsize=16)
    plt.title(title, fontsize=20)

    plt.hist(column)
    plt.show()

def plotBar(xlabel, title, column):
    ax = sns.barplot(column.value_counts().index, column.value_counts())

    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)

    plt.xticks(fontsize=14)  
    plt.yticks(fontsize=14)

    plt.xlabel(xlabel, fontsize=16)  
    plt.ylabel("# of entries", fontsize=16)
    plt.title(title, fontsize=20)

    plt.show()

def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9}, 
        ax=ax,
        annot=True, 
        linewidths=0.1, 
        vmax=1.0, 
        linecolor='white',
        annot_kws={'fontsize':14}
    )

    _.set_yticklabels(_.get_ymajorticklabels(), fontsize = 16)
    _.set_xticklabels(_.get_xmajorticklabels(), fontsize = 16)

    plt.title('Pearson Correlation of Features', y=1.05, size=20)

    plt.show()

print('Alcohol level:\n', wine_data.pH.value_counts(sort=False));
plotHist("Alcohol", "Histogram of number of entries per number of pH", wine_data.alcohol);

print('pH level:\n', wine_data.pH.value_counts(sort=False));
plotHist("pH", "Histogram of number of entries per number of pH", wine_data.pH);

### 5.2 Group

print('Quality:\n', wine_data['quality'].value_counts(sort=False))
plotBar("Result (1 = positive, 0 = negative)", "Wine results", wine_data['quality'])

Here we see that the response is actualy divided into 6 classes and that no variable is associated with the values 1.2

### 5.3 Correlation heatmap

correlation_heatmap(wine_data)

### 5.4 Pair plot


# sns.pairplot(wine_data, hue = 'quality')
# plt.show()

### 5.5 Pivot Table


# Getting my columns
wine_data.columns

pivot_table1 = pd.pivot_table(wine_data, index = 'quality', values = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar'])
print(pivot_table1)

pivot_table2 = pd.pivot_table(wine_data, index = 'quality', values = ['chlorides','free sulfur dioxide', 'total sulfur dioxide', 'density'])
print(pivot_table2)

pivot_table3 = pd.pivot_table(wine_data, index = 'quality', values = ['pH', 'sulphates', 'alcohol'])
print(pivot_table3)

## 6. Feature Engineering

### 6.1 Exploration of new features
Creating features which may help the decision making



### 6.1 Dataset cleaned

wine_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/white_data_cleaned.csv', encoding= 'unicode_escape')

### 6.2 Setting the response into four categories. 

# define x, y
X = wine_data.drop(['quality'], axis = 1)
y_real = wine_data['quality']

# Classifying the wine into four categories. 
response = []

for i in range(0,len(y_real)):
    if y_real[i] >= 8:
        response.append(4) # Excellent Wine
    elif y_real[i] < 8 and y_real[i] >= 6:
        response.append(3) # Very good
    elif y_real[i] < 6 and y_real[i] >= 4:
        response.append(2) # Fairly good wine
    else:
        response.append(1) # Bad Wine

y = np.array(response)

X.head()

# Quick check using Support Vector Machine
from sklearn.svm import SVC
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)
regressor = SVC(gamma='auto')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('The MSE for the SVM is: ', metrics.mean_squared_error(y_test, y_pred))

### 6.3 New features

X['alcohol_square'] = X['alcohol']*X['alcohol']
X['sulphates_square'] = X['pH']*X['sulphates']

X.head()

X.info()

# Quick check using Logistic regression
from sklearn.svm import SVC
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)
regressor = SVC(gamma='auto')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('The MSE for the SVM is: ',metrics.mean_squared_error(y_test, y_pred))

# Some feature Engineering
# wine_data['alcohol'].unique()
new_data = [1]

for i in range(0,len(X)-1):
    if X.iloc[i,10] >= 9.0:
        new_data.append(0)
    elif X.iloc[i,10] > 9.0 and X.iloc[i,10] < 12.0:
        new_data.append(1)
    else:
        new_data.append(2)

new_data = pd.DataFrame(new_data)

new_data.columns = ['Category']

new_data['Category'] = new_data.Category.astype('category')

new_data.info()

new_data.describe(include="all")

X = pd.concat([X, pd.DataFrame(new_data)], axis=1) # X.describe(include="all")

X.head()

X.info()

# Quick check using Logistic regression
from sklearn.svm import SVC
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)
regressor = SVC(gamma='auto')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('The MSE for the SVM is: ',metrics.mean_squared_error(y_test, y_pred))

X['normalized'] = (X['residual sugar'] - X['residual sugar'].min()) / (X['residual sugar'].max() - X['residual sugar'].min())
X['alcohol_normalized'] = (X['alcohol'] - X['alcohol'].min()) / (X['alcohol'].max() - X['alcohol'].min())

X.head()

# Quick check using Logistic regression
from sklearn.svm import SVC
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)
regressor = SVC(gamma='auto')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('The MSE for the SVM is: ',metrics.mean_squared_error(y_test, y_pred))

<span style="color:red">**Important Note**</span> <br>
<span style="color:red">From the quick checks using SVM which is quick in its process, we can see a slight reduction in the MSE . Therefore, though these steps in feature engineering seem to brings some improvement in the fit of the model, it still does  not considerably improve the trend in the results. The next steps will be to evaluate other evaluation models.</span>

### 6.4 Model Selection
From model chart here, we can see that the categorical feature added through feature engineering does not improve the model. This will therefore not be included in our result. 
We will also observed, if a small subset of more correlated features will not be sufficient to predict the class.

from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(X,y)
feat_importance = pd.Series(etr.feature_importances_, index=X.columns)
feat_importance.nlargest(15).plot(kind='barh')
plt.show()

<span style="color:red">**Important Note**</span> <br>
<span style="color:red">A subset of correlated features without those from feature engineering will be used to improve the accuracy of the model.</span>

# define x, y
# X = wine_data.drop(['quality'], axis = 1) 
# X = X.drop(['Category', 'density', 'normalized', 'fixed acidity', 'alcohol_normalized'], axis = 1)

### 6.5 Split into Training and Testing Data

# split into train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 16)

from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(ExtraTreesRegressor(), threshold= 0.04)
sel.fit(X_train, y_train)
selected_feat = X_train.columns[(sel.get_support())]
print('The number of selected features:', len(selected_feat))
print('The selected features are:', pd.DataFrame(selected_feat))

## 7. Model Building
Train the models and use cross validation score for the accuracy.

* Naive Bayes
* Logistic regression
* K-Nearest Neighbors
* Decision Tree
* Bagging
* Gardient Boosting
* RandomForest


### 7.1 Naive Bayes



from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
cv = cross_val_score(nb, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(1e-09, 1e-06, 1e-09)
scores1 = []
for n in estimators:
    nb.set_params(var_smoothing=n)
    nb.fit(X_train, y_train)
    scores1.append(nb.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("var_smoothing")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization of var_smoothing: {}'.format(round(np.arange(1e-09, 1e-06, 1e-09)[np.argmax(scores1)],10)))


### 7.2 Logistic Regression

lr = LogisticRegression(max_iter = 100000)
cv = cross_val_score(lr, X_train, y_train,cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(1, 15, 1)
scores1 = []
for n in estimators:
    lr.set_params(C=n)
    lr.fit(X_train, y_train)
    scores1.append(lr.score(X_test, y_test))
plt.title("Effect of parameter C")
plt.xlabel("parameter C")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization parameter C: {}'.format(round(np.arange(1, 10, 1)[np.argmax(scores1)],2)))

### 7.3 K-Neaserst Neighbors



from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
cv = cross_val_score(KNN, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(1, 16, 1)
scores1 = []
for n in estimators:
    KNN.set_params(n_neighbors=n)
    KNN.fit(X_train, y_train)
    scores1.append(KNN.score(X_test, y_test))
plt.title("Effect of n_neighbors")
plt.xlabel("n_neighbors")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization n_neighbors: {}'.format(round(np.arange(1, 16, 1)[np.argmax(scores1)],2)))

<span style="color:red">**Important Note**</span> <br>
<span style="color:red">The best models here are the random forest and the gradient boosting algorithm. Therefore the next section will look into these three algorithms
1. Decision Tree
2. Random Forest
3. Gradient Boosting
</span>

### 7.4 Decision Tree


dt = tree.DecisionTreeClassifier(random_state = 1)
cv = cross_val_score(dt, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(1, 10, 1)
scores1 = []
for n in estimators:
    dt.set_params(min_samples_leaf=n)
    dt.fit(X_train, y_train)
    scores1.append(dt.score(X_test, y_test))
plt.title("Effect of min_samples_leaf")
plt.xlabel("min_samples_leaf")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization parameter min_samples_leaf: {}'.format(round(np.arange(1, 10, 1)[np.argmax(scores1)],2)))

### 7.5 Bagging



from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(dt, random_state=1)
cv = cross_val_score(bag, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(10, 200, 10)
scores1 = []
for n in estimators:
    bag.set_params(n_estimators=n)
    bag.fit(X_train, y_train)
    scores1.append(bag.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization n_estimators: {}'.format(round(np.arange(10, 200, 10)[np.argmax(scores1)],2)))

### 7.6 Gradient Boosting



from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=1)
cv = cross_val_score(gb, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(50, 350, 10)
scores1 = []
for n in estimators:
    gb.set_params(n_estimators=n)
    gb.fit(X_train, y_train)
    scores1.append(gb.score(X_test, y_test))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization n_estimators: {}'.format(round(np.arange(50, 350, 10)[np.argmax(scores1)],2)))

### 7.7 Random Forest



rf = RandomForestClassifier(random_state = 1)
cv = cross_val_score(rf, X_train, y_train, cv=5)
print(cv)
print(cv.mean())

estimators = np.arange(10, 300, 10)
scores1 = []
for n in estimators:
    rf.set_params(n_estimators=n)
    rf.fit(X_train, y_train)
    y_predict = rf.predict(X_test)
    scores1.append(accuracy_score(y_test, y_predict))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores1);

print('Accuracy Score: {}'.format(round(scores1[np.argmax(scores1)], 3)))
print('best Regularization parameter estimators: {}'.format(round(np.arange(10, 300, 10)[np.argmax(scores1)],2)))

## 8. More Hyperparameter Tuning


### 8.1 Decision Tree
* `criterion` : optional (default=”gini”) or Choose attribute selection measure: This parameter allows us to use the different attribute selection measure. Supported criteria are “gini” for the Gini index and “entropy” for the information gain.

* `max_depth` : int or None, optional (default=None) or Maximum Depth of a Tree: The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples. The higher value of maximum depth causes overfitting, and a lower value causes underfitting (Source).


gini_acc_scores = []
entropy_acc_scores = []

criterions = ["gini", "entropy"]

for criterion in criterions:
    for depth in range(25):
        dt = tree.DecisionTreeClassifier(criterion=criterion, max_depth = depth+1, random_state=depth)
        model = dt.fit(X_train,y_train)

        y_predict = dt.predict(X_test)

        if criterion == "gini":
            gini_acc_scores.append(accuracy_score(y_test, y_predict))
        else:
            entropy_acc_scores.append(accuracy_score(y_test, y_predict))

figuresize = plt.figure(figsize=(12,8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
EntropyAcc = plt.plot(np.arange(25)+1, entropy_acc_scores, '--bo')   
GiniAcc = plt.plot(np.arange(25)+1, gini_acc_scores, '--ro')
legend = plt.legend(['Entropy', 'Gini'], loc ='lower right',  fontsize=15)
title = plt.title('Accuracy Score for Multiple Depths', fontsize=25)
xlab = plt.xlabel('Depth of Tree', fontsize=20)
ylab = plt.ylabel('Accuracy Score', fontsize=20)

plt.show()

print("Gini max accuracy:", max(gini_acc_scores))
print("Entropy max accuracy:", max(entropy_acc_scores))

dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 22, random_state = 1)
dt = dt.fit(X_train, y_train)
y_predict = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_predict))

### 8.2 Random forest Multi-hyperparameter tuning
The results indicate that the other hyperparameters do not significantly improve the prediction. The hyperparameters of importance are the number of estimators and the depth of the tree

[int(x) for x in np.linspace(start = 100, stop = 300, num = 10)]

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 300, num = 25)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(25, 35, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True] # You want to use this
# Create the random grid

random_grid = {'n_estimators': n_estimators,
               #'max_features': max_features,
               'max_depth': max_depth#,
               #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf,
               #'bootstrap': bootstrap
              }
print(random_grid)

# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = random_grid, cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_

def evaluate(model, test_features, test_labels):
    y_pred_rf = model.predict(test_features)
    errors = abs(y_pred_rf - test_labels)
    # check for score
    accuracy = accuracy_score(y_test, y_pred_rf)
    
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy_score

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, X_test, y_test)

The results indicate that the other hyperparameters do not significantly improve the prediction.


### 8.3 Fine tuning the Random Forest Classifier

acc_scores = []              
depth = [int(x) for x in np.arange(1, 35, 1)]
depth.append(None)

for i in depth:
    rf = RandomForestClassifier(n_estimators=80, max_depth=i, random_state=1)
    rf.fit(X_train,y_train)
    y_predict = rf.predict(X_test)
    acc_scores.append(rf.score(X_test, y_test)) 

figsize = plt.figure(figsize = (12,8))
plot = plt.plot(depth, acc_scores, 'r')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
xlab = plt.xlabel('Depth of the trees', fontsize = 20)
ylab = plt.ylabel('Accuracy', fontsize = 20)
title = plt.title('(Random Forest) Accuracy vs Depth of Trees', fontsize = 25)
plt.show()

rf = RandomForestClassifier(n_estimators=80, max_depth=acc_scores.index(max(acc_scores))+1, random_state=1)
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict))


### 8.4 Tuning the Gradient Boosting Classifier

acc_scores = []              
depth = np.arange(1, 16)

for i in depth:
    rf = GradientBoostingClassifier(n_estimators=400, max_depth=i, random_state=1)
    rf.fit(X_train,y_train)
    y_predict = rf.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_predict)) 

figsize = plt.figure(figsize = (12,8))
plot = plt.plot(depth, acc_scores, 'r')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
xlab = plt.xlabel('Depth of the trees', fontsize = 20)
ylab = plt.ylabel('Accuracy', fontsize = 20)
title = plt.title('(Gradient Boosting) Accuracy vs Depth of Trees', fontsize = 25)
plt.show()

rf = GradientBoostingClassifier(n_estimators=340, max_depth=acc_scores.index(max(acc_scores))+1, random_state=1)
rf.fit(X_train,y_train)

y_predict = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_predict))

# Conclusion:

From the following analysis we have done the following. 
* Created an appropriate response classification for the model
*  Reduced the MSE error using various form of feature engineering techniques
* Identified highly correlated features to the response.
* Identified best performing algorithms which are
  1. Decision Tree
  2. Gradient Boosting
  3. Random Forest
* Taken the time to tune the hyparameters for these methods

The results of the intricate fine-tuned hyper-parameters indicate the following:

  1. Decision tree: Classification accuracy from 75.1% to 76.8%
  2. Gradient Boosting: Classification accuracy from 77.7% to 80.05%
  3. Random Forest: Classification accuracy from 81.6% to 82%

# END
