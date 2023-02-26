"""

Positive/Negative: outcome of the test
True/False: whether that test actually reflects the underlying reality or not

False negative: test turned out negative, but it is false, and the disease is in fact present
False positive: test turned out positive, but it is false, and there is in fact no disease present.
True negative: test turned out negative, and this truly reflects that there is no disease present
True positive: test turned out positive, and this truly represents that the disease is present

Positive/Negative: outcome of the prediction
	=> y_pred_i = 1 iff y_pred_i \in Positive
	=> y_pred_i = 0 iff y_pred_i \in Negative;)
True/False: whether this actually describes the underlying reality or not 
	=> y_pred_i == y_true_i <=> True 
	=> y_pred_i != y_true_i <=> False


"""



#https://towardsdatascience.com/logistic-regression-classifier-on-census-income-data-e1dbef0b5738

import numpy as np
import pandas as pd
import io
import requests
import seaborn as sns
from matplotlib import pyplot as plt
import pickle
import os
from pandas.api.types import CategoricalDtypefrom sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import cross_val_score

%matplotlib inline


#1 DATASET LOADING
#function to load dataset from web
def load_dataset(path, urls):
	if not os.path.exists(path):
		os.mkdir(path)

	for url in urls:
		data = requests.get(url).content
		filename = os.path.join(path, os.path.basename(url))
		with open(filename, "wb") as file:
			file.write(data)



urls = ["http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
		"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names",
		"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"]

columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status",
			"occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
			"hours-per-week", "native-country", "income"]
#sep=" *, *" removes whitespace; skiprows=1 skips the first line; na_values="?" set "?" as the value for missing fields
train_data = pd.read_csv('data/adult.data', names=columns, sep=' *, *', na_values='?')
test_data	= pd.read_csv('data/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')



#2 DIAGNOSTICS
#useful functions for basic diagnostics on the dataset

print(train_data.info())

num_attributes = train_data.select_dtypes(include=['int'])
print(num_attributes.columns)['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
num_attributes.hist(figsize=(10,10))

train_data.describe()

sns.countplot(y='workClass', hue='income', data = cat_attributes
sns.countplot(y='occupation', hue='income', data = cat_attributes)



#3 PIPELINES
#preparing the data for use

class ColumnsSelector(BaseEstimator, TransformerMixin):
	
	def __init__(self, type):
		self.type = type

	def fit(self, X, y=None):
		return self

	def transform(self,X):
		return X.select_dtypes(include=[self.type])


num_pipeline = Pipeline(steps=[
	("num_attr_selector", ColumnsSelector(type='int')),
	("scaler", StandardScaler())
])


class CategoricalImputer(BaseEstimator, TransformerMixin):
	
	def __init__(self, columns = None, strategy='most_frequent'):
		self.columns = columns
		self.strategy = strategy

	def fit(self,X, y=None):
		if self.columns is None:
			self.columns = X.columns
		
		if self.strategy is 'most_frequent':
			self.fill = {column: X[column].value_counts().index[0] for 
				column in self.columns}
		else:
			self.fill ={column: '0' for column in self.columns}
			
		return self

	def transform(self,X):
		X_copy = X.copy()
		for column in self.columns:
			X_copy[column] = X_copy[column].fillna(self.fill[column])
		return X_copy


class CategoricalEncoder(BaseEstimator, TransformerMixin):
	
	def __init__(self, dropFirst=True):
		self.categories=dict()
		self.dropFirst=dropFirst
		
	def fit(self, X, y=None):
		join_df = pd.concat([train_data, test_data])
		join_df = join_df.select_dtypes(include=['object'])
		for column in join_df.columns:
			self.categories[column] = 
					join_df[column].value_counts().index.tolist()
		return self
		
	def transform(self, X):
		X_copy = X.copy()
		X_copy = X_copy.select_dtypes(include=['object'])
		for column in X_copy.columns:
			X_copy[column] = X_copy[column].astype({column:
								CategoricalDtype(self.categories[column])})
		return pd.get_dummies(X_copy, drop_first=self.dropFirst)


cat_pipeline = Pipeline(steps=[
    ("cat_attr_selector", ColumnsSelector(type='object')),
    ("cat_imputer", CategoricalImputer(columns=
          ['workClass','occupation', 'native-country'])),
    ("encoder", CategoricalEncoder(dropFirst=True))
])

full_pipeline = FeatureUnion([("num_pipe", num_pipeline), 
                ("cat_pipeline", cat_pipeline)])



#4 TRAINING
#

train_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
test_data.drop(['fnlwgt', 'education'], axis=1, inplace=True)

train_copy = train_data.copy()
train_copy["income"] = train_copy["income"].apply(lambda x:0 if x =='<=50K' else 1)
X_train = train_copy.drop('income', axis =1)
Y_train = train_copy['income']

X_train_processed=full_pipeline.fit_transform(X_train)
model = LogisticRegression(random_state=0)
model.fit(X_train_processed, Y_train)



#5 TESTING
#


test_copy = test_data.copy()
test_copy["income"] = test_copy["income"].apply(lambda x:0 if x == '<=50K.' else 1)
X_test = test_copy.drop('income', axis =1)
Y_test = test_copy['income']


X_test_processed = full_pipeline.fit_transform(X_test)
predicted_classes = model.predict(X_test_processed)



#6 EVALUATION

accuracy_score(predicted_classes, Y_test.values)
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
cfm = confusion_matrix(predicted_classes, Y_test.values)
sns.heatmap(cfm, annot=True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')


#cross validation
cross_val_model = LogisticRegression(random_state=0)
scores = cross_val_score(cross_val_model, X_train_processed, Y_train, cv=5)
print(np.mean(scores))



#7 FINE TUNING
# penalty specifies the norm in the penalization
penalty = ['l1', 'l2']

# C is the inverse of the regularization parameter
C = np.logspace(0, 4, 10)random_state=[0]

# creating a dictionary of hyperparameters
hyperparameters = dict(C=C, penalty=penalty, random_state=random_state)


#Using GridSearchCV to find the optimal parameters
clf = GridSearchCV(estimator = model, param_grid = hyperparameters, cv=5)
best_model = clf.fit(X_train_processed, Y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

best_predicted_values = best_model.predict(X_test_processed)
accuracy_score(best_predicted_values, Y_test.values)



#8 SAVING THE MODEL
filename = 'final_model.sav'
pickle.dump(model, open(filename, 'wb'))


#9 LOADING
#saved_model = pickle.load(open(filename, 'rb')) 