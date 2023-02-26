import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


from ex00 import sigmoid_
from ex01 import log_predict_, log_loss_
from ex02 import log_gradient_
from ex03 import vec_log_predict_, vec_log_loss_
from ex04 import vec_log_gradient_
from ex05 import LogisticRegression
from ex06 import accuracy_score_
from ex07 import precision_score_
from ex08 import recall_score_
from ex09 import f1_score_
from ex10 import confusion_matrix_


#ex00
print("ex00")
x = -4
print(sigmoid_(x))
# 0.01798620996209156
x= 2
print(sigmoid_(x))
# 0.8807970779778823
x = [-4, 2, 0]
print(sigmoid_(x))
# [0.01798620996209156, 0.8807970779778823, 0.5]



#ex01
print("\nex01")
# Test n.1
x = 4
y_true = 1
theta = [0., 0.5]
y_pred = log_predict_(theta, x)
m = 1 # length of y_true is 1
print(log_loss_(y_true, y_pred, m))
# 0.12692801104297152

# Test n.2
x = [1, 2, 3, 4]
y_true = 0
theta = [0., -1.5, 2.3, 1.4, 0.7]
#x_dot_theta = sum([a*b for a, b in zip(x, theta)])
#y_pred = sigmoid_(x_dot_theta)
y_pred = log_predict_(theta, x)
m = 1
print(log_loss_(y_true, y_pred, m))
# 10.100041078687479

# Test n.3
x = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
y_true = [1, 0, 1]
theta = [0., -1.5, 2.3, 1.4, 0.7]
x_dot_theta = []
#for i in range(len(x)):
#	my_sum = 0
#	for j in range(len(x[i])):
#		my_sum += x[i][j] * theta[j]
#	x_dot_theta.append(my_sum)
#y_pred = sigmoid_(x_dot_theta)
y_pred = log_predict_(theta, x)
m = len(y_true)
print(log_loss_(y_true, y_pred, m))
# 7.233346147374828



#ex02
print("\nex02")
# Test n.1
x = [4.2] # 1 represent the intercept
y_true = 1
theta = [0.5, -0.5]
#x_dot_theta = sum([a*b for a, b in zip(x, theta)])
#y_pred = sigmoid_(x_dot_theta)
y_pred = log_predict_(theta, x)
print(log_gradient_(x, y_pred, y_true))
# [0.8320183851339245, 3.494477217562483]

# Test n.2
x = [-0.5, 2.3, -1.5, 3.2]
y_true = 0
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
#x_dot_theta = sum([a*b for a, b in zip(x, theta)])
#y_pred = sigmoid_(x_dot_theta)
y_pred = log_predict_(theta, x)
print(log_gradient_(x, y_true, y_pred))
# [0.99999685596372, -0.49999842798186, 2.299992768716556, -1.4999952839455801, 3.1999899390839044]

# Test n.3
x = [[2, 3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
# first column of x_new are intercept values initialized to 1
y_true = [1, 0, 1]
theta = [0.5, -0.5, 1.2, -1.2, 2.3]
#x_dot_theta = []
#for i in range(len(x)):
#	my_sum = 0
#	for j in range(len(x[i])):
#		my_sum += x[i][j] * theta[j]
#	x_dot_theta.append(my_sum)
#y_pred = sigmoid_(x_dot_theta)
y_pred = log_predict_(theta, x)
print(log_gradient_(x, y_true, y_pred))
# [0.9999445100449934, 5.999888854245219, 6.999833364290213, 7.999777874335206, 8.999722384380199]



#ex03
print("\nex03")
# Test n.1
x= 4
y_true = 1
theta = 0.5
y_pred = sigmoid_(x * theta)
m = 1 # length of y_true is 1
print(vec_log_loss_(y_true, y_pred, m))
# 0.12692801104297152

# Test n.2
x = np.array([1, 2, 3, 4])
y_true = 0
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x, theta))
m= 1
print(vec_log_loss_(y_true, y_pred, m))
# 10.100041078687479

# Test n.3
x_new = np.arange(1, 13).reshape((3, 4))
y_true = np.array([1, 0, 1])
theta = np.array([-1.5, 2.3, 1.4, 0.7])
y_pred = sigmoid_(np.dot(x_new, theta))
m = len(y_true)
print(vec_log_loss_(y_true, y_pred, m))
# 7.233346147374828



#ex04
print("\nex04")
# Test n.1
x = np.array([1, 4.2]) # x[0] represent the intercept
y_true = 1
theta = np.array([0.5, -0.5])
y_pred = sigmoid_(np.dot(x, theta))
print(vec_log_gradient_(x, y_pred, y_true))
# [0.83201839 3.49447722]

# Test n.2
x = np.array([1, -0.5, 2.3, -1.5, 3.2]) # x[0] represent the intercept
y_true = 0
theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
y_pred = sigmoid_(np.dot(x, theta))
print(vec_log_gradient_(x, y_true, y_pred))
# [ 0.99999686 -0.49999843 2.29999277 -1.49999528 3.19998994]

# Test n.3
x_new = np.arange(2, 14).reshape((3, 4))
x_new = np.insert(x_new, 0, 1, axis=1)
# first column of x_new are now intercept values initialized to 1
y_true = np.array([1, 0, 1])
theta = np.array([0.5, -0.5, 1.2, -1.2, 2.3])
y_pred = sigmoid_(np.dot(x_new, theta))
print(vec_log_gradient_(x_new, y_true, y_pred))
# [0.99994451 5.99988885 6.99983336 7.99977787 8.99972238]



#ex05
print("\nex05")
# We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
df_train = pd.read_csv('./resources/train_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
df_test = pd.read_csv('./resources/test_dataset_clean.csv', delimiter=',', header=None, index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]

# We set our model with our hyperparameters : alpha, verbose and learning_rate
model = LogisticRegression(alpha=0.01, n_cycle=10, n_epoch=8, verbose=True, learning_rate_type='invscaling')

# We fit our model to our dataset and display the score for the train and test datasets

y_test = LogisticRegression.np_matrix_from_any(y_test)

#print(y_train)
#print(y_test)

model.train_(x_train, y_train, x_test, y_test, True)
y_train = LogisticRegression.np_matrix_from_any(y_train).T
print(f'Score on train dataset : {model.score_(x_train, y_train)}')
y_pred = model.predict_class_(x_test)

y_test = LogisticRegression.np_matrix_from_any(y_test).T
print(f'Score on test dataset : {(y_pred == y_test).mean()}')

# epoch 0 : loss 2.711028065632692
# epoch 150 : loss 1.760555094793668
# epoch 300 : loss 1.165023422947427
# epoch 450 : loss 0.830808383847448
# epoch 600 : loss 0.652110347325305
# epoch 750 : loss 0.555867078788320
# epoch 900 : loss 0.501596689945403
# epoch 1050 : loss 0.469145216528238
# epoch 1200 : loss 0.448682476966280
# epoch 1350 : loss 0.435197719530431
# epoch 1500 : loss 0.425934034947101
# Score on train dataset : 0.7591904425539756
# Score on test dataset : 0.7637737239727289

# This is an example with verbose set to True, you could choose to display your loss at the epochs you want.
# Here I choose to only display 11 rows no matter how many epochs I had.
# Your score should be pretty close to mine.
# Your loss may be quite different weither you choose different hyperparameters, if you add an intercept to your x_train
# or if you shuffle your x_train at each epochs (this introduce stochasticity !) etc...
# You might not get a score as good as sklearn.linear_model.LogisticRegression because it uses a different algorithm and
# more optimized parameters that would require more time to implement.


#ex06
print("\nex06")
# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(accuracy_score_(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
# 0.5
# 0.5

# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(accuracy_score_(y_true, y_pred))
print(accuracy_score(y_true, y_pred))
# 0.625
# 0.625


#ex07
print("\nex07")
# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(precision_score_(y_true, y_pred))
print(precision_score(y_true, y_pred))
# 0.4
# 0.4

# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(precision_score_(y_true, y_pred, label='dog'))
print(precision_score(y_true, y_pred, pos_label='dog'))
# 0.6
# 0.6

# Test n.3
print(precision_score_(y_true, y_pred, label='norminet'))
print(precision_score(y_true, y_pred, pos_label='norminet'))
# 0.6666666666666666
# 0.6666666666666666



#ex08
print("\nex08")
# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(recall_score_(y_true, y_pred))
print(recall_score(y_true, y_pred))
# 0.6666666666666666
# 0.6666666666666666

# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(recall_score_(y_true, y_pred, label='dog'))
print(recall_score(y_true, y_pred, pos_label='dog'))
# 0.75
# 0.75

# Test n.3
print(recall_score_(y_true, y_pred, label='norminet'))
print(recall_score(y_true, y_pred, pos_label='norminet'))
# 0.5
# 0.5




#ex09
print("\nex09")
# Test n.1
y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
print(f1_score_(y_true, y_pred))
print(f1_score(y_true, y_pred))
# 0.5
# 0.5

# Test n.2
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
print(f1_score_(y_true, y_pred, label='dog'))
print(f1_score(y_true, y_pred, pos_label='dog'))
# 0.6666666666666665
# 0.6666666666666665

# Test n.3
print(f1_score_(y_true, y_pred, label='norminet'))
print(f1_score(y_true, y_pred, pos_label='norminet'))
# 0.5714285714285715
# 0.5714285714285715



#ex10
y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
# [[0 0 0]
# [0 2 1]
# [1 0 2]]

# [[0 0 0]
# [0 2 1]
# [1 0 2]]
print(confusion_matrix_(y_true, y_pred, labels=['dog', 'norminet']))
print(confusion_matrix(y_true, y_pred, labels=['dog', 'norminet']))
# [[2 1]
# [0 2]]

# [[2 1]
# [0 2]]

y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'bird'])
y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet'])
print(confusion_matrix_(y_true, y_pred, df_option=True))
#
# bird
# dog
# norminet
print(confusion_matrix_(y_true, y_pred, labels=['bird', 'dog'], df_option=True))
# bird dog
# bird 0 0
# dog 0 2
