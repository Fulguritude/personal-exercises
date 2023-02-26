from ex00 import entropy
from ex01 import gini
from ex02 import information_gain


#https://victorzhou.com/blog/intro-to-random-forests/

#ex00
print("\nex00")

print(entropy([]))
#None
print(entropy({1, 2}))
#None
print(entropy('bob'))
#None
print(entropy([0, 0, 0, 0, 0, 0]))
#0.0
print(entropy([6]))
#0.0
print(entropy(['a', 'a', 'b', 'b']))
#1.0
print(entropy(['0', '0', '1', '0', 'bob', '1']))
#1.4591479170272448
print(entropy([0, 0, 1, 0, 2, 1]))
#1.4591479170272448
print(entropy(['0', 'bob', '1']))
#1.584962500721156
print(entropy([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
#0.0
print(entropy([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
#0.4689955935892812
print(entropy([0, 0, 1]))
#0.9182958340544896



#ex01
print("\nex01")

print(gini([]))
#None
print(gini({1, 2}))
#None
print(gini('bob'))
#None
print(gini([0, 0, 0, 0, 0, 0]))
#0.0
print(gini([6]))
#0.0
print(gini(['a', 'a', 'b', 'b']))
#0.5
print(gini(['0', '0', '1', '0', 'bob', '1']))
#0.6111111111111112 
print(gini([0, 0, 1, 0, 2, 1]))
#0.6111111111111112
print(gini(['0', 'bob', '1']))
#0.6666666666666667
print(gini([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
#0.0
print(gini([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
#0.18
print(gini([0, 0, 1]))
#0.4444444444444445


#ex02
print("\nex02")

print(information_gain([], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], "gini"))
print(information_gain([], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], "entropy"))
#Information gain between [] and [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
#None with criterion 'gini' 
#None with criterion 'entropy'

print(information_gain(['a', 'a', 'b', 'b'], {1, 2}, "gini"))
print(information_gain(['a', 'a', 'b', 'b'], {1, 2}, "entropy"))
#Information gain between ['a' 'a' 'b' 'b'] and {1, 2}
#None with criterion 'gini' 
#None with criterion 'entropy'

print(information_gain([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], "gini"))
print(information_gain([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], "entropy"))
#Information gain between [0. 1., 1., 1., 1., 1., 1., 1., 1., 1.] and [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
#0.18 with criterion 'gini'
#0.4689955935892812 with criterion 'entropy'

print(information_gain(['0', '0', '1', '0', 'bob', '1'], [0, 0, 1, 0, 2, 1], "gini"))
print(information_gain(['0', '0', '1', '0', 'bob', '1'], [0, 0, 1, 0, 2, 1], "entropy"))
#Information gain between ['0' '0' '1' '0' 'bob' '1'] and [0 0 1 0 2 1]
#0.0 with criterion 'gini'
#0.0 with criterion 'entropy'

