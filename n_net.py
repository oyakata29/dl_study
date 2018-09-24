import numpy as np


def step(x):
	tmp = x > 0
	tmp = tmp.astype(np.int)
	return tmp	

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def ReLU(x):
	return  np.array([i if i > 0 else 0 for i in x])

"""
上記関数のテスト関数
import matplotlib.pyplot as plt
x = np.arange(-5,5,0.1)

y = ReLU(x)
plt.plot(x,y,linestyle="--",label = ReLU)

y = step(x)
plt.plot(x,y,linestyle="dashdot",label = step)

y = sigmoid(x)
plt.plot(x,y,label = sigmoid)

plt.show()
#print(x,y)
"""

#行列を使うと複数の罫線がいっぺんにできるぞ。便利！
# A = w11 * x1 + w12 * x2 + b を全てに対しておこなっいる
x = np.array([1.0,0.5])
w1 = np.array([[0.5,0.3,0.1],[0.5,0.7,-0.9]])
b1 = np.array([1.0,0.2,-0.8])

A1 = np.dot(x,w1) + b1

print(A1) 
