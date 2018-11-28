import numpy as np


def step(x):
	tmp = x > 0
	tmp = tmp.astype(np.int)
	return tmp	

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def ReLU(x):
	return  np.array([i if i > 0 else 0 for i in x])


#出力層の着火関数 
#割り算だから同じ値をかけてやる分には変換なし　というのを利用し
#入力の中の最大値 C を引くことによりオーバーフローを防ぐ
def soft_max(a):
	c = np.max(a)
	exp_a = np.exp(a-c)
	sum_exp = np.sum(exp_a)
	y = exp_a / sum_exp
	return y

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


"""
ニューラルネットワークの計算
#行列を使うと複数の罫線がいっぺんにできるぞ。便利！
# A = w11 * x1 + w12 * x2 + b を全てに対しておこなっいる
#第一層
x = np.array([1.0,0.5])
w1 = np.array([[0.5,0.3,0.1],[0.5,0.7,-0.9]])
b1 = np.array([1.0,0.2,-0.8])

A1 = np.dot(x,w1) + b1
#シグモイドで着火
Z1 = sigmoid(A1)

#第二層
w2 = np.array([[1.0,0.5],[0.2,0.4],[-0.2,-0.5]])
b2 = np.array([-0.7,0.6])

A2 =np.dot(Z1,w2) + b2
Z2 = sigmoid(A2)
"""

#上のコメントアウト部を関数にすると
def init_net():
	network = {}
	network["W1"] = np.array([[0.5,0.3,0.1],[0.5,0.7,-0.9]])
	network["W2"] = np.array([[1.0,0.5],[0.2,0.4],[-0.2,-0.5]]) 
	network["W3"] = np.array([[0.1,0.2],[0.4,-1.0]])
	network["B1"] = np.array([1.0,0.2,-0.8])
	network["B2"] = np.array([-0.7,0.6])
	network["B3"] = np.array([0.1,0.2]) 
	
	return network

def forword(network,x):
	w1,w2,w3 = network["W1"], network["W2"],network["W3"]
	b1,b2,b3 = network["B1"],network["B2"],network["B3"]

	A1 = np.dot(x,w1) + b1
	#シグモイドで着火
	Z1 = sigmoid(A1)

	
	A2 =np.dot(Z1,w2) + b2
	Z2 = sigmoid(A2)

	
	A3 =np.dot(Z2,w3) + b3
	Y = soft_max(A3) #ここには出力用の着火関数が入る

	return Y

x = np.array([1.0,0.5])
net = init_net()
A = forword(net,x)
	
print(A) 
