#ニューラルネットワークの学習
"""
損失関数
ニューラルネットワークで学習した、値が正しいのか確認したい
（教師あり学習の場合）誤差率、正当との差を図ることにより誤差を求めたい

認識制度ではなく誤差を用いる理由
誤差だと微分ができる　=　小さくするにはどうすればいいかわかる
認識後だと微分は大体０

以下そのための関数
"""

import numpy as np
import n_net
#二乗誤差出力
def meam_squared_error(y,t):
	return 0.5 * np.sum((y - t) ** 2)

#交差エントロピー誤差 if文でミニバッチ処理に対応
def cross_entropy_error(y,t):
	#一次配列（バッチが一つ）なら以下の処理
	if y.ndim == 1:
		t = t.reshape(1,t.size)
		y = y.reshape(1,y.size)		

	my_del = 1e-7
	return -np.sum(t * np.log(y + my_del))


"""
微分
ある等句点の地点の傾きを得る
ほんとの式はうまく使えないので
前方誤差でごまかす
"""

def num_diff(f,x):
	h = 1e-4 #小さすぎると計算出来ないのでこんなもん
	return  (f(x+h) - f(x-h))/(2 * h)
	


#適当な式
def fonc(x):
	return 0.001*x**2 + 0.1 * x
	
#編微分
"""
変数を二つ扱った関数で偏微分は普遍的にはできない？
例　ｘ＾２＋ｙ＾２　
関数で表すなら	
"""
def func2(x):
	return np.sum(x**2)
	
#だけど偏微分するには変数は一つしか使えない
#例題　ｘ=３　ｙ　＝４の場合 ｘを偏微分したければ
def func_x(x):
	return x[0] ** 2 + 4 ** 2

def test_func(x):
	
	return x[0]**2 + x[1] **2

#偏微分、全部一緒くたにやるには？
#勾配　渡した関数の偏微分をすべてやってベクトルとして返してくれる

##### 勾配の刺す先は関数の値を一番小さくしてくれる場所
def num_grad(f,x):
	h = 1e-4
	grid = np.zeros_like(x)
	
	for idx in range(x.size):
		tmp_val = x[idx]
		
		x[idx] = tmp_val + h
		hx1 = f(x)
		
		x[idx] = tmp_val - h
		hx2 = f(x)
		
		grid[idx] = (hx1 - hx2 )/ (2 * h)
		
		x[idx] = tmp_val
	
	return grid

##実際に使う勾配法を使ってテスト関数の最小の値のXを求めてみよう
def grid_descent(f, init_x, lr= 0.01,step_num =100):
	x = init_x

	for _ in range(step_num):
		grad = num_grad(f,x)
		x -= lr * grad

	return x

##NNに勾配法を使ってみよう
class Tow_Layer_net:
	def __init__(self,in_size,hide_size,out_size,w_init = 0.01):
		#各層の重みの初期化
		self.para = {}
		self.para["W1"] = w_init* np.random.randn(in_size,hide_size)
		self.para["B1"] = np.zeros(hide_size)
		self.para["W2"] = w_init* np.random.randn(hide_size,out_size)
		self.para["B2"] = np.zeros(out_size)

	def prodict(self,x):
		W1,W2 = self.para["W1"],self.para["W2"]
		B1,B2 = self.para["B1"],self.para["B2"]

		a1 = np.dot(x,W1) + B1
		z1 = n_net.sigmoid(a1)
		a2 = np.dot(z1,W2) + B2
		y = n_net.soft_max(a2)

		return y

		#損失関数
	def loss(self,x,t):
		y = self.prodict(x)
		return cross_entropy_error(y,t)

		#認識精度
	def acctiary(self,x,t):
		y = self.prodict(x)
		y = np.argmax(y,axis=1)
		t = np.argmax(t,axis=1)

		acctiary = np.sum(t==y) / float(x.shape[0])
		return acctiary
	#勾配法
	def num_grad(self,x,t):
		loss_w = lambda W :self.loss(x,t)

		grad = {}
		grad["W1"] = num_grad(loss_w,self.para["W1"])
		grad["W2"] = num_grad(loss_w,self.para["W2"])
		grad["B1"] = num_grad(loss_w,self.para["B1"])
		grad["B2"] = num_grad(loss_w,self.para["B2"])

		return grad






