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


#二乗誤差出力
def meam_squared_error(y,t):
	return 0.5 * np.sum((y - t) ** 2)

#交差エントロピー誤差 if文でミニバッチ処理に対応
def cross_entropy_error(y,t):
	#一次配列（バッチが一つ）なら以下の処理
	if y.ndim == 1:
		t = t.reshape(1,t.size)
		y = y.reshape(1,y.size)		

	del = 1e - 7
	return -np.sum(t * no.log(y + del))
"""
微分ある等句点の地点の傾きを得る
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
	
#だけど偏微分ス類は変数は日一つしか使えない
#例題　ｘ=３　ｙ　＝４の場合 ｘを偏微分したければ
def func_x(x):
	retunn x[0] ** 2 + 4 ** 2

x' = num_diff(func_x,x):

#偏微分、全部一緒くたにやるには？
#勾配　渡した関数の偏微分をすべてやってベクトルとして返してくれる

##### 勾配の刺す先は関数の値を一番小さくしてくれる場所
def num_grid(f,x):
	h = 1e-4
	grid = np.zeros_like(x)
	
	for idx in range(x.size):
		tmp_val = x[idx]
		
		x[idx] = tmp_val + h
		hx1 = f(x)
		
		x[idx] = tmp_val - h
		hx2 = f(x)
		
		grid(idx) = (hx1 + hx2 )/ (2 * h)
		x[idx] = tmp_val

	return grid
	
	
print()




