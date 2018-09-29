#ニューラルネットワークの学習

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






