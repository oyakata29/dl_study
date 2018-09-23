"""
パーセプトロンの学習
b + w1*x + w2*y <= 0   出力　0    
　　以下で０
b + w1*x + w2*y >  0   出力　1 
　　以上で１
　bは下駄とか呼ばれる

パーセプトロンは他州化することで柔軟な対応が可能
	コンピュータも作れる！(NANDができる限りなんでも作れる)
しかし、出力の設定は人間の手で行う（手間)

"""

import numpy as np

def myand(x,y):
	i = np.array([x,y])
	w = np.array([0.5,0.5])
	b = -0.7
	tmp = np.sum(w*i) + b
	if tmp > 0:
		return 1
	elif tmp <= 0:
		return 0

def mynand(x,y):
        i = np.array([x,y])
        w = np.array([-0.5,-0.5])
        b = 0.7
        tmp = np.sum(w*i) + b
        if tmp > 0:
                return 1
        elif tmp <= 0:
                return 0

def myor(x,y):
	i = np.array([x,y])
	w = np.array([0.5,0.5])
	b = -0.4
	tmp = np.sum(w * i) + b
	if tmp > 0:
		return 1
	elif tmp <= 0:
		return 0	
#非線形の実現にはパーセプトロンの多重化で対応できる。
def myxor(x,y):
	i1 = mynand(x,y)
	i2 = myor(x,y)
	return	myand(i1,i2)
	

x = int(input("x = "))
y = int(input("y = "))
print(myxor(x,y))



