import numpy as np 
import matplotlib.pyplot as plt
import n_net2  
import n_net
#n_net2では勾配法で行っていた数式の訂正を逆伝播法で試すよ
"""
計算グラフとは？　
逆伝播についての説明のため用いる
使うのはノードと値
値を受け取って　－　ノードで計算
ノードは一つの計算しか行わない
図

"""
"""
各ノードのルール
1,返り値は一つに統一
2,forwordの引数は一つにする
3,重みやバイアスは初期設定ですます

"""

class Mul_layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forword(self,x,y):
        self.x = x
        self.y = y
        out = x * y

        return out
    def backword(self,dout):
        dx = dout * self.y
        dy = dout * self.x
        
        return dx,dy

class Add_layer:
    def __init__(self):
        self.x = None
        self.y = None

    def forword(self,x,y):
        self.x = x
        self.y = y

        out = x + y

        return out

    def backword(self,dout):
        dx = dout * 1
        dy = dout * 1

        return dx  

class Relu:
    def __init__(self):
        self.mask = None
    
    def forword(self,x):
        self.mask = (x <= 0)
        out = x
        out[self.mask] = 0

        return out

    def backword(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class sigmoid:
    def __init__(self):
        self.x = None
    
    def forword(self,x): 
        out = 1/(1 + np.exp(-x))
        self.x = out

        return out

    def backword(self,dout):
        dx = dout * (1.0 - self.x) * self.x

        return dx

class Affine:
    def __init__(self,w,b):
        self.x = None
        self.w = w
        self.b = b

    def forword(self,X):
        self.x  =  X
        

        out  =  np.dot(X,self.w) + self.b
        return out

    def backword(self,dout):
        dx = 0
        dw = 0
        db = 0

        db = np.sum(dout,axis = 0) 

        dx = np.dot(dout,self.w.T)
        dw = np.dot(self.x.T,dout)

        return dx

class softmaxwithLoss:
    def __init__(self):
        self.x = None
        self.t = None
        self.y=None

    def forword(self,x,t):
        self.x = x
        self.t = t
        self.y = n_net.soft_max(x)
        self.loss = n_net2.cross_entropy_error(self.y,t)

        return self.y,self.loss

    def backword(self, dout= 1):
        batch_size  = self.t.shape[0]
        dx = (self.y - self.t) /batch_size

        return dx
        



"""
ADDlayer と　mul_layerの確認
apple = 100
apple_num = 2
orenge = 150
orenge_num = 3
tax = 1.1

mul_apple_l = Mul_layer()
mul_orange_l  = Mul_layer()
mul_tax_l = Mul_layer()
add_appll_l =  Add_layer()



#forword
app_price = mul_apple_l.forword(apple,apple_num)
orange_price = mul_orange_l.forword(orenge,orenge_num)
price = add_appll_l.forword(orange_price,app_price)
tex_price = mul_tax_l.forword(price,tax)

print(tex_price)
#backword
dprice  = 1
dprice,dtax = mul_tax_l.backword(dprice)
dorange_price,dapple_price  = add_appll_l.backword(dprice)
dapple,dapple_num = mul_apple_l.backword(dapple_price)
dorange,dorange_num = mul_orange_l.backword(dorange_price)

print(dprice,dtax)
print(dapple_price,dorange_price)
print(dapple,dapple_num)
print(dorange,dorange_num)
"""

"""
逆伝播法とは？

"""