import sys ,os
sys.path.append(os.pardir)
import numpy as np
from n_net2 import num_grad
#from n_net import *
from backward_propagation import *
from collections import OrderedDict

class TowLayerNet:


    def __init__(self,input_size,hidden_size,output_size,weight_init_std = 0.01):
        #各レイヤの重さをらんだむに、バイアスを０設定
        self.params = {}
        self.params["w1"] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["b2"] = np.zeros(output_size)

        #レイヤを作成
        self.layer = OrderedDict()
        self.layer["Affine1"] = Affine(self.params["w1"],self.params["b1"])
        self.layer["Relu1"] = Relu()
        self.layer["Affine2"] = Affine(self.params["w2"],self.params["b2"])

        #最後はソフトマックスで丸める
        self.lastlayer = softmaxwithLoss()

    # NNを計算する
    def predict(self,x):
        for layer in self.layer.values():
            x = layer.forword(x)
        return x

    #教師データｔと入力データｘの計算結果のロスを求める
    def loss(self,x,t):
        y = self.predict(x)
        return self.lastlayer.forword(y,t)

    #教師データｔと入力データｘの認識精度を求める
    def accuracy(self,x,t):
        y = self.predict(x)
        #帰ってきた答えからそれぞれ最大の要素（回答）はどれかを調べる
        y = np.argmax(y,axis=1)
        #教師データが一次元配列ではなかった場合ｙと同じような形に直す。
        #正当はどのデータか？
        if t.ndim != 1:
            t = np.argmax(t,axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy
    #逆伝播法を用いて勾配を求めます
    def gradient(self,x,t):
        self.loss(x,t)
        
        dout = 1
        dout = self.lastlayer.backword(dout)

        layers = list(self.layer.values)
        layers.reverse()
        for layer in layers:
            dout = layer.backword(dout)

        grid = {}
        grid["w1"] = self.layer["Affine1"].dw
        grid["w2"] = self.layer["Affine2"].dw
        grid["b1"] = self.layer["Affine1"].db
        grid["b2"] = self.layer["Affine2"].b2

        return grid      

    #数値微分を用いて勾配と求める
    def num_gradient(self,x,t):
        loss_w = lambda W: self.loss(x,t)

        grads = {}
        grads["w1"] = num_grad(loss_w,self.params["w1"])
        grads["w2"] = num_grad(loss_w,self.params["w2"])
        grads["b1"] = num_grad(loss_w,self.params["b1"])
        grads["b2"] = num_grad(loss_w,self.params["b2"])

        return grads