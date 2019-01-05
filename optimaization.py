"""
NNの目標は判別を行うための一番最適な値を求めることである。（最適化）
    ーーー今までのソースコードで言う、ｗ１やｂ１を最適にすること

この最適化を行う際、どのような最適化の方法があるのか？
"""

import numpy as np

#SGD　今までのコードで使ったもの 欠点 三次元の値で片方が極端に小さいと、最適化に時間がかかる
class SGD:
    def __init__(self,lr = 0.01):
        self.lr = lr
    
    def updete(self,parms,grads):
        for kye in parms.kyes():
            parms[kye] -= self.lr * grads[kye]

#momentam　運動量　最初はおおざっぱに鵜が課していき計算が進むにつれ動きを小さくしていく
class momentum:
    def __init__(self,lr = 0.01,moment = 0.9):
        self.lr = lr
        self.momentum = moment
        self.v = None

    def update(self,parms,grads):
        if self.v is None:
            self.v = {}
            for kye,value in parms.items():
                self.v[kye] = np.zeros_like(value)

            for kye in parms.kye():
                self.v[kye] = self.momentum
                parms[kye] += self.v[key]

#学習率を調整する鳳凰、 最初のうちは学習率を大きく進むにつれて小さくしていく計算法
class adagrad:
    def __init__(self,rl = 0.01):
        self.rl = 0.01
        self.h = None

    def update(self,parms,grads):
        if self.h is None:
            self.h = {}
            for key,value in parms.items():
                self.h[key] = np.zeros_like(value)

        for key in parms.key:
            self.h[key] += parms[key] * parms[key]
            parms[key] -= self.rl * grads[key] / (np.sqrt(self.h[key] + 1e -7))


