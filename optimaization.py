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