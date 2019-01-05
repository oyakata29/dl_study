"""
初期の重みを最適にするには？
初期の重みは各層でｗべつにしないと全てに同じ最適化が施されてしまう

着火関数が線形　（sigmod）
、Xavierの初期重み
着火関数が非線形  (Relu)
heの初期重み

重みの初期値を考えずに済む方法
Batch Normalizationを使うと良い
なにするの？
各層の出力を分散を１、平均を０になるように計算し直した後、
着火関数に渡す。
利点
計算が早く済む
与吉にそれほど依存しない
過学習を促成する

過学習　データセットの数が少ない等の問題によりそのデータセットにのみ特化した
数式になってしまう。

解決方法
過学習は、重みが一定以上の場合発生することが多い。
重みの際計算をして一定以下に抑える手法　　、Weight decay

"""
class Dropout:
    def __init__(self,dropout_rate = 0.5):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forword(self,x,train_frag = true):
        if train_frag:
            self.mask = np.random.rand(*x.sharp) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)

    def bacword(self,self,dout):
        return dout * self.mask


"""
ハイパーパラメータの設定
ハイパーパラメータとは？
バッチ数や、各層のニューロン数、学習係数、など、、、

どうやって決めるのか。
テストデータを用いるのはOUT ー＞　テストデータに対する過学習になってしまう。
なので、ハイパーパラメータ用にデータを抽出　＝ 検証データ　

"""