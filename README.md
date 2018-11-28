# dl_study
deep learningの勉強用のリポジトリ

## persep.py

 パーセプトロンの学習

* b + w1*x + w2*y <= 0   出力　0  
　　以下で０  
* b + w1*x + w2*y >  0   出力　1  
　　以上で１

Wは重み
bはバイアス、下駄とか呼ばれる

パーセプトロンは他州化することで柔軟な対応が可能
,コンピュータも作れる！(NANDができる限りなんでも作れる)  
しかし、出力の設定は人間の手で行う（手間)

## n_net.py

 ニューラルネットワークの学習
 この項では人間が作ったネットワーク型の式をいかに記述するかという話

基本式　　b + w1*x + w2*y  ｘｙが入力、ｂがバイアス　W１，２が重み

このあとの計算により値を変化
* 活性化関数 入力された値から判定してこのあと使うか（活性化？）決める
* ステップ関数　前回のパーセプロンと同じもの
* シグモイド関数 ステップ関数を１〜０の間で滑らかに
* ReLO関数 ０以上なら値をそのまま返す

複数の入力を複数の層に渡すため多次元行列の値を収めると楽　　
	注意  [a,b] * [b,c] =  [a,c]   
の形じゃないと計算できない

3層ニューラルネットワークの実装


## n_net2.py
 ニューラルネットワークの学習２

 では、そのネットワーク型の式をどうすればプログラムが勝手に重み等を考えて作ってくれるか という話


教師あり学習の場合  
正答と式が出した回答を見比べ（ずれがあったら）正当に近づくようにしてやればよい。  
　↓  
損失関数  
* 二乗和誤差法  
	出力とone-hotの正答との差異を求めるそれを２乗し足し合わせることにより誤差を求める
* 交差エントロピー法  
	logx がｘ＝１で０、ｘ＝０で-５（大体）になることを利用する。
	出力が正答（１）より小さければ小さいほど値が大きくなっていく
one-hotとは？  
　正当を１それ以外を０としたリスト  
　例　正解が３　－＞　{0,0,0,1,0,0,0} 
　みたいな感じ

なぜ損失関数なのか、正答率ではないのか？
正答率　＝　００％のように出力されるがこれはステップであるため細かい計算に向かない

勾配法  
ある関数と変数に入れるための値を渡すと勾配という数値を返してくれる。
勾配は、その変数がその関数の最小の値を返す回答から変数に入れた値がどれだけ離れているかを求めてくれる。  
この勾配を使えば、損失関数が最適の値（重み）が見つかる！！  
勾配降下法と勾配上昇法がある  
※　実際の複雑な問題では勾配の示す先が最小値でなかったり、最小値＝最優値でなかったりします。



そのためには？  
式を逆順していき、どの式がどれくらい答えに影響を与えているか

***
参考文献

ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装
オライリー・ジャパン　２０１６　斎藤 康毅 

