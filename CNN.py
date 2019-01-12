"""
今までのNNは前結合型　＝　形は一次元のみしか扱えない
入力の形を無視してしまう

CNNなら形のまま入力出力が行える

word 
特徴マップ
入力特徴マップ
出力特徴マップ

"""

#入力されたデータを単純にFORで回すえお非常に遅い
#入力されたデータを２次元的に展開する関数
def im2col(input_date,filter_h,filter_w,stride = 1,pad = 0):
    n, c, h, w = inout_date.shape
    out_h = (h + 2 * pad - filter_h)//srtide + 1
    out_w = (w + 2 * pad - filter_w)//stride + 1

    img = np.pad(input_date,[(0,0),(0,0),(pad,pad),(pad,pad)],'constant')
    col = np.zeros((n,c,filter_h,filter_w,out_h,out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(n*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class convolution:
    def __init__(self,w,b,strde = 1,pad = 0):
        self.w = w
        self.b = b
        self.stride = srtide
        self.pad = pad

    def forword(self,x):
        fn,c,fh,fw = self.w.shape
        n,c,h,w = x.shape

        out_h = int(1+(h + 2 * self.pad -fh)/self.stride)
        out_w = int(1+(w + 2 * self.pad -fw)/self.stride)

        col = im2col(x,fh,fe,self.stride,self.pad)
        col_w = self.w.reshape(fn,-1).transpose
        out = np.dot(col,col_w) + self.b

        out = out.reshape(n,out_h,out_w,-1).transpose(0,3,2,1)

        return out

    