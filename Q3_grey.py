import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import pywt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

# 将水印嵌入到图像中并返回嵌入水印后的图像
def set_dwt_watermark(I, W):
    I = I.astype(float)
    W = W.astype(bool)
    mI, nI = I.shape
    mW, nW = W.shape

    # 对输入的原图像进行两级Haar小波变换
    coeffs = pywt.dwt2(I, 'haar')
    ca2, (ch2, cv2, cd2) = pywt.dwt2(coeffs[0], 'haar') # 对低频部分(LL)继续分解
    
    # 对水印进行置乱
    Wa = W.copy()
    M = np.random.permutation(mW)
    N = np.random.permutation(nW)
    for i in range(mW):
        for j in range(nW):
            Wa[i, j] = W[M[i], N[j]]
    
    # 将置乱后的水印嵌入到小波系数中
    ca2w = ca2.copy()
    idx = np.random.permutation(ca2.size)[:Wa.size] # 生成随机排列的索引
    for i in range(Wa.size):
        c = ca2.flat[idx[i]]
        z = c % nW
        if Wa.flat[i]:
            if z < nW/4:
                f = c - nW/4 - z
            else:
                f = c + nW*3/4 - z
        else:
            if z < nW*3/4:
                f = c + nW/4 - z
            else:
                f = c + nW*5/4 - z
        ca2w.flat[idx[i]] = f
    
    # 执行两级Haar小波反变换
    ca1w = pywt.idwt2((ca2w, (ch2, cv2, cd2)), 'haar')
    Iw = pywt.idwt2((ca1w, (coeffs[1][0], coeffs[1][1], coeffs[1][2])), 'haar')

    # 截取图像使其与原始图像大小相等
    Iw = Iw[:mI, :nI]
    
    # 显示嵌入水印后的图像
    plt.figure('嵌入水印的图像')
    plt.subplot(121)
    plt.imshow(I, cmap='gray')
    plt.title('原始图像')
    plt.subplot(122)
    plt.imshow(Iw, cmap='gray')
    plt.title('添加水印')
    plt.show()

    np.savez('pic/encode.npz', M=M, N=N, idx=idx)  # 保存

    return Iw

# 从图像中提取水印并返回提取水印后的结果
def get_dwt_watermark(Iw, W):
    mW, nW = W.shape
    Iw = Iw.astype(float)
    W = W.astype(bool)

    data = np.load('pic/encode.npz')
    idx = data['idx']
    M = data['M']
    N = data['N']

    # 执行两级Haar小波变换
    coeffs = pywt.dwt2(Iw, 'haar')
    ca2w, (ch2w, cv2w, cd2w) = pywt.dwt2(coeffs[0], 'haar')
    
    # 从小波系数中提取水印
    Wa = W.copy()    
    for i in range(Wa.size):
        c = ca2w.flat[idx[i]]
        z = c % nW
        if z < nW/2:
            Wa.flat[i] = 0
        else:
            Wa.flat[i] = 1
    
    # 对提取的水印进行反置乱
    Wg = Wa.copy()
    for i in range(mW):
        for j in range(nW):
            Wg[M[i], N[j]] = Wa[i, j]
    
    # 显示提取的水印
    plt.figure('数字水印提取结果')
    plt.subplot(121)
    plt.imshow(W, cmap='gray')
    plt.title('原始水印')
    plt.subplot(122)
    plt.imshow(Wg, cmap='gray')
    plt.title('提取水印')
    plt.show()

    return Wg

# 读取图像
I = cv.imread('pic/src.png', cv.IMREAD_GRAYSCALE)[:704, :1264]
W = cv.imread('pic/watermark.png', cv.IMREAD_GRAYSCALE)
W = cv.threshold(W, 128, 255, cv.THRESH_BINARY)[1]

# 显示原始图像和水印图像
plt.figure('载体图像')
plt.imshow(I, cmap='gray')
plt.title('载体图像')
plt.show()

plt.figure('水印图像')
plt.imshow(W, cmap='gray')
plt.title('水印图像')
plt.show()

# 添加水印并保存结果图像
Iw = set_dwt_watermark(I, W)
cv.imwrite('pic/watermarked_image.jpg', Iw)

# 读取水印图像
Iw = cv.imread('pic/watermarked_image.jpg', cv.IMREAD_GRAYSCALE)

# 提取水印并显示结果
Wg = get_dwt_watermark(Iw, W)

# Ia = Iw.copy()
# Ia = cv.resize(Iw, None, fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
# Ia = cv.resize(Ia, (Iw.shape[1], Iw.shape[0]), interpolation=cv.INTER_NEAREST)
# Wg = get_dwt_watermark(Ia, W)
