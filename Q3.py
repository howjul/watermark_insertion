import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
import pywt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

########################以下为函数定义########################
# 将水印嵌入到图像中并返回嵌入水印后的图像
def set_dwt_watermark(I, W):
    # 取出R通道
    R_channel = I[:, :, 0]
    G_channel = I[:, :, 1]
    B_channel = I[:, :, 2]
    I_before = I.copy()
    I = R_channel

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
    
    Rw_channel = Iw.copy()
    Rw_channel = Rw_channel.astype(np.uint8) # 转化成uint8类型
    Iw = cv.merge([B_channel, G_channel, Rw_channel]) # 合并通道

    # 显示嵌入水印后的图像
    plt.figure('嵌入水印的图像', figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(I_before)
    plt.title('原始图像')
    plt.subplot(122)
    plt.imshow(cv.cvtColor(Iw, cv.COLOR_BGR2RGB))
    plt.title('添加水印')
    plt.show()

    np.savez('pic/encode.npz', M=M, N=N, idx=idx)  # 保存

    return Iw

# 从图像中提取水印并返回提取水印后的结果
def get_dwt_watermark(Iw, W, flag):
    Rw_channel = Iw[:, :, 2]
    Iw = Rw_channel
    
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
    if flag:
      plt.figure('数字水印提取结果', figsize=(12, 8))
      plt.subplot(121)
      plt.imshow(W, cmap='gray')
      plt.title('原始水印')
      plt.subplot(122)
      plt.imshow(Wg, cmap='gray')
      plt.title('提取水印')
      plt.show()

    return Wg

########################以下为主程序########################
# 读取图像
I = cv.imread('pic/src.png')[:704, :1264]
I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

# 读取水印图像并转化成二值图
W = cv.imread('pic/watermark.png', cv.IMREAD_GRAYSCALE)
W = cv.threshold(W, 128, 255, cv.THRESH_BINARY)[1]

# 显示原始图像和水印图像
plt.figure('载体图像和水印图像', figsize=(12, 8))
plt.subplot(121)
plt.imshow(I)
plt.title('载体图像')
plt.subplot(122)
plt.imshow(W, cmap='gray')
plt.title('水印图像')
plt.show()

# 对R通道添加水印并保存结果图像
Iw = set_dwt_watermark(I, W)
cv.imwrite('pic/watermarked_image.jpg', Iw)

# 读取水印图像并取出R通道
Iw = cv.imread('pic/watermarked_image.jpg')

# 提取水印并显示结果
Wg = get_dwt_watermark(Iw, W, 1)


########################以下为攻击试验########################
# 涂抹攻击试验
Ia = Iw.copy()
Ia[50:400, 50:400, 0:3] = np.random.randn(350, 350, 3)
Wg = get_dwt_watermark(Ia, W, 0)

plt.figure('涂抹攻击试验', figsize=(12, 8))
    
plt.subplot(221)
plt.imshow(cv.cvtColor(Iw, cv.COLOR_BGR2RGB))
plt.title('嵌入水印图像')

plt.subplot(222)
plt.imshow(cv.cvtColor(Ia, cv.COLOR_BGR2RGB))
plt.title('遭受涂抹攻击的图像')

plt.subplot(223)
plt.imshow(W, cmap='gray')
plt.title('原始水印图像')

plt.subplot(224)
plt.imshow(Wg, cmap='gray')
plt.title('提取水印')

plt.show()

# 缩放攻击试验
Ia = Iw.copy()
Ia = cv.resize(Iw, None, fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)
Ia = cv.resize(Ia, (Iw.shape[1], Iw.shape[0]), interpolation=cv.INTER_NEAREST)
Wg = get_dwt_watermark(Ia, W, 0)

plt.figure('缩放攻击试验', figsize=(12, 8))
    
plt.subplot(221)
plt.imshow(cv.cvtColor(Iw, cv.COLOR_BGR2RGB))
plt.title('嵌入水印图像')

plt.subplot(222)
plt.imshow(cv.cvtColor(Ia, cv.COLOR_BGR2RGB))
plt.title('遭受缩放攻击的图像')

plt.subplot(223)
plt.imshow(W, cmap='gray')
plt.title('原始水印图像')

plt.subplot(224)
plt.imshow(Wg, cmap='gray')
plt.title('提取水印')

plt.show()

# 噪声攻击试验
Ia = np.copy(Iw)
h,w = Iw.shape[:2]  #获取图像的宽高信息
nums = 5000
rows = np.random.randint(0, h, (5000), dtype = np.int32)
cols = np.random.randint(0, w,(5000),dtype = np.int32)
for i in range(nums):
    if i%2 == 1:
        Ia[rows[i],cols[i]] = (255,255,255)
    else:
        Ia[rows[i],cols[i]] = (0,0,0)
gnoise = np.zeros(Iw.shape,Iw.dtype)
m = (2,2,2)  #噪声均值
s = (10,10,10) #噪声方差
cv.randn(gnoise,m,s) #产生高斯噪声
Ia = cv.add(Ia,gnoise) #将高斯噪声图像加到原图上去
Wg = get_dwt_watermark(Ia, W, 0)

plt.figure('高斯噪声攻击试验', figsize=(12, 8))
    
plt.subplot(221)
plt.imshow(cv.cvtColor(Iw, cv.COLOR_BGR2RGB))
plt.title('嵌入水印图像')

plt.subplot(222)
plt.imshow(cv.cvtColor(Ia, cv.COLOR_BGR2RGB))
plt.title('遭受高斯攻击的图像')

plt.subplot(223)
plt.imshow(W, cmap='gray')
plt.title('原始水印图像')

plt.subplot(224)
plt.imshow(Wg, cmap='gray')
plt.title('提取水印')

plt.show()

# 裁剪攻击试验
Ia = Iw[:Iw.shape[0]-2, :Iw.shape[1], :]
Ia = cv.resize(Ia, (Iw.shape[1], Iw.shape[0]))
Wg = get_dwt_watermark(Ia, W, 0)

plt.figure('裁剪攻击试验', figsize=(12, 8))
    
plt.subplot(221)
plt.imshow(cv.cvtColor(Iw, cv.COLOR_BGR2RGB))
plt.title('嵌入水印图像')

plt.subplot(222)
plt.imshow(cv.cvtColor(Ia, cv.COLOR_BGR2RGB))
plt.title('遭受裁剪攻击的图像')

plt.subplot(223)
plt.imshow(W, cmap='gray')
plt.title('原始水印图像')

plt.subplot(224)
plt.imshow(Wg, cmap='gray')
plt.title('提取水印')

plt.show()

# 均值滤波攻击试验
Ia = cv.medianBlur(Iw, 5)  # 均值滤波
Wg = get_dwt_watermark(Ia, W, 0)

plt.figure('均值滤波攻击试验', figsize=(12, 8))
    
plt.subplot(221)
plt.imshow(cv.cvtColor(Iw, cv.COLOR_BGR2RGB))
plt.title('嵌入水印图像')

plt.subplot(222)
plt.imshow(cv.cvtColor(Ia, cv.COLOR_BGR2RGB))
plt.title('遭受均值滤波攻击的图像')

plt.subplot(223)
plt.imshow(W, cmap='gray')
plt.title('原始水印图像')

plt.subplot(224)
plt.imshow(Wg, cmap='gray')
plt.title('提取水印')

plt.show()