import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

# 加载之前保存的编码信息
data = np.load('pic/encode.npz')
M = data['M']
N = data['N']
alpha = data['alpha']

# 读取嵌入水印后的图像
FAO = plt.imread('pic/watermarked_image.png')
FAO = FAO[:, :, 0:3]  # 去除 alpha 通道

# 获取原图像的傅里叶变换 FA
im = plt.imread('pic/src.png')
FA = np.fft.fft2(im)

# 获取水印图像的大小 marksize
mark = plt.imread('pic/watermark.png')
marksize = mark.shape

# 对嵌入水印后的图像进行二维傅里叶变换
FA2 = np.fft.fft2(FAO)

# 计算频域差异
G = (FA2 - FA) / alpha
G = np.abs(G)
G[G > 1] = 1.0

# 将频域差异转化为二值图
G = np.mean(G, axis=2)
threshold = 0.45 # np.max(G) * 0.4
G[G > threshold] = 1.0
G[G <= threshold] = 0.0
G = np.expand_dims(G, axis=2)
G = np.concatenate((G, G, G), axis=2)

# 对频域差异进行重新排序（反操作）
recovered_mark = np.zeros((mark.shape[0], mark.shape[1], 3))
for i in range(marksize[0]):
    for j in range(marksize[1]):
        recovered_mark[M[i], N[j], 0:3] = G[i, j, 0:3]

recovered_mark = np.abs(recovered_mark)
recovered_mark[recovered_mark > 1] = 1.0

# 显示提取的水印
plt.imshow(recovered_mark)
plt.title('提取的水印')
plt.axis('off') # 不显示坐标轴
plt.show()

# 保存提取的水印
plt.imsave('pic/recovered_mark.jpg', recovered_mark)
