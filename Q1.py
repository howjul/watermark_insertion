import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负数的负号显示问题

# 读取水印和原图像
im = plt.imread('pic/src.png')
mark = plt.imread('pic/watermark.png')

# 将mark转化为二值图
mark[:,:,0] = mark[:,:,0]*0.3
mark[:,:,1] = mark[:,:,1]*0.59
mark[:,:,2] = mark[:,:,2]*0.11
mark[:,:,3] = mark[:,:,3]*0
mark = np.sum(mark, axis=2)
mark[mark > 0.5] = 1.0
mark[mark <= 0.5] = 0.0
mark = np.expand_dims(mark, axis=2)
mark = np.concatenate((mark, mark, mark), axis=2)
plt.imshow(mark)
plt.title('原始水印')
plt.axis('off') # 不显示坐标轴
plt.show()

# 获取图像尺寸
marksize = mark.shape

# 为 mark 生成填入编码位置
M = np.random.permutation(marksize[0])
N = np.random.permutation(marksize[1])

# 调整水印强度
alpha = 10

np.savez('pic/encode.npz', M=M, N=N, alpha=alpha)  # 保存

# 将 mark 填入 mark_
mark_ = np.zeros_like(im)
for i in range(marksize[0]):
    for j in range(marksize[1]):
        mark_[i, j, 0:3] = mark[M[i], N[j], 0:3]

# 二维傅里叶变换
FA = np.fft.fft2(im)

# 频域叠加
FB = FA + alpha * mark_

# 二维逆傅里叶变换
FAO = np.fft.ifft2(FB)

# 为了可视化，将复数取幅值
FAO2 = np.abs(FAO)  # 取模
FAO2[FAO2 > 1] = 1   # 防止溢出

# 显示图像
plt.imshow(im)
plt.title('原始图像')
plt.axis('off') # 不显示坐标轴
plt.show()

plt.imshow(FAO2)
plt.title('嵌入水印后的图像')
plt.axis('off') # 不显示坐标轴
plt.show()

plt.imsave('pic/watermarked_image.png', FAO2)


# # 提取水印
# # 对嵌入水印后的图像进行二维傅里叶变换
# FA2 = np.fft.fft2(FAO)

# # 计算频域差异
# G = (FA2 - FA) / alpha
# G = np.abs(G)

# # 对频域差异进行重新排序（反操作）
# recovered_mark = np.zeros((mark.shape[0], mark.shape[1], 3))
# for i in range(marksize[0]):
#     for j in range(marksize[1]):
#         recovered_mark[M[i], N[j], 0:3] = G[i, j, 0:3]

# recovered_mark = np.abs(recovered_mark)
# recovered_mark[recovered_mark > 1] = 1.0

# # 显示提取的水印
# plt.imshow(recovered_mark)
# plt.title('提取的水印')
# plt.axis('off') # 不显示坐标轴
# plt.show()

# # 保存提取的水印
# plt.imsave('pic/recovered_mark.jpg', recovered_mark)