clear;

% 加载之前保存的编码信息
load('pic\encode.mat');

% 读取嵌入水印后的图像
FAO = double(imread('pic\watermarked_image.png'))/255;

% 获取原来图像的傅里叶变换FA
im = double(imread('pic\chuying.png'))/255;
FA = fft2(im);

% 获取水印图像的大小marksize
mark = double(imread('pic\lena.png'))/255;
marksize = size(mark);

% 对嵌入水印后的图像进行二维傅里叶变换
FA2 = fft2(FAO);

% 计算频域差异
G = (FA2 - FA) / alpha;

% 对频域差异进行重新排序（反操作）
recovered_mark = zeros(marksize);
for i = 1:marksize(1)
    for j = 1:marksize(2)
        recovered_mark(M(i), N(j), :) = G(i, j, :);
    end
end

recovered_mark = abs(recovered_mark);

% 显示提取的水印
figure, imshow(recovered_mark), title('提取的水印');
imwrite(recovered_mark, 'pic\recovered_mark.png');
