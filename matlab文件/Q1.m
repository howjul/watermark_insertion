clear;

% 读取水印和原图像
im = double(imread('pic\chuying.png'))/255;
mark = double(imread('pic\lena.png'))/255;

% 获取图像尺寸
marksize = size(mark);

% 为 mark 生成填入编码位置
M = randperm(marksize(1));
N = randperm(marksize(2));

% 调整水印强度
alpha = 100; 

save('pic\encode.mat', 'M', 'N', "alpha"); % 保存

% 将 mark 填入 mark_
mark_ = zeros(size(im));
for i = 1:marksize(1)
    for j = 1:marksize(2)
        mark_(i, j, 1:3) = mark(M(i), N(j), 1:3);
    end
end

% 二维傅里叶变换
FA = fft2(im);

% 频域叠加
FB = FA + alpha * double(mark_);

% 二维逆傅里叶变换
FAO = ifft2(FB);

% 为了可视化，将复数取幅值
FAO = abs(FAO); % 取模

% 显示图像
% figure, imshow(mark), title('水印图像');
figure
subplot(2,1,1), imshow(im), title('原始图像');
subplot(2,1,2), imshow(FAO), title('嵌入水印后的图像');
imwrite(FAO, 'pic\watermarked_image.png');


% % 对嵌入水印后的图像进行二维傅里叶变换
% FA2 = fft2(FAO);

% % 计算频域差异
% G = (FA2 - FA) / alpha;

% % 对频域差异进行重新排序（反操作）
% recovered_mark = zeros(marksize);
% for i = 1:marksize(1)
%    for j = 1:marksize(2)
%        recovered_mark(M(i), N(j), :) = G(i, j, :);
%    end
% end

% recovered_mark = abs(recovered_mark);

% % 显示提取的水印
% figure, imshow(recovered_mark), title('提取的水印');
% imwrite(recovered_mark, 'pic\recovered_mark.png');


