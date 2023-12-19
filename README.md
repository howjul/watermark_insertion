# 图像隐水印

1. python环境配置

```shell
pip install numpy
pip install matplotlib
pip install opencv-python
pip install PyWavelets
```

2. 文件说明

   - `Q1.py`为第一题，也就是使用傅里叶变换嵌入水印

   - `Q2.py`为第二题，也就是使用傅里叶变换获取水印
   - `Q3.py`为第三题，是利用小波变换嵌入和获取水印的文件，同时还会自动进行一些攻击测试
   - `Q3_grey.py`为第三题，也是利用小波变换进行水印的嵌入和提取，但是原图像会变成灰度图
   - `pic`文件夹，保存着水印图`./pic/src.png`和原图像`./pic/watermark.png`，同时`Q1/Q3.py`的随机序列也存在这里
   - `matlab文件`文件夹，里面是刚开始的按照PPT的思路进行的实验代码，没什么用

   