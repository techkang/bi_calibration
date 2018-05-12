# 普通相机图像测距
代码运行环境： `Python 3.6.3`  
下载本仓库并安装相关包：

    git clone https://github.com/techkang/postgraduate
    cd postgraduate
    pip3 install -r requirements.txt
## 1.文件目录
├── result（存放相关结果）     
│   └── ...  
├── code  （代码）  
│   └── calibration.py  （图像标定）  
├── left  （测试图片）  
│   ├── left01.jpg  
│   ├── ...  
│   └── left14.jpg  
├── README.md  
├── right  （测试图片）  
│   ├── right01.jpg  
│   ├── ...  
│   └── right14.jpg  
└── requirements.txt

## 2.单目相机标定
### 运行代码

	cd code
	python3 calibration.py [file]
其中，file 是可选参数，默认为 01,表示对 left 文件夹中 left01.jpg 图像进行标定。可以指定其他值对left文件夹中的图片进行标定（注意0不能省略）。
### 运行结果
以 left01.jpg 为例，原图和修正后的图像如下：
![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/left/left01.jpg)
![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/result/calibresult/left01.png)
## 3.双目相机标定
### 运行代码

	cd code
    python3 stereo_calibration.py [file]
其中，file 是可选参数，默认为 01,表示对 left 文件夹中 left01.jpg 图像和 right 文件夹中的 right01.jpg 图像进行标定。可以指定其他值对left和right文件夹中的图片进行标定（注意0不能省略）。
### 运行结果
运行程序，输出对指定图片双目标定的结果，同时将结果保存在`result/stereo_calibresult/`文件夹下。本节代码未能按照预期的结果运行，输出照片发生了明显的旋转。以 left01.jpg 和 right01.jpg 为例，输出结果如图所示：
![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/result/stereo_calibresult/calibrated.png)

经过分析，发现 OpenCV 一般不能很好的标定双目图像，故使用 MATLAB 先标定图像，例如 left01 和 right01 的 R (旋转矩阵)和 T (平移矩阵)如下：

 |R ||T|
 :--:|:--:|:--:|:--:|
 1|-0.0032|-0.0057|-83.0973|
 0.0033|0.9999|0.0096|1.0605|
 0.0057|-0.0097|0.9999|0.0392|
 
 在程序中修改代码，手动指定 R 和 T 的值，运行程序，输出校正后的图像如下：
 ![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/result/stereo_calibresult/matlab-calib.png)
 
