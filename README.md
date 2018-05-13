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
│   ├── calibration.py  (单目标定、校正）
│   ├── draw.py  （推导视差公式时绘图的代码）
│   ├── main.py  （单、双目标定、校正及测距）
│   └── stereo_calibration.py  （双目标定、校正）
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

 ||R ||T|
 |:--:|:--:|:--:|:--:|
 |1|-0.0032|-0.0057|-83.0973|
 |0.0033|0.9999|0.0096|1.0605|
 |0.0057|-0.0097|0.9999|0.0392|
 | | | | | |
 
 在程序中修改代码，手动指定 R 和 T 的值，运行程序，输出校正后的图像如下：
 ![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/result/stereo_calibresult/matlab-calib.png)
 
## 4.视差测距
本函数可以同时实现单目标定、双目标定和校正，以及测距。
### 运行代码

	cd code
    python3 mian.py [--key=value]
其中，`[--key=value]`是可选参数，可选择的值、含义及其默认值如下：

	sample = '01'  # 测试图片
    disp_calib = False  # 是否展示单目校正结果
    stereo_calib = True  # 是否进行双目校正
    disp_stereo_calib = False  # 是否展示双目校正结果
    disparity = True  # 是否利用视差估算距离
    R = dict({'01': np.array([[1, -0.0032, -0.005], [0.0033, 0.9999, 0.0096],
                              [0.0057, -0.0097, 0.9999]])})  # 由 MATLAB 标定的旋转矩阵
    T = dict({'01': np.array([-83.0973, 1.0605, 0.0392])})  # 由 MATLAB 标定的平移矩阵
    matlab = True  # 在双目校正时是否使用 matlab 标定的值
    num = 3  # StereoSGBM_create 函数参数：最小可能的差异值
    blockSize = 5  # StereoSGBM_create 函数参数：匹配的块大小。

例如，输入命令`python3 main.py --disp_calib=True`，可以在测距的同时，还会输出单目测距的结果。如果保持默认值，则会对 left01 和 right01 图像进行测距，不输出单目校正和双目校正的结果，使用由 MATLAB 标定的旋转矩阵和平移矩阵。
### 运行结果
按照默认参数，输出结果如图：

 ![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/result/disparity/ranging.png)
 
 点击左图，可以在命令行中输出该点的深度，然而，无论是从输出结果，还是右边的视差图，都离预期结果相差较远。点击左图时发现，经常有些点为负值。右图视差图中，标定板的视差应该相近，为一个平滑的渐变颜色，然而视差图没有体现出这一点。