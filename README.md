# 普通相机图像测距代码
运行本仓库代码需要先安装相关包：

    git clone https://github.com/techkang/postgraduate
    cd postgraduate
    mkdir calibresult
    pip3 install -r requirements.txt
## 文件目录
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

## 相机标定
### 运行代码

	cd code
	python3 calibration.py [count]
其中，count 是可选参数，默认为 01,表示对 left 文件夹中 left01.jpg 图像进行标定。
### 运行结果
以 left01.jpg 为例，原图和修正后的图像如下：
![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/left/left01.png)
![Alt](https://raw.githubusercontent.com/techkang/postgraduate/master/result/calibresult/left01.png)