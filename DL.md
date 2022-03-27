# 机器学习

## 概念

### 什么是机器学习？

通过计算机，得到一个解决某个复杂问题的函数，就是机器学习。比如图像识别，要得到一个函数，它接收一张图片，输出图片中的对象是什么。

### 根据函数的不同类型对任务进行分类

1. regression: 输出的是一个数值（预测PM2.5）
2. classification: 输出的是一个类别（垃圾邮件分类）
3. structured Learning: 计算机要创造有结构的东西（画图、写作）

### 机器学习的三个步骤？

以下三个步骤叫做训练：

1. 写出带有未知参数的函数（这个函数也叫Model，如y=b+w*x1，其中x1叫feature）（在训练之后可能要修改Model）
2. 定义损失函数（这是一个关于未知参数的函数，如L(b,w)）（损失函数可能会选择MAE、MSE、cross-entropy（y是概率时））（根据不同参数画出来的Loss的等高线图叫error surface）（真实的$\hat{y}$叫做Label）
3. 优化，找到让Loss最小的参数（优化方法有Gradient Descent等，自己要设置的东西叫hyperparameter）

### 改进Model

#### 1.怎么使Model更有弹性？

可能Model不是线性的，那我们可以用多段的“z字型”的蓝色的函数（其实叫Hard Sigmoid）加上一个常数来得到一个非线性的Model
![model](/image_host/0.png)
那么怎么得到这些蓝色的函数呢，可以用(soft) Sigmoid Function来模拟
![model](/image_host/1.png)
Model想要模拟任何曲线，就要调整Sigmoid Function中的参数
![model](/image_host/2.png)
可以考虑多个feature
![model](/image_host/3.png)
做一些符号上的简化
![model](/image_host/4.png)
![model](/image_host/5.png)
把Model写成矩阵和向量相乘的样子，其中δ表示Sigmoid Function
![model](/image_host/6.png)
把这些参数中的向量拉直（把矩阵分成row或column），拼在一起组成一个向量，称作Θ，即Θ代表所有未知参数
![model](/image_host/7.png)

#### 2.使用了新的Model的损失函数

![model](/image_host/8.png)

#### 3.使用了新的Model的优化过程，其中η表示learning rate

![model](/image_host/9.png)
实际操作时可能不是对整个数据集算Loss（进而算Gradient），而是分批做
![model](/image_host/10.png)

#### 4.其实不一定要用soft sigmoid function来模拟hard sigmoid function，也可以用两个ReLU叠加得到一个hard sigmoid function（其实ReLU效果更好）

![model](/image_host/11.png)
使用ReLU后的Model（sigmoid、ReLU这些都叫activation function）
![model](/image_host/12.png)

#### 5.为了改进模型，可以使用更多层

![model](/image_host/13.png)

#### 6.深度学习名字的由来

sigmoid、ReLU叫做Neuron，原来的神经网络现在叫深度学习
![model](/image_host/14.png)
