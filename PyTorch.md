# PyTorch

## 环境搭建

### 1.Anaconda工具包

#### 1.1下载安装

1. 在官网下载安装
2. 在菜单栏打开Anaconda Prompt（Anaconda的命令行），如果左侧有(base)，说明安装成功

#### 1.2为PyTorch创建环境

因为官网下载PyTorch太慢，先配置清华的镜像，打开Anaconda Prompt，先后输入以下命令
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`
`conda config --set show_channel_urls yes`
把pytorch对应的库也配置进来
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/`

1. 打开Anaconda Prompt，输入`conda create -n pytorch python=3.6`
2. 输入`conda activate pytorch`，激活pytorch环境
3. 输入`pip list`，查看有当前环境有哪些工具包
4. 安装PyTorch，进入PyTorch官网，根据自己实际情况选择，（其中注意看自己GPU支持什么cuda版本，具体方法见下文），然后得到相关的安装指令，如`conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`
5. 进入pytorch环境，输入安装PyTorch的指令（注意要把最后的-c pytorch去掉，否则默认从官网下载用不了镜像）
6. 输入`pip list`，查看是否安装成功
7. 输入`python`->输入`import torch`验证是否能成功导入->输入`torch.cuda.is_available()`验证是否能使用GPU

其中看自己GPU支持什么cuda版本有两种方法:

1. 直接在命令行输入`nvidia-smi`查看（如果没有添加path要先找到nvidia-smi.exe文件所在路径）
2. 先打开英伟达的控制面板查看自己显卡的型号，然后到[英伟达官网](https://www.nvidia.cn/geforce/technologies/cuda/supported-gpus/)查看自己显卡的支持的cuda版本

### 2.PyCharm

#### 2.1下载安装配置使用

1. 在官网下载安装，安装过程中注意勾选create association以便默认用PyCharm打开Python文件，可以先跳过插件的安装
2. Create new project->选择项目存放位置，建议先创建一个文件夹专门存放pytorch项目代码
3. 点开Python interpreter选择解释器->选择previously configured interpreter->选择Conda环境->选择路径为`Anaconda\envs\pytorch\python.exe`的程序文件->勾选该解释器为所有项目可用
4. 点击左下角的python console->输入`import torch`验证是否能成功导入->输入`torch.cuda.is_available()`验证是否能使用GPU

### 3.Jupyter

#### 3.1下载安装配置使用

1. 安装Anaconda会自动安装Jupyter，但是默认只在base环境中安装了Jupyter，我们要么在base环境中重新安装PyTorch，要么在pytorch环境中安装Jupyter，这里采用后者
2. 进入pytorch环境->输入`conda install nb_conda`，安装启动Jupyter所需的包
3. 输入`jupyter notebook`进入Jupyter
4. 点击右上角new->选择环境conda env:pytorch->输入`import torch`验证是否能成功导入->shift+enter运行当前代码块并跳转到下一个代码块->输入`torch.cuda.is_available()`验证是否能使用GPU

这里在pytorch环境中使用notebook出现问题尚未解决。

## 使用

### 1.两个有用的函数

1. dir()，参数为包名，可以看下属包
2. help()，参数为函数名或者包名，注意函数不加()，可以看函数或包的使用说明