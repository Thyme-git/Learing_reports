## 培训目标

1. 能运用 Python 语言进行代码的编写，独立完成研究性项目代码的编写
2. 学习深度学习、强化学习基础内容，了解其中的原理
3. 学习 git、linux 等开发工具，上手项目工作

## 培训计划

### Step1 Git、Linux基础

学习时长：2天

学习目标：Git代码管理、Linux基础操作

任务：

1. [X] 使用 git 建立并管理自己的 github 仓库，更新并维护后两步的学习任务代码
2. [X] 配置 vscode 远程开发环境

遇到的问题：

  服务器的地址以及登陆账号没有获取(已解决)

### Step2 深度学习基础

学习时长：1周

学习目标：学习pytorch的使用和深度学习基础概念

任务：

1. [X] 基础：使用线性层、激活函数搭建网络，实现写数据集 MNIST 的分类任务

    [usage]

    ```
    cd Step2_MNIST
    python3 train.py
    	(optional) --lr <learning rate(defult 1e-3)>
    	(optional) --epoches <num of epoch(defult 100)>
    	(optional) --outpath <path model to save model(defult  Step2_MNIST/model)>
    	(optional) -p <usepretrained model>
    python3 eval.py
    ```
2. [X] 进阶：使用卷积层搭建网络，实现 CIFAR-10 数据集的分类任务

    [usage]

    ```
    cd Step2_CIFAR10
    python3 train.py
    	(optional) --lr <learning rate(defult 1e-3)>
    	(optional) --epoches <num of epoch(defult 100)>
    	(optional) --outpath <path model to save model(defult  Step2_CIFAR10/model)>
    	(optional) -p <use pretrained model>
    python3 eval.py
    ```


参考资料：

1. 斯坦福CV课程：cs231n
2. 电子书：[《动手学深度学习》 ](https://zh-v2.d2l.ai/)

### Step3 强化学习基础

学习时长：2、3周

学习目标：学习强化学习基础概念和 DQN、DDPG 算法族

任务：

1. 基础：使用 DQN 算法，学习如何玩gym游戏：Pong-v0
2. 基础：使用 DDPG算法，学习如何玩gym游戏：Pong-v0
3. 进阶：学习DQN算法族：Double DQN、Dueling DQN

参考资料：

1. [刘建平Pinard](https://www.cnblogs.com/pinard/default.html?page=2)
2. [强化学习中文教程（蘑菇书）](https://github.com/datawhalechina/easy-rl)
