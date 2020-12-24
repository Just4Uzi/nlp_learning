来源： https://zhuanlan.zhihu.com/p/37683646


* Global average pooling是什么？


    * 最早用于卷积神经网络中，global average pooling用于替代全连接层。Global average pooling就是平均所有的feature map，然后将平均后的feature map喂给softmax进行分类。

* 为什么用来替代全连接层？有什么优势？


    * 全连接层比较容易过拟合，影响整个模型的泛化能力，dropout的引入部分解决了dense layer的过拟合问题。global average pooling的优势在于只是平均，没有参数。其使用了卷积层+dense layer，很难解释从loss back-propagate回去的是什么，更像是一个黑盒子，而global average用简单的average建立起了feature map和category之间的联系，简单来说，以手写体识别的分类问题为例，就是每个类别仅仅对应一个feature map，每个feature map内部求平均，10个feature map就变成了一个10维向量，然后输入到softmax中。全连接层过于依赖dropout来避免过拟合，而global average pooling可以看做是一种结构化正则，即对多个feature map的**空间平均（spatial average）**，能够应对输入的许多空间变化（翻转、平移等）。只要是能减少模型参数的办法，都能降低过拟合。。。深度学习的打开方式也是狂加 参数+layer 来增强模型学习能力，让模型先过拟合，再加 正则+降参数 避免过拟合。。。![img](https://pic1.zhimg.com/80/v2-9036dbeb3c1adc2b73f6b0d6bf60924c_720w.jpg)

* 

* 

* 

* 

* 

* 

    


