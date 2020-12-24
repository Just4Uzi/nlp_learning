来源：https://blog.csdn.net/a358463121/article/details/103772991



# 理解Jacobian矩阵



## Locally Linear

我们考虑一下简单的函数：
f ( [ x y ] ) = [ x + sin ⁡ ( y ) y + sin ⁡ ( x ) ] f\left(\left[

xyxy

\right]\right)=\left[

x+sin(y)y+sin(x)x+sin⁡(y)y+sin⁡(x)

\right]*f*([*x**y*])=[*x*+sin(*y*)*y*+sin(*x*)]
他将一个[x,y]的点，经过一个变换，就像下图那样：



![在这里插入图片描述](D:\NLP\nlp_learning\Mathematics\pic\20191230202943325.gif)



## 线性变换

线性变换是什么意思？我们知道，一个向量乘一个矩阵其实就是一个线性变换，但直观来看，是什么样的，我们看以下这个线性变换是怎样的。

[ 2 − 3 1 1 ] [ x y ] → [ 2 x + ( − 3 ) y 1 x + 1 y ] \left[

21−312−311

\right]\left[

xyxy

\right] \rightarrow \left[

2x+(−3)1x+1yy2x+(−3)y1x+1y

\right][21−31][*x**y*]→[2*x*+(−3)1*x*+1*y**y*]

我们发现，基变换后的结果恰好对应与变换矩阵的第一列和第二列！



## Jacobian Matrix

现在回到正题，我们刚才说了，**非线性的变换在某个局部点上的变换，可以看作是一个线性变换，而这个线性变换应该是一个2\*2的矩阵来的，我们希望知道这个矩阵是什么。**

现在我们开始分析一下这个局部变换：先考虑在原空间上x轴一个很小的距离dx:



![在这里插入图片描述](D:\NLP\nlp_learning\Mathematics\pic\20191230214739299.png)



经过一个线性变化，这个很短的∂ x \partial x∂*x*变成了在另一个空间中很小的一步（如下图绿色箭头）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191230212311328.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2EzNTg0NjMxMjE=,size_16,color_FFFFFF,t_70)

这个绿色的箭头，就是∂ x \partial x∂*x*经过变换后的样子，可以看到这个变换是斜向下的变换，也就是说他同时改变了两个坐标，是一种2维的变换，该变换可以分解成两个坐标上的变换，在x轴上的变换后的长度就是绿色虚线，这个长度是等于∂ f 1 = ∂ f 1 / ∂ x ∗ ∂ x \partial f_1=\partial f_1/\partial x*\partial x∂*f*1=∂*f*1/∂*x*∗∂*x*，而变换率则是∂ f 1 / ∂ x \partial f_1/\partial x∂*f*1/∂*x* (ps: 之所以可以用导数表示变化率是因为这就是导数的定义：lim ⁡ Δ x → 0 f ( x + Δ x , y ) − f ( x , y ) Δ x \lim_{\Delta x \to 0}\frac{f(x+\Delta x,y)-f(x,y)}{\Delta x}limΔ*x*→0Δ*x**f*(*x*+Δ*x*,*y*)−*f*(*x*,*y*))，同理，在y轴上的变换是红色虚线，∂ f 2 / ∂ x \partial f_2/\partial x∂*f*2/∂*x*.