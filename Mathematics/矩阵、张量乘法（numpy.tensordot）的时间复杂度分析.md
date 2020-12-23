来源: https://liwt31.github.io/2018/10/12/mul-complexity/


两个大小都是N×N的矩阵相乘，如果使用naive的算法，时间复杂度应该是O(N3)，如果使用一些高级的算法，可以使幂指数降到3以下。
对于一般情况的矩阵乘法，特别是张量乘法（numpy中的tensordot函数），时间复杂度又如何呢？

二维矩阵乘法
首先规定一下记号：AMN，表示一个有两个指标，大小是M×N的矩阵A。那么AMNBNL的时间复杂度是O(MNL)。
如果我们把乘法的过程用计算机语言表示出来，这一结论就会非常清晰：

C = np.zeros((M, L))
for m in range(M):
    for l in range(L):
        for n in range(N):
            C[m][l] += A[m][n] * B[n][l]
我们也可以简单地验证一下numpy.dot函数是否满足这样的时间复杂度，首先变化M。为了节省篇幅，一次将其扩大到四倍：

M = 71
N = 513
L = 4097
for i in range(5):
    m1 = np.random.random((M, N))
    m2 = np.random.random((N, L))
    %timeit m1.dot(m2)
    M *= 4
输出是：

100 loops, best of 3: 6.82 ms per loop
10 loops, best of 3: 22.5 ms per loop
10 loops, best of 3: 77.5 ms per loop
1 loop, best of 3: 304 ms per loop
1 loop, best of 3: 1.38 s per loop
可见基本是线性的（耗时一次扩大到四倍）。然后变化N，代码和上面的一段只变了一个字母，输出是：

100 loops, best of 3: 6.79 ms per loop
10 loops, best of 3: 22.1 ms per loop
10 loops, best of 3: 84.4 ms per loop
1 loop, best of 3: 329 ms per loop
1 loop, best of 3: 1.31 s per loop
仍然基本是线性的。最后变化L，输出是：

100 loops, best of 3: 8.42 ms per loop
10 loops, best of 3: 43.5 ms per loop
10 loops, best of 3: 115 ms per loop
1 loop, best of 3: 408 ms per loop
1 loop, best of 3: 1.88 s per loop
耗时是三组实验中最长的。结果汇总起来如下图

1

不难发现，时间与矩阵维度的关系是线性的且斜率为1，所以AMNBNL的时间复杂度是O(MNL)。

高维矩阵（张量）乘法-只对一个轴求和
在numpy中dot，einsum，tensordot等函数都可以做高维矩阵乘法，这里只研究最常见的tensordot。
我们从AMNLBLPQ这样一个例子入手。从理论上分析，AMNLBLPQ的时间复杂度是O(MNLPQ)，感兴趣的读者可以自己写写代码分析，或者看一看我之前写的一篇博文。
这里简单做一下实验，变化M：

M = 63
N = 17
L = 255
P = 127
Q = 31
for i in range(5):
    m1 = np.random.random((M, N, L))
    m2 = np.random.random((L, P, Q))
    %timeit np.tensordot(m1, m2, 1)
    M *= 4
输出是：

10 loops, best of 3: 47.6 ms per loop
1 loop, best of 3: 166 ms per loop
1 loop, best of 3: 700 ms per loop
1 loop, best of 3: 2.7 s per loop
1 loop, best of 3: 11.5 s per loop
而变化L输出是：

10 loops, best of 3: 46.3 ms per loop
10 loops, best of 3: 116 ms per loop
1 loop, best of 3: 368 ms per loop
1 loop, best of 3: 1.52 s per loop
1 loop, best of 3: 6 s per loop
如图所示：

2

类似地，耗时与M和L都是线性关系，后者速度貌似比前者略快。

高维矩阵（张量）乘法-对多个轴求和
下面我们再考虑对多个轴求和的情况，这种情况下“数学语言”已经不好给出清晰的描述了。
如果想举个例子，也只能啰嗦地说：AMNL和BNLP之间进行双点积contract掉维数为N和L的两个指标。倒是计算机语言还算游刃有余：

C = np.zeros((M, P))
for m in range(M):
    for p in range(P):
        for n in range(N):
            for l in range(L):
                C[m][p] += A[m][n][l] * B[n][l][p]
也容易据此估计出时间复杂度为O(MNLP)。实验一下的话，首先试试M：

M = 63
N = 31
L = 255
P = 127
for i in range(5):
    m1 = np.random.random((M, N, L))
    m2 = np.random.random((N, L, P))
    %timeit np.tensordot(m1, m2, 2)
    M *= 4
输出为：

100 loops, best of 3: 2.41 ms per loop
100 loops, best of 3: 5.8 ms per loop
10 loops, best of 3: 23.2 ms per loop
10 loops, best of 3: 171 ms per loop
1 loop, best of 3: 817 ms per loop
然后N和L分别为：

100 loops, best of 3: 2.43 ms per loop
100 loops, best of 3: 8.69 ms per loop
10 loops, best of 3: 33.7 ms per loop
10 loops, best of 3: 138 ms per loop
1 loop, best of 3: 560 ms per loop
和

100 loops, best of 3: 2.69 ms per loop
100 loops, best of 3: 9.01 ms per loop
10 loops, best of 3: 36.2 ms per loop
10 loops, best of 3: 140 ms per loop
1 loop, best of 3: 563 ms per loop
总结起来如图所示：

3

结语

总结规律的话，要想知道矩阵、张量乘法的时间复杂度，就把**两个矩阵、张量所有没contract掉的维度乘起来，再把contract掉的维度两个取一个乘起来**即可。
举个例子：AMNLBLPQ，没有contract掉的维度乘起来即NMPQ，contract掉的维度有两个L，只取一个，最后合起来就是O(MNLPQ)。

这一规律其实很好理解。np.tensordot在实现时实际上是对普通的np.dot的一个包装，进行了一些前处理和后处理。
所谓前处理，基本上就是通过转置和合并（np.reshape）把两个参与运算的高阶张量分别变成矩阵，其中一个指标是原张量所有没contract掉的指标组成的，维度自然就是这些指标的维度的积，而另一个指标是原张量要进行contract的指标组成的，维度也是这些指标的维度的积。
而后处理，就是将np.dot之后的结果再通过np.reshape变回原来的形状。np.tensordot的代码位于numpy/core/numeric.py中，核心部分如下图所示（NumPy 1.15)：

at = a.transpose(newaxes_a).reshape(newshape_a)
bt = b.transpose(newaxes_b).reshape(newshape_b)
res = dot(at, bt)
return res.reshape(olda + oldb)

其中a和b是调用者传入的要进行tensordot的矩阵，newaxes_a等参数是根据调用者指定的contract规则确定的用于将a或者b变形为适合进行np.dot的参数。
得到变形后的at和bt后直接进行dot，再将中间结果reshape回去就得到了最终的结果。所以张量乘法的时间复杂度与矩阵乘法的时间复杂度其实是一回事。