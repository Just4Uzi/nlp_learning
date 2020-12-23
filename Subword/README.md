来源：https://zhuanlan.zhihu.com/p/86965595

前言

Subword算法如今已经成为了一个重要的NLP模型性能提升方法。
自从2018年BERT横空出世横扫NLP界各大排行榜之后，各路预训练语言模型如同雨后春笋般涌现，其中Subword算法在其中已经成为标配。
所以作为NLP界从业者，有必要了解下Subword算法的原理。


1. 与传统空格分隔tokenization技术的对比

* 传统方法无法很好的处理未知或罕见的词汇（OOV问题）
* 传统词tokenization不利于模型学习词缀之间的关系
    * E.g. 模型学到的 old older oldest 之间的关系无法泛化到 smart smarter smartest
* Character embedding 解决OOV问题粒度太细
* Subword 粒度在字符和词之间，能较好的平衡OOV问题

2. Byte Pair Embedding (Sennrich et al., 2015) [1]

* BPE（字节对）编码或二元编码是一种更简单的数据压缩形式，其中最常见的一对连续字节被替换成该数据中不存在的字节[2]。后期使用时需要一个替换表
来重建原始数据。 OpenAI GPT-2 和 Facebook Roberta 均采用此方法构建subword vector

* 优点
    * 可以有效平衡词汇表大小和步数（编码句子所需要的token数量）
* 缺点
    * 基于贪婪和确定的符号替换，不能提供带概率的多个分片结果。

    2.1 算法[3]
    * 准备足够大的训练语料
    * 确定期望的subword词表大小
    * 将单词拆分为字符序列并在末尾添加后缀“ </ w>”，统计单词频率。 本阶段的subword的粒度是字符。 例如，“ low”的频率为5，那么我们将其改写为“ l o w </ w>”：5
    * 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword
    * 重复第4步直到达到第2步设定的subword词表大小或下一个最高频的字节对出现频率为1
    * 停止符"</w>"的意义在于表示subword是词后缀。举例来说："st"字词不加"</w>"可以出现在词首如"st ar"，加了"</w>"表明改字词位于词尾，如"wide st</w>"，二者意义截然不同。

        每次合并后词表可能出现3种变化：

        +1，表明加入合并后的新字词，同时原来的2个子词还保留（2个字词不是完全同时连续出现）
        +0，表明加入合并后的新字词，同时原来的2个子词中一个保留，一个被消解（一个字词完全随着另一个字词的出现而紧跟着出现）
        -1，表明加入合并后的新字词，同时原来的2个子词都被消解（2个字词同时连续出现）
        实际上，随着合并的次数增加，词表大小通常先增加后减小。

    例子

    输入：

    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}
    Iter 1, 最高频连续字节对"e"和"s"出现了6+3=9次，合并成"es"。输出：

    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w es t </w>': 6, 'w i d es t </w>': 3}
    Iter 2, 最高频连续字节对"es"和"t"出现了6+3=9次, 合并成"est"。输出：

    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est </w>': 6, 'w i d est </w>': 3}
    Iter 3, 以此类推，最高频连续字节对为"est"和"</w>" 输出：

    {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w est</w>': 6, 'w i d est</w>': 3}
    ……

    Iter n, 继续迭代直到达到预设的subword词表大小或下一个最高频的字节对出现频率为1。

    2.2 BPE实现[4]
    * 见nlp_learning/Subword/BPE.py

    2.3 编码和解码[4]
        * 编码
            * 经过上述步骤，已经得到Subword的词表，将该词表按照由长到短排序。
            * 编码时，对于每个单词，遍历排好序的词表，寻找词表中是否有当前单词的子字符串，如果有，则该token是表示单词的tokens之一。
            * 按词表中的顺序，由最长的tokens迭代到最短的tokens，尝试将每个单词的子字符串替换成token。
            * 最终，迭代完词表中的所有tokens，并将所有子字符串替换成tokens。如果有子字符串没被替换但所有token都已被迭代完毕，则将剩余
            的子词替换成特殊token，如<unk>。

            例子

            # 给定单词序列
            [“the</w>”, “highest</w>”, “mountain</w>”]

            # 假设已有排好序的subword词表
            [“errrr</w>”, “tain</w>”, “moun”, “est</w>”, “high”, “the</w>”, “a</w>”]

            # 迭代结果
            "the</w>" -> ["the</w>"]
            "highest</w>" -> ["high", "est</w>"]
            "mountain</w>" -> ["moun", "tain</w>"]

            * 编码的计算量很大。 在实践中，我们可以pre-tokenize所有单词，并在词典中保存单词tokenize的结果。
              如果我们看到字典中不存在的未知单词。 我们应用上述编码方法对单词进行tokenize，然后将新单词的tokenization添加到字典中备用。

        * 解码
            * 将所有tokens拼在一起

            例子：

            # 编码序列
            [“the</w>”, “high”, “est</w>”, “moun”, “tain</w>”]

            # 解码序列
            “the</w> highest</w> mountain</w>”


3. WordPiece (Schuster et al., 2012)[5]

    * WordPiece算法是BPE算法的变种。不同之处在于，WordPiece基于概率生成新的SubWord而不是下一最高频字节对。

    3.1 算法[3]
        * 准备足够大的训练语料
        * 确定期望的Subword词表大小
        * 将单词拆分成字符序列
        * 基于第3步数据训练语言模型
        * 从所有可能的Subword单元中，选择加入语言模型后能最大程度增加训练数据概率的单元作为新的单元
        * 重复上一步直到达到第2步设定的Subword词表大小或者概率增量低于某一阈值


4. Unigram Language Model (Kudo, 2018)[6]

    * ULM是另外一种subword分隔算法，它能够输出带概率的多个子词分段。它引入了一个假设：所有subword的出现都是独立的，
    并且subword序列由subword出现概率的乘积产生。WordPiece和ULM都利用语言模型建立subword词表。

    4.1 算法[3]
        * 准备足够大的训练语料
        * 确定期望的subword词表大小
        * 给定词序列优化下一个词出现的概率
        * 计算每个subword的损失
        * 基于损失对subword排序并保留前X%。为了避免OOV，建议保留字符级的单元
        * 重复第3至第5步直到达到第2步设定的subword词表大小或第5步的结果不再变化

5. 总结

subword可以平衡词汇量和对未知词的覆盖。 极端的情况下，我们只能使用26个token（即字符）来表示所有英语单词。
一般情况，建议使用16k或32k子词足以取得良好的效果，Facebook RoBERTa甚至建立的多达50k的词表。
对于包括中文在内的许多亚洲语言，单词不能用空格分隔。 因此，初始词汇量需要比英语大很多。


参考
1^ Sennrich, Rico, Barry Haddow, and Alexandra Birch. "Neural machine translation of rare words with subword units."arXiv preprint arXiv:1508.07909(2015).
2^ Byte pair encoding - Wikipedia https://en.wikipedia.org/wiki/Byte_pair_encoding
3^ abc3 subword algorithms help to improve your NLP model performance https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46
4^ abLei Mao's Log Book – Byte Pair Encoding https://leimao.github.io/blog/Byte-Pair-Encoding/
5^ Schuster, Mike, and Kaisuke Nakajima. "Japanese and korean voice search." 2012 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2012.
6^ Kudo, Taku. "Subword regularization: Improving neural network translation models with multiple subword candidates." arXiv preprint arXiv:1804.10959 (2018).