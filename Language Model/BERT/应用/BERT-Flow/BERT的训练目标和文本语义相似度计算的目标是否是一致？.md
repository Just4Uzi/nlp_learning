来源: 微信公众号-NewBeeNLP- 《BERT-Flow | 文本语义表示新SOTA》


* 简介

    * 如何将预训练模型BERT迁移到文本语义相似度计算任务上？

        * 交互编码 E.g.  CLS + SENTENCE 1 + SEP + SENTENCE2

            * 优点：利用self-attention让两个文本得到充分的交互，在对一些文本交互要求较高的任务上表现的很好
            * 缺点：在文本检索的场景下，从文本库D中搜索和某个query语义最相近的文本需要做|D|次inference，很耗时
    * 向量空间模型 E.g. 利用CLS的hidden states或者用最后一层或者几层的hidden states做average pooling（后者更好），然后用句向量的cosine相似度表示文本的语义相似度
        
        * 优点：速度较快，适合大规模文本检索的场景
            * 缺点：但奇怪的是，实验表明BERT句向量的表现有时候还不如non-contextualized的GloVe句向量。
        * 针对上面的问题，Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (EMNLP 2019)[4]提出了基于siamese双塔结构的SBERT，SBERT利用了语料库中的语义相似度标签有监督地训练BERT生成更好的句向量，并且SBERT的训练目标和cosine相似度度量是完全契合的。**「SBERT将信息交互限制在了最顶层，避免了从底层就开始交互的BERT难以接受的计算复杂度，将 次inference化简到了1次，同时避免了预训练BERT在缺乏下游任务监督的情况下生成的句向量的cosine相似度和真实的语义相似度差距太大的问题」**。但SBERT有一个不可忽视的缺点和一个没有解答的问题：
            * 缺点：**「需要监督信息：」** 想要标注任意句子对的相似度的工作量是 增长的，在绝大多数文本检索的现实场景下，获取大规模句子对标签的代价是很高的。**「因此最理想的方法是在领域语料库上无监督训练的BERT生成的句向量可以直接用来有效地计算文本相似度」**。
            * 问题：**「为什么预训练BERT句向量的cosine相似度不能很好地近似语义相似度？：」**SBERT没有解释为什么没有经过监督式微调的BERT生成的句向量不好用，是因为BERT句向量本身没有包含足够的语义相似度信息，还是因为简单的cosine相似度无法准确刻画语义相似度？如果是后者，**「有没有什么方法可以在无监督的条件下更有效地抽取出BERT句向量中隐含的语义相似度信息呢？」**
* BERT句向量

    * BERT预训练与语义相似性

        * 考虑一个句子![image-20201223160838246](C:\Users\T470\AppData\Roaming\Typora\typora-user-images\image-20201223160838246.png)，语言模型将联合概率![image-20201223160934085](C:\Users\T470\AppData\Roaming\Typora\typora-user-images\image-20201223160934085.png)按自回归的方式分解为![image-20201223161001469](D:\NLP\nlp_learning\Language Model\BERT\应用\pic\image-20201223161001469.png)；
        * BERT的Mask Language Model将其分解为![image-20201223161130271](D:\NLP\nlp_learning\Language Model\BERT\应用\pic\image-20201223161130271.png)
        * 两种语言模型都可以归结成建模单词x与上下文c之间的条件分布![image-20201223161254110](D:\NLP\nlp_learning\Language Model\BERT\应用\pic\image-20201223161254110.png)
        * 我们可以猜想如果两个上下文c和c‘ 和 与同一个词 w有共现关系，那么c 和 c'也应该有相似的语义，具体来说，在训练语言模型时，c 和 w 的共现会使得hc (上下文表示)和 xw（词x的向量表示）相互靠近，对c' 来说也同样如此，因此 hc和 hc'就会变得接近，同时由于softmax标准化的原因，hc 和xw'，w ！= w' 的距离会拉大。通过这样的过程，模型可以建立上下文与上下文潜在的共现关系，这表明BERT的训练过程和语义相似度计算的目标是很接近的，训练得到的句向量应该包含了文本之间的语义相似度信息。
* **我们使用BERT句向量的方法是否不够有效？**

    * **BERT句向量隐含的语义相似度信息没那么容易被抽取出来**

        * 语义向量空间上的词向量分布通常是anisotropic（各向异性的），即X与Y与Z方向上的缩略比不同，且词向量的分布通常呈现锥形分布
        * 高频词的l2范数更小，说明高频词离原点更近，低频词离原点较远，**「这会导致即使一个高频词和一个低频词的语义是等价的，但词频的差异也会带来很大的距离偏差，从而词向量的距离就不能很好地代表语义相关性」**。
        * 高频词之间的 l2距离也更小，说明高频词分布得更紧凑，低频词分布得更稀疏，而稀疏性会导致一些词向量之间的空间是"空"的，这些地方没有明显的语义对应，因为句向量是词向量的平均池化，是一种保凸性运算，然而这些没有语义定义的空间使得分布不是凸性的，所以可以认为BERT句向量空间在一定程度上是**「语义不平滑的(semantically non-smoothing)」**，这导致句向量相似度不一定能够准确表示句子的语义相似度。
* BERT-Flow

    * 



