---
title: 非负矩阵分解算法
author: theq Yang
date: 2023-02-23 13:00:00 +0800
categories: [Blogging, Algorithm]
tags: [Optimization]
math: true
mermaid: true
---

在这里对拉普拉斯对偶问题进行简单分析。 

考虑标准形式的优化问题：

$$\min f_0(x) \\
s.t.~~ f_i(x) \leq 0 ~~i = 1, \dots , m, \\ 
 ~~~~~~~~h_i(x) \leq 0 ~~i = 1, \dots , p. $$
其中自变量$x \in R^{n}$. 

采用拉格朗日乘子法进行求解：
$$L(x, \lambda, \gamma) = f_0(x) + \sum_{i=1}^m \lambda_i f_i (x) + \sum_{i=1}^p \gamma_i h_i (x)$$


其 Lagrange 对偶函数为：

$$g(\lambda, \gamma) = \inf_x L(x, \lambda, \gamma) = \inf_x f_0(x) + \sum_{i=1}^m \lambda_i f_i (x) + \sum_{i=1}^p \gamma_i h_i (x).$$

其中，倘若$p^*$ 是拉格朗日函数的最小值，那么其对偶函数具有如下性质：

$$g(\lambda, \gamma)  \leq p^*$$


<h3 data-toc-skip>概述</h3>



<h4 data-toc-skip>非负矩阵分解</h4>

非负矩阵分解希望将一个大非负矩阵分解成两个非负的小矩阵相乘。其中由于图像等数据是有非负约束的，所以这里的非负约束可以使获得的矩阵类似于NMF算法优化下式：

$$V = WH    \\
s.t. \quad W \geq 0, H \geq 0$$

倘若对于某一个低秩的，所有值非负的矩阵$V \in R^{n \times m}$，假设$rank(V) = r$，那么一定可以找到两个矩阵$W \in R^{n \times r}$和$V\in R^{r \times m}$(不能保证非负性？)，使得$V=WH$成立(类似于SVD，就是稍微变形一下)。在这里可以发现，V的秩较小，意味着其中蕴含的信息较少，通过分解可以得到两个矩阵，通过两个矩阵既可以表示出原矩阵的形态。这里我们将前面的矩阵$W$认为成低维的映射，从$n$维到$r$维。而后者可以看作是映射得到的$V$投影到$W$的向量。

然而当$V$的秩较高时，显然无法找到$W$和$H$来实现如上的分解，那么如何找到这一个映射呢？其解决方案在于找到一个与之相差不大的矩阵，其可以降维就好了。而如何判断其差距大小呢？通常可以采用两种方法：F-norm和KL-divergance。即

$$\min_{W \geq 0, H \geq 0} \frac{1}{2}\left\|V - WH \right\|_F^2  $$
或
$$\min_{W \geq 0, H \geq 0} D_{KL}(V\|WH) = \sum_{i = 1}^n \sum_{j = 1}^m \left(v_{ij}ln\frac{v_{ij}}{WH}_{ij} - v_{ij} + [WH]_{ij} \right)$$


可以很直观理解的其是从欧式距离和相对熵两种不同的角度优化的，而多视图学习中工作以前者为主，而后者可以当作一个构建图像生成概率模型的方法。


<h4 data-toc-skip>NMF方法优化</h4>

NMF问题的求解没有一个trival的解，通常采用迭代的方式求解。最简单的想法是采用梯度下降的方式：
$$W_{ia} \gets W_{ia} - \mu_{ia}  \frac{\partial D(V\|WH)}{W_{ia}}$$
$$H_{au} \gets H_{au} -  \eta_{au}  \frac{\partial D(V\|WH)}{H_{au}} $$
其中$D(V\|WH)$代表某种距离度量的方式。该方法需要合理选择$\mu$和$\eta$，通常收敛速度较慢，而且难以保证其优化结果非负。

更普遍的方法同样采用两步优化的形式，仅对上文的$\mu$和$\eta$赋和$WH$有关的正值即可。
对于欧式距离而言，其梯度可以轻松算出：

$$\frac{\partial D(V\|WH)}{W_{ia}} = -\left((V - WH) H^T \right)_{ia}$$
$$\frac{\partial D(V\|WH)}{H_{au}} = -\left(W^T(V - WH) \right)_{au}$$
观察这两个式子，带入梯度下降的式子，得到：

$$W_{ia} \gets W_{ia} + \mu_{ia} \left((V - WH) H^T \right)_{ia}$$
$$H_{au} \gets H_{au} +  \eta_{au} \left(W^T(V - WH) \right)_{au} $$
可以看出，当$\mu_{ia} = \frac{W_{ia}}{(WHH^T)_{ia}}$，$\eta_{au} = \frac{H_{au}}{(W^TWH)_{au}}$时，其第一项可以被抵消，得到

$$W_{ia} \gets W_{ia} \frac{(VH^T)_{iu}}{(WHH^T)_{iu}}$$
$$H_{au} \gets H_{au}  \frac{(W^TV)_{au}}{(W^TWH)_{au}} \sum_i W_{ia}\frac{V_{iu}}{(WH)_{iu}} H_{au}$$
二者自始至终都是非负的，而且最终会收敛，将梯度下降方法转换为乘法优化，可以实现更快，更稳定的收敛。同理对于KL散度也有类似的优化：
令$\mu_{ia} = \frac{1}{\sum_{j = 1}^nH_{ji}}$，$\eta_{au} = \frac{1}{\sum_{k = 1}^m W_{ka}}$，
带入上式，也可以得到相同的优化结果。所以在Nature上的[文章](http://lsa.colorado.edu/LexicalSemantics/seung-nonneg-matrix.pdf)
将其整理成以下式子：
$$W_{ia} \gets W_{ia} \sum_u \frac{V_{iu}}{(WH)_{iu}} H_{au}$$
$$H_{au} \gets H_{au}\sum_i W_{ia}\frac{V_{iu}}{(WH)_{iu}} H_{au}$$
<!-- $$W_{ia} \gets W_{ia} \sum_u \frac{V_{iu}}{(WH)_{iu}} H_{au}$$ -->


<h4 data-toc-skip>与其他方法比较</h4>

上述方法解决了如何实现NMF的问题，但是这不足以说明其有效性。
上文所述文章中有一张图很好的展示了NMF与PCA， QV的对比。
<!-- ![NMF-contracted.png](https://postimg.cc/62ndMBDx) -->
[![NMF-contracted.png](https://i.postimg.cc/631Y53cq/NMF-contracted.png)](https://postimg.cc/62ndMBDx)
其中QV的$H$采用一元编码，其中某一个列是1，其余是0，相当于一种特征压缩的方式；而PCA需要关于$W$的列和$H$的行都要满足正交条件。
从图中可以看出，QV难以实现对特征的提取；而在可以实现特征提取的PCA和NMF方法中，由于PCA无符号限制，所以其提取的特征有正有负(红色部分)，而NMF算法提取的特征效果较好，可以明显看出来其处理出了局部的细节，然后加权得到结果。

当然，该算法也可以去碰瓷其他的算法，与之比较的也是约束上面的一些不同。
如K-means算法试图优化点到该类聚类中心的距离和。倘若将K-means算法写成类似NMF的形式，得到如下式子：
$$\min_{W_i \in index, H} \frac{1}{2}\left\|V - WH \right\|_F^2 . $$
其中$H$代表选出来的$r$个列向量，即$r$个聚类中心；
而$W$代表$m$个样本分别属于$r$簇中的哪一类。其与NMF的区别也主要在于缺少了非负的约束，而对于$W$的行向量$W_i$加入了one-hot的约束，其中$1$所在的位置代表样本属于哪个类。

<h4 data-toc-skip>总结</h4>
总而言之，NMF方法由来已久，但是其提出的非负约束是其亮点所在，通过该约束可以更好的通过无监督方式提取特征，同时也是获取低维子空间映射的常用方法之一。
<!-- <h4 data-toc-skip> NMF方法</h4> -->

## Blog Lists



### Task list
