---
title: 谱聚类
author: theq Yang
date: 2021-10-03 13:00:00 +0800
categories: [Blogging, Algorithm]
tags: [Spectral Graph]
math: true
mermaid: true
---

最近正在进行对于multiview clustering的学习，其中着重学习了谱聚类以及相关的算法，在这里先简单总结一下，随着学习的深入再慢慢补充。

<h3 data-toc-skip>概述</h3>



<h4 data-toc-skip>为什么要引入谱聚类</h4>

可以很直观理解的是其优化的公式不同，而其对应着不同的物理意义。传统的聚类方法(K-Means)的重点在于聚类，其可以获得一个聚类核$$\mu$$和对应的分类结果。而谱聚类做的更多的是降维的工作，给定了几个使整个图扰动最小的图上的扰动，将原图降维成和聚类个数相同的维度。

由于采用了图模型，和后面讲的距离度量结合，谱聚类容易捕捉到局部粒度的特征。

%与此同时，谱聚类容易获得




<h3>距离度量</h3>

在图上进行谱聚类，首先需要建图，即将采样点数据构造成图结构。其中两点间的距离有三种构图方式:

1. $$\epsilon$$ -neighborhood
2. k-nearest neighborhood
3. fully connected

其中前二者用的比较多，其具体的定义。

在谱聚类中，距离度量对于聚类结果影响性较大，其中对于高斯建图而言，其$$\sigma$$是主要影响建图的因素，应当仔细调节。

---
<br>

<h3>Laplace矩阵</h3>

提到算法的有效性，首先需要介绍Laplace矩阵的性质。首先，在数学意义上，拉普拉斯算子是计算梯度的散度。众所周知，在多变量函数中，梯度是一个向量场，揭示了函数下降最快的方向；而散度则是判断该点是否有源，即流入和流出是否相等，在该链接中也有讲解[[Link]](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives)。所以梯度的散度可以理解成给某一个点以一个微小的扰动，整体得到的增益。

在此基础上，引入了图的Laplace Matrix。

假设具有$$N$$个节点的图 $$G$$ ，图上每点定义的特征函数$$f$$是$$N$$维向量：$$f = (f_1, f_2,...,f_N)$$，其中$$f_i$$为函数$$f$$在图中节点$$i$$ 处的函数值，在谱聚类中$$f$$又可以理解成聚类指示矩阵。而对于某一个$$f_i \neq 0$$，可以看作是对节点$$i$$进行扰动，它可能变为任意一个与之直接相邻的节点$$j \in Neighbor(i)$$。

对$$f$$函数做拉普拉斯

可以将扰动看成加入一个势，$$\Delta f_i = \sum_{j \in Neighbor(i)} W_{i,j}(f_i - f_j)$$ 体现加入该势后整张图在一时刻的变化。譬如对于下面这个图

![SC-1.png](https://postimg.cc/sQQqrdsh)

其Laplace Matrix为

$$$$



## 算法实现

1. 根据数据构造一个图的affinity matrix，图中每一个节点对应一个数据点，affinity matrix中每条边连接两个点，其权重代表数据间的相似度。在邻接矩阵的形式下为$W$，其有三种不同的构图方式(见下文)，也可以直接建出有$$k$$个强连通分量的图。
2. 计算度矩阵$$D_{i,i}=\sum_{j=1}^n W_{i, j}$$,即把与之相连的边全部加起来；算出Laplace矩阵$$L=D-W$$，对于Ncut而言，需要对其进行归一化，即$$L=D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}$$。
3. 求出Laplace矩阵$$L$$的最小的$$k$$ 个特征值$$\lambda_{i=1}^k$$ 以及对应的特征向量$$\{\mathbf{v}\}_{i=1}^k$$
4. 把这 $$k$$个特征向量排列在一起组成一个 $$ N \times k$$的矩阵，将其中每一行看作$$k$$维空间中的一个向量，并使用 K-means 算法进行聚类。聚类的结果中每一行所属的类别就是原图中数据点的类别，也是最初的$$N$$个数据点分别所属的类别。

## Lists

### Ordered list

1. Firstly
2. Secondly
3. Thirdly

### Unordered list

- Chapter
	- Section
      - Paragraph

### Task list

- [ ] TODO
- [x] Completed
- [ ] Defeat COVID-19
  - [x] Vaccine production
  - [ ] Economic recovery
  - [ ] People smile again

### Description list

Sun
: the star around which the earth orbits

Moon
: the natural satellite of the earth, visible by reflected light from the sun


## Block Quote

> This line to shows the Block Quote.

## Tables

| Company                      | Contact          | Country |
| :--------------------------- | :--------------- | ------: |
| Alfreds Futterkiste          | Maria Anders     | Germany |
| Island Trading               | Helen Bennett    |      UK |
| Magazzini Alimentari Riuniti | Giovanni Rovelli |   Italy |

## Links

<http://127.0.0.1:4000>
