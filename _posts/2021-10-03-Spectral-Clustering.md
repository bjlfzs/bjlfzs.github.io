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

在图上进行谱聚类，首先需要建图。

其主要解决如何将采样点数据构造成图结构的问题。其中两点间的距离有三种构图方式:

\1. $$\epsilon$$ -neighborhood
\2. k-nearest neighborhood
\3. fully connected

​    

---
<br>



## 算法实现

1. 根据数据构造一个图的affinity matrix，图中每一个节点对应一个数据点，affinity matrix中每条边连接两个点，其权重代表数据间的相似度。在邻接矩阵的形式下为$W$，其有三种不同的构图方式(见下文)，也可以直接建出有$$k$$个强连通分量的图。
2. 计算出度矩阵$$D_{i,i}=\sum_{j=1}^n W_{i, j}$$，算出Laplace矩阵$$L=D-W$$，对于Ncut而言，需要对其进行归一化，即$$L=D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}$$。
3. 求出Laplace矩阵$$L$$的最小的$$k$$ 个特征值$${\lambda\}_{i=1}^k$$ 以及对应的特征向量$${\mathbf{v}\}_{i=1}^k}$$
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
