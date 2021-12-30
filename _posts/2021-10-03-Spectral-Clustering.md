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

可以很直观理解的是其优化的公式不同，而其对应着不同的物理意义。传统的聚类方法(K-Means)的重点在于聚类，其可以获得一个聚类核$\mu$和对应的分类结果。而谱聚类做的更多的是降维的工作，给定了几个使整个图扰动最小的图上的扰动，将原图降维成和聚类个数相同的维度。

由于采用了图模型，其对于局部信息捕捉效果较好，谱聚类也更容易捕捉到局部粒度的特征。





<h3>距离度量</h3>

在图上进行谱聚类，首先需要建图，即将采样点数据构造成图结构。其中两点间的距离有三种构图方式:

1. $\epsilon$ -neighborhood
2. k-nearest neighborhood
3. fully connected

其中前二者用的比较多，其具体的定义。

在谱聚类中，距离度量对于聚类结果影响性较大，其中对于高斯建图而言，其$\sigma$是主要影响建图的因素，应当仔细调节。

