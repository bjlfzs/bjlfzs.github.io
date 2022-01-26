---
title: COMPLETER--Incomplete Multi-view Clustering via Contrastive Prediction
author: theq Yang
date: 2021-10-09 19:00:00 +0800
categories: [Blogging, Article]
tags: [MvC, Information Theory]
math: true
mermaid: true
---

## 基于对比预测的缺失视图聚类方法

<h3 data-toc-skip>摘要</h3>

实际应用中，由于数据采集和传输过程的复杂性，数据可能会丢失部分视图，这就导致了信息不完备下的视图缺失问题。例如在线会议中，一些视频帧可能由于传感器故障而丢失了视觉或音频信号。针对该问题，过去十多年已提出了一些不完全多视图聚类方法并取得了显著效果。但视图缺失问题仍面临两个主要挑战：

1）如何在不利用标签信息的情况下学习**一致的多视图公共表示** ；

2）如何从**部分缺失的数据**中还原完整的数据。

针对上述挑战，受近期Tsai等在ICLR2021上发表的工作[1]所启发，本文提供了一个新的不完全多视图聚类见解，即不完全多视图聚类中的数据恢复和一致性学习是一体两面的，两者可统一到信息论的框架中。基于上述观察，论文提出了对偶预测范式并将其与对比学习结合，通过一个新的损失函数实现了跨视图一致性与可恢复性的联合优化。大量的实验验证了所提出的损失函数的有效性。

![representation.png](https://i.postimg.cc/FKK4N2MG/representation.png)

这个图很有意思，画的让我难以理解，经过师兄的提点，又补习了一下信息论，终于理解了。

其中实线框表示第一个视图（模态）里包含的信息$X^1$；虚线框表示第二个视图（模态）里包含的信息$X^2$，左边的蓝色+灰色是$X^1$的representation$Z^1$，灰色+右边的蓝色是$Z^2$，即$X^2$的representation。注意这里两块不是连续的，他俩的并集作为$Z^2$，这令我瞪了好久。

这里的表征用Auto-encoder取中间层得到的，在这里不仅要优化互信息$I(Z^1,Z^2)$最大，而且要使得$H(Z^i|Z^j)$最小，这样的理想情况是图就变成了右边的样子，即$Z^1$与$Z^2$完全包含，其互信息为$X^1$和$X^2$的公共区域，且两边没有多余的信息。 这样做可以discard the inconsistent information across-views, and thus the consistency could be further improved.

![Com-Pleter.png](https://i.postimg.cc/BvYtJSMk/Com-Pleter.png)

这里分为了三个模块：Within-view Reconstruction, Cross-view Contrastive Learning, Cross-view Dual Prediction， 分别对应视角内约束，视角间约束和对偶预测。

视角内约束:Auto-encoder

$$\ell_{rec} = \sum_{v=1}^{2}  \sum_{t=1}^{m} \left\|X_t^v - g^{(v)}(f^{(v)}(X_t^v) \right\|_2^2$$

对比学习：互信息

用autoencoder获取隐层信息。

视角间约束：对比学习

用两个view的联合分类概率来当作。

对偶预测：用网络。

这篇文章的特点：1. 在多视角聚类中引入了互信息的概念。 2. 通过对比学习可以实现对原视图的还原。 3. 在学习表征的同时实现了视图还原。

思考：为什么能中CVPR？ 
