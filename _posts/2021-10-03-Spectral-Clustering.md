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

在谱聚类中，距离度量对于聚类结果影响性较大，其中对于高斯建图$a_{i,j} = e^{-\frac{\| x_i - x_j\|^2}{2\sigma^2}}$而言，其$\sigma$是主要影响建图的因素，希望最终的图中对于长短距离有较大区别。当$\sigma$过大时，$e$的幂次趋近于0，其长短距离都趋近于1；当$\sigma$过小时，$e$的幂次趋近于$-\infty$，其长短距离都趋近于0。所以合理选择该参数是聚类成功的一个重要的条件。
在CLR方法中，聂老师也提出过一种建图方式，和k-nearest近似，但是仅用基础四则运算既可获得更加准确的距离度量。

---
<br>

<h3>Laplace矩阵</h3>

提到算法的有效性，首先需要介绍Laplace矩阵的性质。首先，在数学意义上，拉普拉斯算子是计算梯度的散度。众所周知，在多变量函数中，梯度是一个向量场，揭示了函数下降最快的方向；而散度则是判断该点是否有源，即流入和流出是否相等，在该链接中也有讲解[[Link]](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives)。所以梯度的散度可以理解成给某一个点以一个微小的扰动，整体得到的增益。

在此基础上，引入了图的Laplace Matrix。

假设具有$N$个节点的图 $G$ ，图上每点定义的特征函数$f$是$N$维向量：$f = (f_1, f_2,...,f_N)$，其中$f_i$为函数$f$在图中节点$i$ 处的函数值，在谱聚类中$f$又可以理解成聚类指示矩阵。而对于某一个$f_i \neq 0$，可以看作是对节点$i$进行扰动，它可能变为任意一个与之直接相邻的节点$j \in Neighbor(i)$。

对$f$函数做拉普拉斯，相当于将扰动看成一个外加势，$\Delta f_i = \sum_{j \in Neighbor(i)} W_{i,j}(f_i - f_j)$ 体现加入该势后各个点在一瞬刻的变化。

同时有
    $$ \Delta f_i = \sum_{j \in Neighbor(i)} W_{i,j}(f_i - f_j) = \sum_{j} W_{i,j}f_i -  \sum_{j} W_{i,j}f_j 	= d_i f_i - W_{i:}f $$

所以可以认为 $ \Delta f = (D - W)f = L f $

即认为L是图拉普拉斯矩阵，其与每个点的特征属性$f$作用，可以计算$f$作用下各点的扰动。

譬如对于下面这个图，假设各条边权为1，

<img src="https://i.postimg.cc/C1mgbS2m/SC-1.png" alt="SC-1.png" style="zoom:50%;" />

其Laplace Matrix为：
$$
\left[ 
\begin{matrix}
2 & -1 & -1 & 0 \\
-1 & 2 & -1 & 0 \\	
-1 & -1 & 3 & -1 \\
0 & 0 & -1 & 1 
\end{matrix}
\right]
$$

假设$\begin{matrix} f = [1&0&0&0]^T \end{matrix}$， 则$\begin{matrix} \Delta f = L f = [2 & -1& -1 & 0]^T \end{matrix}$ ，即在第1个点加入1个单位的势，其2、3个点会(暂时地)呈现出负势的状态，使得第1个点的势向(与之相邻的)2、3个点流去，其中流的多少取决于连接的紧密程度，在现实场景下其各点值未必相等，方可体现出这种亲疏程度。

而对于这种振动，若在相同的势下观测整体的振动程度，即为$f^T L f = 2$。这样的形式很类似于能量，其表现的含义也与能量类似：即给定一种特征的分割，其按照该分割所需的能量。在这里，$\begin{matrix} f = [1&0&0&0]^T \end{matrix}$将第1个节点和2、3、4个节点分隔开来，其需要切两条值为1的边，其消耗的能量即为2。这里$f$为聚类指示矩阵，仅由1和0构成；而现实中则由于优化的需求，$f$常常无法满足该性质，通常将其约束放松成连续值，其意义却是相似的。所以谱聚类的需求就是最小化这个能量函数$f^T L f$。对于多类而言，其需要多个$f$才能实现聚类，采用$F_{n \times k} = \{f^1, f^2,...f^k\}$，取$trace(F^T L F)$作为优化目标。

<h4> 拉普拉斯矩阵性质</h4>

关于图的拉普拉斯矩阵，满足以下性质：
1. 对于任意的向量$f \in R^n$，有$$f^T Lf = \frac{1}{2} \sum_{i, j=1}^nw_{i,j}(f_i - f_j)^2$$
2. 由于$L$是对称矩阵，由性质1也可以看出，$L$是半正定阵。
3. 全1向量$\textbf{1}$是其特征向量，对应特征值为0。
    可以用$f_i=f_j$带入性质1中的式子验证，也可以用$L$矩阵的性质证明。


## 算法实现

1. 根据数据构造一个图的affinity matrix，图中每一个节点对应一个数据点，affinity matrix中每条边连接两个点，其权重代表数据间的相似度。在邻接矩阵的形式下为$W$，其有三种不同的构图方式(见下文)，也可以直接建出有$k$个强连通分量的图。
2. 计算度矩阵$D_{i,i}=\sum\limits_{j=1}^n W_{i, j}$,即把与之相连的边全部加起来；算出Laplace矩阵$L=D-W$，对于Ncut而言，需要对其进行归一化，即$\tilde{L}=D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}$。
3. 求出Laplace矩阵$L$的最小的$k$ 个特征值$\lambda_{i=1}^k$ 以及对应的特征向量$\{\mathbf{v}\}_{i=1}^k$
4. 把这 $k$个特征向量排列在一起组成一个 $ N \times k$的矩阵，将其中每一行看作$k$维空间中的一个向量，并使用 K-means 算法进行聚类。聚类的结果中每一行所属的类别就是原图中数据点的类别，也是最初的$N$个数据点分别所属的类别。

<h3>Cut</h3>

在图论中，一种Cut代表一种切割图的方式，其可以将一个完整的图变为若干子图。对于某一种分割将数据集$A$分为了$k$个组:$A_1, A_2, ... A_k$，对于这种Cut的损失记作:

$$cut(A_1, A_2, ... , A_k) = \frac{1}{2} \sum_{i=1}^k W(A_i, \bar{A_i})$$

最小化该损失，即最小化将图分割成$k$个子图所需断开的边权和。

基于图论的聚类方法是这样的一个思想，其中$W$为相似性，在图中找到一种切割方法，使得切割后的各个组之间相似性很小，而组内数据之间相似性很大，即找到合理的簇划分，使得

$$min \quad cut(A_1, A_2, ... , A_k).$$

假设$F_{n \times k}$是聚类指示矩阵，表示第i个样本被分在了第j类。
$$
    F_{i, j} = \left\{
      \begin{array} {rcl}
          1 & if ~ i_{th} ~ sample ~ in ~ j_{th} ~ cluster \\
          0 & others
      \end{array}
  \right..
$$

优化上式最小割问题等价于优化如下式子

$$\min_F \quad trace(F^T L F)$$

Proof：
这里假设$h$是$F$的第$i$个列向量，即第$i$个类的指示向量，对于第i个分割，有以下式子
$$
\begin{align}
 W(A_i, \bar{A_i}) = \sum_{i,j = 1}^n W_{i,j}(h_i - h_j)^2.
\end{align}
$$

由$f$的定义很容易理解，只有当一条边的两点满足一点在簇内，一点在簇外，即$h_i \neq h_j$时才会统计其切割的损失。而其对于每个$i$相加，可以得到上述式子。

tips：在图中，最小割问题和最大流问题可以互相转换(虽然没啥用)。


<h3>RatioCut</h3>

最小割虽然十分理想，但是其非常容易从孤立点处分割（因为孤立点出边权和较为稀疏，cut损失较小）(此处应有图)，会对聚类平衡性的影响很大。

为此，RatioCut和NCut是两种不同的割图损失被提出出来。

$$RatioCut (A_1, A_2, ... , A_k) = \frac{1}{2} \sum_{i=1}^k \frac{W(A_i, \bar{A_i})}{|A_i|} = \frac{1}{2} \sum_{i=1}^k \frac{cut(A_i, \bar{A_i})}{|A_i|} $$

其中$|A_i|$为每一类的元素数目，可以看出其可以抑制只割出少量点的特殊解，而偏好取得更加均匀的解。在上述关于min-Cut的讨论中，我们已经得到了$W(A_i, \bar{A_i})$的表达式，这里从$\{A_i\}$中取出某一项记作$A$；而在这里设计$h$的取值
$$
  \begin{equation}
    h_i = \left\{
      \begin{array} {rcl}
          \frac{1}{\sqrt{A}} & v_i \in A \\
          0 & v_i \notin A
      \end{array}
  \right..
  \end{equation}
$$
对于建图能量损失$h^TLh$，有
$$h^TLh = \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N W_{i,j}(h_{i} - h_{j})^2 
= \frac{1}{2} \sum_{i \in A, j \in \bar{A}}^N 2 W_{i, j} \frac{1}{|A|} = \frac{cut(A, \bar{A})}{|A|}$$
这是对于二分类任务而言的，对于多分类任务，只需将其按照$\{A_i\}$累加即可。

所以$RatioCut$问题可以转化为优化下式
$$RatioCut (A_1, A_2, ... , A_k) = \sum_{i = 1}^k h^TLh = \sum_{i = 1}^k (F^TLF)_{ii} = tr(F^TLF) \\
s.t. \quad elements~~ of~~ h ~~ satisfy ~~ equ.(2)$$
其中$h = F_i$。因为每一个$h_i$的取值有两种可能，暴力求解有$k*2^n$种情况，无法在多项式时间内求解出。所以考虑对条件进行放松，将离散情况的“选择”变为连续条件的“优化”，容易看出equ. (2)对$h$的约束有每个$h$正交且$h^Th = 1$，所以将其约束写为$F^TF=I$，所以RatioCut问题对如下式子进行优化：
$$
\begin{equation}
\min_F \quad trace(F^T L F) \quad
s.t. \quad F^TF = I
\end{equation}
$$

这里通常采用最小的$k$个特征向量来堆叠起来作为$F$。为什么要这样呢？有两种解释：
其一用到了瑞丽熵
在equ.(3)中采用拉格朗日乘子，得到$trace(F^TLF) - \lambda_1 F^TF$
<!-- 这里正交条件约束了$F$。假设$L$的特征值和单位特征向量分别为$\{\lambda_1, \lambda_2,...,\lambda_N\}$和$\{\alpha_1, \alpha_2,..., \alpha_N\}$，由于$L$对称，所以$\{\alpha_i\}_{i=1..N}$正交。由矩阵论可以得知，对于任意单位向量$\alpha$，都可以将其拆成特征向量之和的形式，即$\alpha = k_1\alpha_1 + k_2\alpha_2 +...+k_N\alpha_N$，其中满足$k_1+k_2+...+k_N = 1$ 可以认为 -->
可以认为$F$


<h3>NCut</h3>

$$NCut (A_1, A_2, ... , A_k) = \frac{1}{2} \sum_{i=1}^k \frac{W(A_i, \bar{A_i})}{vol(A_i)} = \frac{1}{2} \sum_{i=1}^k \frac{cut(A_i, \bar{A_i})}{vol(A_i)}  $$

这里的$vol(A_i)$是$A_i$里所有边的度之和，即$vol(A_i) = \sum_{x \in A_i} d_{xx}$。这里采用内部的权值之和来进行约束，又是一种特殊的抑制特解的方式。

## Lists



### Task list

- [x] Spectral Clustering
- [ ] Multi-view Clustering
  - [ ] One Problem
  - [ ] the other Problem
  - [ ] another Problem