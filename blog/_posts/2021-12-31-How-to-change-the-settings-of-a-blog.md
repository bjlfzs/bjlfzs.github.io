---
title: 如何修改Jekyll中的设置
author: theq Yang
date: 2021-12-31 17:00:00 +0800
categories: [Blogging, Article]
tags: [Techniques]
math: true
mermaid: true
---

## 如何修改Jekyll中的设置

分为avatar（博客头像），information（简介）和favicon（网页标题头像），其修改在_config.yml中去修改设置，其中对于我这个模板，与文字有关的设置分别为title和tagline；其中":"后面需要加空格，否则会发生报错。
<!-- 或用 '' 将文本括起来(容易成为报错的原因)。 -->

博客的头像和图片类似，在图床上插入图片并把链接copy到avatar后面。关于网页标题头像则需要在.\assets\img\favicons中修改，其参考官方文档[link0](https://chirpy.cotes.info/posts/customize-the-favicon/)即可，仅需要Generate the favicon即可。

其余的改动仍在研究，上面主要是参照[link1](https://www.jianshu.com/p/5425e77263ac)做出的尝试。

