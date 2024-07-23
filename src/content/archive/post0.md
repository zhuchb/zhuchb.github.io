---
title: "学习笔记 Deep Learning for Visual Data"
description: "Cluster 1: Intro to Deep Learning"
pubDate: "Jun 23 2024"
heroImage: "/blog_images/IMG_9083.jpg"
tags: ["Deep Learning", "Notes"]
---

前言
---
最近通过看在线课程 [CS-198-126-Deep-Learning-for-Visual-Data](https://ml-berkeley.notion.site/CS-198-126-Deep-Learning-for-Visual-Data-a57e2aca54c046edb7014439f81ba1d5) 复习了一下深度学习和CV的一些知识。想做一下笔记记录一下里面的知识点，便于自己复习。所以我只会挑一部份知识点，有一些理解和解释会借助chatgpt。这个课程偏向intuitive，只是给初学者一个大概的直观的印象，另外知识点也很新，包括很多现代的神经网络架构都有介绍。但是只通过这个课程想深入掌握所有知识点还有点不够。这篇笔记目前是写给我自己复习用的，用我自己的大白话解释一下课程里的知识点。课程里还有一些没讲清楚很快就过的知识点，我自己感兴趣的会查资料写下自己的感想。如果有理论性的错误还请见谅。

第一课：What is Machine Learning?
---
* 什么是Machine Learning (ML)?
    * Machine Learning是通过数据学习到某种函数的方法论
    * 什么样的“函数”？→ 通过你选择的model class决定的
    * 函数的细节是怎么样的？→ 通过学习而决定模型的parameters
* ML的分类：Supervised, Unsupervised, Reinforcement
* 一些概念
    * Function/Model: 你的ML模版
    * Weights/Biases: ML模型中通过数据学习的learnable parameters
    * Hyperparameter: 在训练模型前提前手动决定的超参数 non-learnable parameter
    * Loss Function/Cost Function/Risk Function
    * Feature: 输入模型的的数据特征。
    * One Hot Labeling: 比如[0,1,0]代表这是第2个类的标签。
    * Probability distribution: 比如[0.15, 0.75, 0.1]代表这个sample是第1个类的可能性是15%, 第2个类的可能性是75%, 是第3个类的可能性是10%。
    * Augmenting the Data: 当数据不够的时候通过对数据做一些变换“造”出一些新的数据，以增加数量，但是这些“造”出来的数据不能改变原数据的语义。
* Bias/Variance
    * Bias: 偏差是指模型在训练数据和测试数据上表现出系统性误差的程度。反映了模型对数据的拟合能力。高偏差的模型通常过于简单，无法捕捉数据的复杂模式。
    
        比如我们想预测A，实际上A被因素B, C, D所影响，但是我们预测的时候只考虑了B，那么即使在训练数据上，预测值和实际值之间的差距也很大，表现为高偏差，欠拟合(Imderfitting)。
    * Variance: 方差是指模型对训练数据的敏感程度，即模型在不同的训练数据集上表现出多少变化。高方差的模型通常过于复杂，能很好地拟合训练数据，但对新数据的泛化能力较差，容易出现过拟合（Overfitting）

        比如我们某个模型，它在训练集上表现的很好，但是遇到新的测试数据时效果就很差，表现为高方差。
* Overfitting/Underfitting
    * Underfitting(欠拟合): 模型在训练和验证集上都表现得很差。可以考虑使用更复杂的模型。
    * Overfitting(过拟合): 模型在训练及上表现得很好但是在验证集上表现得很差。可以考虑降低模型的复杂度。

第二课: Deep Learning 1
--
* Gradient（梯度）: 偏微分的向量表现形式。在梯度方向上，函数值的变化最快。所以想让函数值快速变小的话，应该沿着负梯度的方向走。
* Perceptron(感知机): 多个输入，一个输出。激活函数为ReLU时，数学公式如下。字母头上带杠$\bar{x}$意思是它是一个向量。$$f(\bar{x}, \bar{w}, b) = ReLU(w_1x_1 + w_2x_2 + ... + w_nx_n + b) = ReLU (\bar{w}^\top\bar{x} + b)$$ $x$: input, $w$: weights, $b$: bias。
* Perceptron Layer: 对于相同的输入，有n个perceptron，且每个Perceptron的输出成为下一层的输入。$W$为矩阵，$\bar{b}$为向量。每组代表每个perceptron的参数。$f(\bar{x}, W, \bar{b})$输出向量，即每一个perceptron的输出。
$$f(\bar{x}, W, \bar{b}) = ReLU (W\bar{x} + \bar{b})$$
* Multi-Layer Perceptron(多层感知机): 就是上面的Perceptron Layer堆叠好几层而成的一种神经网络。
* Classification: 最后的输出要表明你这个input是属于哪个类别的，比如输出[0.15, 0.75, 0.1]，代表这个sample是第1个类的可能性是15%, 第2个类的可能性是75%, 是第3个类的可能性是10%。这个时候需要用到softmax。
