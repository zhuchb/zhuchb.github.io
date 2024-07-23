---
title: "论文笔记 UniAudio: An Audio Foundation Model Toward Universal Audio Generation"
description: "Notes about the paper UniAudio"
pubDate: "July 23 2024"
heroImage: "/blog_images/UniAudio.png"
tags: ["Audio Generation", "LLM"]
---
## 总结文章内容

这篇文章做了什么：
* 把语音生成的很多任务，如TTS， VC， Singing voice synthesis(SVS) 用一个模型全部解决。
> which intentds to accomplish multiple audio generation tasks with only one unified model.

创新点在哪里：

* 对于audio generation，很多LLM-based的方法，但是都是为了单个task训练一个模型。该文章想用一个模型解决多种task
> most existing LLM-based works are still designed for single tasks
* 方法是把audio或者其他task tokenize，然后把source和target拼接成一个single sequence丢进LLM做next-token prediction.

表现如何：
* 做了11个audio generation task。
> supports 11 audio generation tasks: the training stage includes 7 audio generation tasks, while 4 tasks are further added in the fine-tuning stage.
* 指标上来说，在有一些任务上比那个任务专用的模型好一点，有一些就差一点。
* 在audio sequence的输入和预测上，本文不是将RVQ的离散token完全平铺，而是用了一个multi-scale Transformer的架构，可以理解为一种hierarchical的架构吧。目的是为了降低计算复杂度，提高效率。flatten是生成质量最好的，文章称multi-scale Transformer在生成质量和效率上的平衡更好。
> We claim that the proposed multi-scale transformer is an auto-regressive architecture that achieves a better trade-off between generation quality and efficiency.


## Methodology

### 各种modality的quantization方法

| Modality | Token Sequence|
| ----------- | ----------- |
| audio (speech, sounds, music, singing) | RVQ |
| phoneme | phoneme sequence + duration(if possible) |
| MIDI | frame-level F0 sequence|
| text | continuous embeddings derived from pre-trained text LLM|
| semantic token | continuous embeddings by audio SSL models|

### sequence组合为输入的例子

`<start><sound_task><text_start>text_sequence<text_end><audio_start>audio_sequence<audio_end><end>`

### Multi-scale Transformer

该文章觉把audio token的RVQ离散向量全部平铺的话，整个处理的序列就会很长，使得Transformer的计算量太大。所以为了降低复杂度，使用了multi-scale Transformer。

这篇文章的RVQ是3层，所以可以理解每1帧的1个continuous audio representation由3个离散的token来表示。在multi-scale Transformer中，它先把3个离散token加起来（近似于continuous audio representation），输入global transformer中，输出一个可以代表这1帧的guidance token。

后面还有一个Local Transformer，它吃guidance token，但是预测RVQ的3个向量，相当于预测的时候又把它们分开。

audio以外的modality就把他们的embedding重复3次输入上述的multi-scale Transformer。
