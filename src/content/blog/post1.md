---
title: "论文笔记 AnyGPT:Unified Multimodal LLM with Discrete Sequence Modeling"
description: "Cluster 1: Intro to Deep Learning"
pubDate: "July 18 2024"
heroImage: "/blog_images/anygpt.png"
tags: ["Multimodal", "LLM"]
---

## 总结文章内容
[文章的github主页](https://junzhan2000.github.io/AnyGPT.github.io/)

这篇文章做了什么：
* 做了一个可以输入和输出 speech, text, image和music四种模态的GPT。

创新点在哪里：
* 文章称其创新点在于可以利用现有的LLM架构，对于多模态的处理只是各种数据的预处理，所以可以方便地整合新的模态。（该文章的LLM架构使用的是LLaMA-2）。
* 文章用GPT-4和LLM做了一个包含上述speech, text, image和music四种模态的multi-turn conversations的数据集。

表现如何：
* 在Image Understanding, Image Generation, ASR, TTS, Music Understanding and Music Generation tasks的指标上与各个领域的state-of-art模型进行对比。一句话概括就是在各个领域表现不是最好的，但是强在各种任务它都可以做，表现也还不错。


## Introduction
文章的意义在哪里：

* 有一部分的LLM只实现了了text + x (image or audio)两种模态的理解和生成，3种以上的模态是比较困难的
* 有一部分LLM实现了3种以上模态的理解和生成，但有各自的问题。1.缺少一个稳定的LLM，导致系统的推理能力低下。 2. 使用的encoder和decoder不一致，导致训练和预测变得困难。

AnyGPT可以解决这些问题。

## Methodology
架构如下图。每种模态的数据使用不同种类的tokenizer与detokenizer，使得每种模态的数据变成LLM可以处理的离散序列。LLM上的训练目标为next token prediction. 同时将其他模态的token加入到LLM的词表中，使得LLM可以预测这些token并输出text以外的模态。
<div class="mermaid">
graph TD;
    D[speech/image/music]-->A;
    A[\Tokenizer/]-->B[LLM];
    B-->C[/Detokenizer\]-->E[speech/image/music];
</div>

speech/image/music模态使用的tokenizer分别来自不同的论文。

### Image Tokenizer: SEED
[Making llama see and draw with seed tokenizer]("https://arxiv.org/pdf/2310.01218") ("See" and "D"raw)

这篇文章的目的是在LLM中处理text和image两种模态。
![Seed_figure2](/post1/SEED-Figure2.png "Figure 2")
![Seed_figure3](/post1/SEED-Figure3.png "Figure 3")

### Speech Tokenizer: SpeechTokenizer
[SpeechTokenizer: Unified Speech Tokenizer For
Speech Language Models]("https://arxiv.org/pdf/2308.16692")

这篇文章的目的是把semantic 和acoustic token结合起来。可以理解为semantic token更多关于说话内容的信息，acoustic token更多关于声音波形的信息。

> Semantic tokens are typically from self-supervised
pre-trained models with masked language modeling as training objective (Hsu et al., 2021; Baevski et al., 2020; Chung et al., 2021)

> Acoustic tokens can be extracted from neural audio codecs with reconstruction as training objective (Zeghidour et al., 2021; Défossez et al., 2022)

![Seed_figure2](/post1/speech-figure2.png "Figure 2")

### Music Tokenizer: Encodec

[High Fidelity Neural Audio Compression]("https://arxiv.org/abs/2210.13438")

这篇文章主要的创新点是设计了一个loss，包含了频谱上的construction loss, RVQ的loss，还有对抗式的discriminator的loss。

估计是考虑到music和speech相比acoustic的信息更加丰富，使用的是对应上面写到的acoustic tokenizer的Encodec。

![Encoder_figure1](/post1/encodec.png "Figure 1")

### Multimodal Generation

因为LLM处理在semantic level，并且为了减少LLM需要处理的序列，对于image和speech的生成还要外加一些生成模型。

**image:**
<div class="mermaid">
graph TD;
 A[LLM]-->B([token]);
 B([token])-->C[Diffusion Model];
 C-->D([image]);
</div>

**speech:**
<div class="mermaid">
graph TD;
 A[LLM]-->B([semantic token]);
 B-->C[SoundStorm];
 C-->D([acoustic token]);
 B-->E[SpeechTokenizer's Decoder];
 D-->E[SpeechTokenizer's Decoder];
 E-->F([audio])
</div>

## Data
数据集的构造过程。从topic扩写到multi-turn conversation。并给LLM提供具体的包含了多种模态的例子。扩写的conversation中speech，music和image先用text描述，再合成出来。

<div class="mermaid">
graph TD;
    A([100 meta topics])-->B[GPT-4];
    B[GPT-4]-->C([20,000 specific topics]);
    C-->D[LLM];
    H([examples])-->D;
    D-->E([specific dialogue scenarios]);
    E-->F[GPT4];
    F-->G([multi-turn conversations in text form]);
    G-->I[generative models];
    I-->J([multi-turn conversations with multimodal elements]);

</div>



<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>

