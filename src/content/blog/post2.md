---
title: "论文笔记 AudioLM: a Language Modeling Approach to Audio Generation"
description: "Notes about the paper AudioLM"
pubDate: "July 19 2024"
heroImage: "/blog_images/audiolm.png"
tags: ["Audio Generation", "LLM"]
---

## 总结文章内容

这篇文章做了什么：
* 用SoundStream抽出acoustic token, 用w2v-BERT抽出semantic token，把两者结合丢进LM。 

创新点在哪里：

* 用一种hierarchical framework把acoustic token和semantic token结合起来，使得生成的语音同时具有high audio quality & long-term consistency.

表现如何：
* 做了2个下游任务：Speech continuation 和 Piano continuation。在Speech continuation上，生成的语音可以保持prompt的speaker identity。生成的语音再跑一次ASR，WER/CER指标与单一语者的语音合成系统差不多。在Piano continuation上，结合使用acoustic token和semantic token比仅使用acoustic token的效果要好，说明semantic token对于捕捉long-term structure是有帮助的。

## Introduction
文章的意义在哪里:
* 基于ar, adversarial training的方法生成的语音信号质量很高，但在缺少strong conditioning的情况下生成的是unstructured audio (不具有特定语义内容或明确结构的音频); 基于LM方法生成的语音有连续的内容，但是训练在clean speech上并且局限于single speaker。
* 大概意思就是，两种大方向的语音生成各有各的优点和缺点，我就把他们结合起来，这样就可以同时拥有他们的优点了（我全都要.jpg）。

> In this work, we introduce AudioLM, a framework that enables high-quality audio generation with long-term coherent structure.

> AudioLM tackles both challenges of long-term coherence and high-quality by combining semantic and acoustic tokens in a generative framework.

## Methodology

大的架构是Encoder-LLM-Decoder模式。

<div class="mermaid">
graph TD;
    A([audio])-->B[\Tokenizer/];
    B[\Tokenizer/]-->C([token]);
    C-->D[Language model];
    D-->E([token]);
    E-->F[/Detokenizer\];
    F-->G([audio]);

</div>

如何通过semantic token和acoustic token生成audio，有3个阶段。大概的流程是，生成semantic token，然后用semantic token生成coarse acoustic token，也就是 SoundStream RVQ的前半部分。再用coarse acoustic token生成fine acoustic token，也就是SoundStream RVQ的后半部分。

1. 先根据之前时间的semantic token预测当前时间的semantic token; 
<div class="mermaid">
graph TD;
    D([prompt])--mapping-->A
    A([当前时间t以前的semantic token])-->B[w2v-BERT];
    B-->C([当前时间t的semantic token]);
</div>

2. 用semantic token作为condition，结合当前时间的coarse acoustic token， 输入coarse acoustic model。

这篇文章里SoundStream的RVQ的quantizer有Q个。每个quantizer都有一个输出。那么coarse acoustic token就是取前$Q^{'}$个quantizer的输出。实验中Q=12, $Q^{'}$=4。所以coarse acoustic token = {q1, q2, q3, q4}, fine acoustic token = {q5, q6, ..., q12}。文章称coarse acoustic token包含speaker identity 和recording conditions的信息。fine acoustic token则可以提高音质。

<div class="mermaid">
graph TD;
    A([vector])--residual-->B[quantizer];
    B-->E([q1]);
    B--residual-->C[quantizer];
    C-->F([q2])
    C--residual-->D[quantizer];
    D-->G([qn])
</div>

semantic token和coarse acoustic token的使用如下图。
 
 <div class="mermaid">
graph TD;
 subgraph concatenation
    direction LR
    A([整个序列的semantic token]) x--x D([当前时间t以前的coarse acoustic token]);
 end
 concatenation-->B[Decoder-only Transformer];
 B-->C[当前时间t的coarse acoustic token];
 </div>


3. coarse acoustic token和fine acoustic token的使用如下图。最后再concatenate整个序列的coarse acoustic token和fine acoustic token，丢进SoundStream Decoder就可以生成语音啦。
 <div class="mermaid">
graph TD;
 subgraph concatenation
    direction LR
    A([整个序列的coarse acoustic token]) x--x D([当前时间t以前的fine acoustic token]);
 end
 concatenation-->B[Decoder-only Transformer];
 B-->C[当前时间t的fine acoustic token];
 </div>

## 

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>