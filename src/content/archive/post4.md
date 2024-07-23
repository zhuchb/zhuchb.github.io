---
title: "那一堆名字很像的audio SSL的区别是什么"
description: "CPC, wav2vec, vq-wav2vec, wav2vec 2.0, HuBERT, w2v-BERT"
pubDate: "July 23 2024"
heroImage: "/blog_images/anygpt.png"
tags: ["SSL"]
---

### CPC
元祖。所谓的contrastive loss，当前token和随便拿来的token我判断是不是属于同一个句子。如果是同一个句子后面的几个token的话，应该和我现在的token越像越好，maximize mutual information。反之，如果是从一个其他的句子拿过来的token，那么应该不像才对。
### wav2vec和CPC的区别
把SSL输出的向量用作ASR的input。架构和CPC是一样的。
### vq-wav2vec和wav2vec的区别
在wav2vec的架构基础上，用CNN从audio抽出特征之后做vector quantization (vq)。把语音变成有限个的特征，所以可以拿来做BERT的训练。但是需要2个阶段，vq-wav2vec和BERT。
### wav2vec 2.0 和wav2vec的区别
wav2vec 2.0 CNN后做vq，模型用Transformer。wav2vec无vq，CNN后用的是GRU。
wav2vec 2.0 还会做mask，可以看masked token前后来预测当前token。wav2vec用的是GRU，所以是看前面的token预测当前token，是单向的。
### HuBERT
masking + transformer。Loss用的是predictive loss，和上面的loss不一样，不属于一个家族。因为audio不像text有词表，所以就用k-means自己造一个词表pseudo label出来，然后像BERT一样预测pseudo label。
### w2v-BERT
Contractive loss和predictive loss我全都要.jpg。还是end-to-end的