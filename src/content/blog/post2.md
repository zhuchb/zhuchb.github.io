---
title: "拆解Grad-TTS代码"
description: "Read Grad-TTS codes"
pubDate: "Oct. 4 2024"
heroImage: "/blog_images/Grad-TTS.png"
tags: ["Diffusion", "TTS"]
---

## Grad-TTS

github在这里。这是我看的第一个diffusion-based TTS的代码，本文是我一边看一边记得一些笔记。
https://github.com/huawei-noah/Speech-Backbones/tree/main

- train.py - 训练

- inference.py -顾名思义inference

- data.py - 处理数据集。将音频转换为梅尔频谱。train和inference用到

- diffusion.py - 定义Diffusion class。tts.py需要用到

- tts.py - 定义GradTTS class。作为train和inference和generator

- text_encoder.py - 主要定义了DurationPredictor和TextEncoder。来自GlowTTS的代码。

## Inference.py


intersperse: 序列变为原来的两倍，间隔地插入text的元素，间隔位置是len(symbols) 
为什么？？

text_to_sequence: 根据CMU词典把text转换成各个音素或者符号的ID的sequence
len(symbols): 词典的大小。这个词典包括arphabet和各种可能的符号。
```python
text = "some text"
x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))
```

Grad-TTS生成, y_dec是decoder的输出。最后通过vocoder把梅尔频转换成wav


## train.py

```python
#load data。来自data.py
#batch_collate用的是一个自己写好的BatchCollate
train_dataset = TextMelDataset(train_filelist_path,...)
batch_collate = TextMelBatchCollate()
#定义模型
model = GradTTS(...)
```

每个epoch进行train和eval。因为param.save_every被设为了1.

n_epochs是10000

```python
#当epoch是params.save_every的倍数的时候，才会执行后面的eval和保存checkpoint
#这是为了一定时间执行一次eval和save
if epoch % params.save_every > 0:
    continue
# tests_batch只有一个
    for i, item in enumerate(test_batch):
        model.eval()
```

训练过程中有3个Loss：dur_loss, prior_loss, diff_loss

总的loss是三个loss相加
```python
loss = sum([dur_loss, prior_loss, diff_loss])
```


**梯度裁剪 gradient clipping**

max_norm=1设置了梯度的最大范数(norm，表示某种距离)。如果L2_norm(grad) > max_norm，则按比例缩小所有梯度，使总范数等于 max_norm。通常在反向传播之后、优化器步骤之前执行。有助于防止训练不稳定。
```python
enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),max_norm=1)
dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),max_norm=1)
```


## text_encoder.py

**Duration predictor**

和FastSpeech一样。输出的是log。是对每个音素进行长度预测。

*The duration predictor is composed of two convolutional layers with ReLU activation, layer normalization, and dropout followed by a projection layer.*


**Text Encoder**

From FastSpeech2:

*The encoder converts the **phoneme embedding sequence** into the **phoneme hidden sequence**, and then the **variance adaptor** adds different variance information such as duration, pitch and energy into the hidden sequence, finally the **mel-spectrogram decoder** converts the adapted hidden sequence into mel-spectrogram sequence in parallel.*

**Encoder**

感觉是一个Transformer模块。这只是1个layer的模块，有n的layer堆叠在一起


graph TD;
A[Attention] --> B[Drop];
B --> C[LayerNorm];
C --> D[Feed-Forward];
D --> E[Drop];
E --> F[LayerNorm];


**TextEncoder**

graph TD
A[embedding] --> b1[Conv1d];
subgraph prenet-ConvReluNorm
b1 --> b2[LayerNorm];
b2 --> b3[ReLU];
b3 --> b4[Dropout];
b4 --> b5[residual];
end
A --> b5;
b5 --> C[encoder];
C --> D[Conv1d];
C --> E[DurationPredictor];
E --> F[时间预测];
D --> G[梅尔频谱];

## tts.py

定义了GradTTS

**GradLogPEstimator2d**

用于预测 $\bigtriangledown logp_t(X_t)$ ，是U-Net架构。

**get_noise**

t=0时的噪声：$\beta_0$

t=1时的噪声：$\beta_T$

$t\in[0,1]$

t时刻的noise：$\beta_t = \beta_0 + (\beta_T - \beta_0)*t$ 线性插值

从0到t累积的噪声：$\int_0^t\beta_t dt= \beta_0 t + \frac{1}{2}(\beta_t - \beta_0)t^2$ 对上面式子的积分

**forward_diffusion**

给定$x_0$和时刻$t$，最终高斯分布的均值$\mu$，求$x_t$

$mean = x_0 e^{\frac{1}{2}\int_0^t\beta_tdt} + \mu (1 - e^{\frac{1}{2}\int_0^t\beta_tdt})$

$variance = 1 - e^{-\int_0^t\beta_tdt}$

$x_t = mean + z\cdot \sqrt{variance}$

z是一个随机数（噪声），所以$x_t$是在从高斯分布中采样

**reverse_diffusion**

input

mu： Encoder的初步估计

z: mu添加噪声得到的decoder的输入

n_timesteps = 50 diffusion的步数

时间步长 $h = 1/timesteps$

stoc： true的时候使用SDE，false的时候使用ODE？

梯度$\bigtriangledown logx_t$是通过UNet GradLogPEstimator2d估测出来的

**SDE的表达式**

$dX_t = (\frac{1}{2}(\mu - x_t) - \bigtriangledown logp_t(X_t)) \cdot \beta_t \cdot \frac{1}{timesteps} + \sqrt{\beta_t} \cdot random \cdot\sqrt{\frac{1}{timesteps}}$

$random \cdot\sqrt{\frac{1}{timesteps}}$ 相当于原文的 $dW_t$

$dW_t$ 变成了方差为h的高斯噪声

**ODE的表达式**

$dX_t = \frac{1}{2}(\mu - x_t - \bigtriangledown logp_t(X_t)) \cdot \beta_t \cdot \frac{1}{timesteps} $

$\frac{1}{timesteps}$相当于原文中的dt。注意这里的噪声和forward不一样用的是当前时刻的噪声，非累积噪声。

$X_t = (X_t - dX_t)*mask$

**compute_loss**

随机生成时间步t，计算loss

**loss_t**

根据论文，因为$X_t$是高斯函数采样得到的，mean和variance也都是设计好的。所以它的$\bigtriangledown logp_t(X_t)$ 可以直接通过高斯函数的表达式计算出来

$variance = 1 - e^{-\int_0^t\beta_tdt}$

$x_t = mean + z\cdot \sqrt{variance}$

$cum-noise = \int_0^t\beta_t dt$

所以这个estimator用来预测noise了？score的预测和noise的预测是等价的？

对于一个正态分布，它的score function由一下式子计算

$\bigtriangledown xlog(x; \mu, \Sigma) = -\Sigma^{-1} (x-\mu)$

论文中$\Sigma = I$

所以

$ \begin{aligned}\bigtriangledown xlog(x) &= -(x-\mu)\\&= -(mean + z\cdot \sqrt{variance} - mean) \\ &= - z \cdot \sqrt{variance} \\ &= -z \cdot \sqrt{1 - e^{-\int_0^t\beta_tdt}} \end{aligned}$

GradLogPEstimator2d其实是在预测-z，然后乘上根号项。

<span style="color: red; ">为什么不是estimator的输出直接和-z算距离？？根据EMD，直接预测乘了根号项的noise scale太大，不利于网络的训练。</span>

loss函数算的是GradLogPEstimator2d的预测值和-z的距离，所以是$(noise - (-z))^2$ ，并且分母做归一化

```python
def loss_t(self, x0, mask, mu, t, spk=None):
    #对给定的起始x0和随机时间步t，
    #返回forward_diffusion到这一步时的xt还有随机采样时的噪声z
    xt, z = self.forward_diffusion(x0, mask, mu, t)
    time = t.unsqueeze(-1).unsqueeze(-1)
    #当前时间步t的累积噪声
    cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
    #当前时间步推测的score
    noise_estimation = self.estimator(xt, mask, mu, t, spk)
    noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
    loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
    return loss, xt
```

**compute_loss**

- duration loss

        encoder首先输出一个预测的梅尔频谱和duration。（音素级别的？）

通过Monotonic Alignment Search (MAS)算出 text embedding **mu_x**和梅尔频谱**y**的对应关系，通过对齐矩阵获取每个音素的duration。这是ground truth.

计算MAS的时候要提供一个log prior distribution。这个高斯分布的期望是**mu_x**, 方差为**1***, D是数据的维度，也是梅尔频谱的维度, 在这里是80.

$p(y|\mu_x) = \frac{1}{(\sqrt{2\pi})^D}e^{-\frac{(y-\mu_x)^2}{2}} $

$\begin{aligned} log p(y|\mu_x) &= -0.5(y-\mu_x)^2 + (-0.5)Dlog2\pi \\ &= -0.5 (y^2 -2\mu_xy + \mu_x^2) + (-0.5)Dlog2\pi\end{aligned}$

计算encoder预测的duration和 L2 loss

- diffusion loss
  
  将梅尔频谱切成长度为2s，随机选择一些片段进行训练。由decoder.compute_loss算出
  
- prior loss
  
  encoder给的初步预测也要和实际的梅尔频谱y接近
  
  mu_x: encoder给出的音素text embedding。通过与MAS算出的attn对应关系相乘，可以得到长度伸缩的mu_y
  
  mu_y: 每一个音素字符对应的梅尔频谱长度 [batch_size, mel_length, encoder_channels]
  
  mu_y与实际的y计算L2 loss
  

train.py把3个loss相加，反向传播

## data.py

处理数据输入

train.txt, valid.txt, test.txt

wav path|text

**class: TextMelDataset**

dataloader会根据其中____getitem____ 函数来获得数据

**get_mel**

输入语音返回80维梅尔频谱(80, F), F-> 梅尔频谱采样的时间步数，根据语音的长度和hop_length计算出来的

**get_text**

首先是text_to_sequence

输入text，返回每个单词拆分为arphabet后，每个arphabet的序号。

例如, 输入text = "Nice to meet you" 输出 [119, 86, 131, 11, 133, 141, 11, 118, 113, 133, 11, 145, 141], 11是空格的序号，nice对应的 N AY1 S序号是119, 86, 131

然后是intersperse，在上述序列的偶数位置插入len(symbols)=148, 输出[148, 119, 148, 86, 148, 131, 148, 11, 148, 133, 148, 141, 148, 11, 148, 118, 148, 113, 148, 133, 148, 11, 148, 145, 148, 141, 148]

插入的元素通常代表一个空白符号或静音，插入到每个音素之间，以便模型可以学习音素之间的过渡。

**get_arpabet**

输入单词，返回{arphabet}，例如：nice, {N AY1 S}

**sample_test_batch**

输入size，随机抽取一组size大小的测试数据，返回test_batch (size, {y, x})

y通过get_mel得到梅尔频谱(80, F)

x通过get_text得到的(, N) N个字符序列的1维数组

**class: TextMelBatchCollate**

输入batch。

size是params中定义的是16，每个batch包含了文本x和语音y的内容。这个函数提取这个batch中的x_max_length和y_max_length， 对x创造张量(batch_size, n_feats, x_max_length), 对y创造张量(batch_size, n_feats, y_max_length)。x和y在张量中左对齐填充，剩余部分保持为0.

返回{'x': x, 'x_lengths': x_lengths, 'y': y, 'y_lengths': y_lengths}

x_lengths, y_lengths包含了原始序列的长度

<script type="module">
  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
  mermaid.initialize({ startOnLoad: true });
</script>