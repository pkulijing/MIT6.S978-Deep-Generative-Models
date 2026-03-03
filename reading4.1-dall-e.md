# MIT 6.S978 Reading 4.1 [DALL-E: Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)

## 1. 论文的动机: The Bitter Lesson

在DALL-E之前，文本到图像生成作为多模态AI的重要任务，主要依靠三种策略来提升性能：

1. **架构改进**：多尺度生成器、注意力机制整合
2. **辅助损失**：文本-图像匹配损失、对抗损失的组合优化
3. **额外条件信息**：物体部位标签、分割掩码等结构化监督信息

然而，这些传统方法背后的数据集却很小，训练出的模型也很小。作者提出了一个问题：已经在其他领域获得成功的scaling模式：更大的数据量、更大的参数量、更大的计算量，配合简单的模型架构，是否有可能取得更好的效果？

本文基于来自互联网数据的2.5亿个图片-文本对（作为对比：MS-COCO仅330K 图文对、CUB-200仅11K 图文对），训练120亿参数的自回归Transformer。优势是碾压性的：效果吊打之前基于小数据集训练的传统模型，而且是zero-shot的，根本不需要MS-COCO数据集的知识；模型还涌现出了图像翻译等能力。答案是残酷的，[The Bitter Lesson](https://en.wikipedia.org/wiki/Bitter_lesson) 又一次应验了。

## 2. 论文的前序知识

### 2.1 Byte Pair Encoding (BPE)

BPE (Sennrich et al., 2015) 是处理文本的子词标记化方法，在 DALL-E 中用于将自然语言转换为 token 序列，是现代大语言模型最常用的文本 Encoder.

#### **2.1.1 BPE 算法原理**

1. 初始化词汇表为所有字符
2. 重复以下步骤直到达到目标词汇表大小：
   - 统计所有相邻符号对的频率
   - 将频率最高的符号对合并为新符号
   - 更新词汇表

#### **2.2.2 BPE Dropout**

BPE 的编码不是在 DALL-E 训练时一起训练的，而是预先训练好、可以直接用的。但这个预制的BPE编码模块支持 BPE Dropout (Provilkov et al., 2019)：随机跳过某些合并操作，产生不同的子词分割，增强模型鲁棒性。DALL-E 使用了10%的 BP Dropout.

### 2.2 离散随机变量的梯度估计

离散潜变量的梯度估计是 DALL-E 第一阶段训练用的离散 VAE 的核心技术挑战。对于离散分布，传统的重参数化技巧不可用。本文中使用的是 Gumbel-Softmax 松弛技巧 (Jang et al., 2016; Maddison et al., 2016)。

#### 2.2.1 Gumbel 分布

对于服从均匀分布的随机变量 $u\sim U(0,1)$，对其进行如下操作可以得到一个服从 Gumbel 分布的随机变量 $g\sim G(0,1)$:

$$
g=-\ln(-\ln(u))
$$

其概率密度函数和累积概率密度函数为：

$$
g(x) = e^{-(x+e^{-x})} \quad F(x)=e^{-e^{-x}}
$$

其概率密度函数的图像如下：

![Uploaded Image](https://www.genspark.ai/api/files/s/5kHzaOS6)

正态分布用于对常规现象、平均行为建模，Gumbel分布则用于对极端现象建模。如一个水文观测站没每年测到的最高水位的分布、一个气温测量点测到的当年最低温，就符合Gumbel分布。

#### 2.2.2 带温度参数的 Softmax 函数

在经典 softmax 的每一个 logits 上添加一个**温度参数**：

$$
y_i = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{K} \exp(z_j / \tau)}
$$

就得到了带温度的 softmax 函数，它可以用于将离散的 argmax 操作连续化，从而允许梯度回传。不同的 $\tau$ 会让该函数表现出不同的行为：

- $\tau\rightarrow\infty$: logits 的区别被抹平，几乎是均匀分布
- $\tau\rightarrow 0$: logits 的区别被放大，几乎是argmax

当梯度回传被离散的 argmax 操作阻断时，我们引入这个函数，不再做 argmax 操作，而是使用 $y_i$ 作为权重，将所有 token 求平均值:

$$
E_{out} = \sum_{i=1}^K y_i E_i
$$

注意这里只做了线性加权平均，没有依据 softmax 或者 argmax 采样了。这样离散的 argmax 操作成功实现连续化，梯度就可以顺利回传。而训练的效果通过所谓退火操作 (annealing)实现，即在训练过程中调整 $\tau$ 的值：

- 开始阶段很大，所有 token 的影响平均地回传，梯度快速更新
- 逐渐减小直至接近0，函数“硬化”为 argmax 操作，回传正确的 token 的影响

注意以上操作只在训练时进行，推理时直接做 argmax.

### 2.3 Transformer

#### **2.3.1 Transformer 自注意力机制**

Transformer 的核心是 scaled dot-product attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中：

- $Q = XW_Q$：query 矩阵
- $K = XW_K$：key 矩阵
- $V = XW_V$：value 矩阵
- $d_k$：key 维度，用于缩放防止 softmax 饱和

多头注意力：

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

#### **2.3.2 因果掩码 (Causal Masking)**

为保证自回归性质，需要阻止模型看到"未来"的 token。因果掩码将注意力矩阵的上三角设为 $-\infty$：

$$
M_{ij} = \begin{cases}
0 & \text{if } i \geq j \\
-\infty & \text{if } i < j
\end{cases}
$$

掩码后的 attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

#### **2.3.3 稀疏 Transformer**

标准 Transformer 的注意力复杂度为 $O(n^2)$，对长序列不现实。稀疏 Transformer (Child et al., 2019) 通过限制注意力模式降低复杂度。DALL-E 使用的三种稀疏模式：

1. **Row attention**：每个位置只关注同行和之前行的位置
2. **Column attention**：每个位置只关注同列之前的位置
3. **Convolutional attention**：每个位置关注局部邻域（如 11 x 11 窗口）

## 3. 论文的主要逻辑

### 3.1 整体框架：两阶段训练

#### 3.1.1. 概述

直接从像素出发做训练有两个问题：

- 参数量：对高分辨率图像，模型会太大，无法训练
- 学习目标：模型会学习过多的高频细节而非低频宏观结构，后者才是图像生成重点关注的

因此DALL-E 采用两阶段训练策略，整体架构简单而优雅：

- 第一阶段：采用离散 VAE 训练 encoder，将 3 x 256 x 256 的 RGB 图像压缩为 32 x 32 x 8192 的图像 token (8192为 token 字典的大小），这就将自回归的迭代步数、也就是 Transformer 的最大 context 长度缩减为原来的 1/192.
- 第二阶段：将 BPE 编码的 256 个文字 token 与 1024 个图像 token 做 concatenation，训练 1280 个token的自回归 Transformer, 建模文字与图像的联合概率分布。

后续讨论中使用如下符号： images $x$, 文本 token $y$, 图像 token $z$。

#### **3.1.2 变分推断视角**

从文本到图像的生成问题本质上是要最大化联合似然 $p(\mathrm{图像},\mathrm{文本})$。文本我们直接用BPE做 token 化，图像没有现成的 token 化可以用，因此文本可以直接用其 token $y$ 代表，图像则需要引入 token 变量 $z$。使用 VAE 中推导 ELBO 的方式，我们可以得到 $p(x,y)$ 似然最大化问题的 ELBO:

$$
\begin{aligned}
\log p(x,y)&=\int q(z)\log p(x,y)dz\\
&=\int q(z)\log\frac{p(x,y,z)}{p(z\mid x,y)}dz\\
&=\int q(z)\log\frac{p(x,y,z)}{p(z\mid x,y)}\frac{q(z)}{q(z)}dz\\
&=\int q(z)\left\lbrack\log\frac{p(x,y,z)}{q(z)}+\log\frac{q(z)}{p(z\mid x,y)}\right\rbrack dz\\
&=\mathbb{E}_{z\sim q(z)}\log\frac{p(x,y,z)}{q(z)}+D_{KL}(q(z), p(z\mid x,y)) \\
&\ge \mathbb{E}_{z\sim q(z)}\log\frac{p(x,y,z)}{q(z)}
\end{aligned}
$$

在对 ELBO 做进一步分解时，我们换一个视角：上述推导里 $z$ 是唯一的 latent 变量，但接下来我们把 $y,z$ 都视作 latent 变量：

$$
\begin{aligned}
\mathrm{ELBO}&=\mathbb{E}_{z\sim q(z)}\log\frac{p(x\mid y,z)p(y,z)}{q(z)}\\
&=\mathbb{E}_{z\sim q(z)}\log p(x\mid y,z) - \mathbb{E}_{z\sim q(z)}\log\frac{q(z)}{p(y,z)} \\
&=\mathbb{E}_{z\sim q(z)}\log p(x\mid y,z) - D_{KL}\left(q(z),p(y,z)\right)
\end{aligned}
$$

这样，我们成功得到了文本到图像生成问题涉及的三个分布：

- $p_{\theta}(x\mid y,z) = p_{\theta}(x\mid z)$：从图像 token 到图像的decoder，与 y 无关
- $q_{\phi}(z\mid x)$: 从图像到图像 token 的encoder，与 y 无关
- $p_{\psi}(y,z)$: 文本 token 和图像 token 的联合分布

前两个分布在第一阶段使用离散 VAE 建模，第三个分布在第二阶段使用自回归建模。

本文实际训练时用到了 VAE 的变体 $\beta$-VAE，给上述 ELBO 中的 KL 散度项乘了一个系数 $\beta$，当 $\beta > 1$ 时，正则化项的作用被加强。

### 3.2 训练技巧

与之前分析的论文不同的是，到这里 DALL-E 的所有理论背景就介绍完了，其余的都是工程技巧：

- 分布式训练：模型规模很大，需要在GPU集群上分布式训练
- 混合精度：使用半精度节约显存，如何避免梯度消失或爆炸
- 数据清洗：2.5 亿图文对训练集的搭建

### 3.3 样本生成与 CLIP 重排序

DALL-E 的推理过程结合了自回归生成和基于 CLIP 的后选择。

#### **3.3.1 自回归生成过程**

**第一步：文本编码**

```
text_tokens = BPE_encode(caption)  # 最多 256 tokens
```

**第二步：图像 token 生成**

直接使用 softmax 采样，推理不需要 Gumbel Softmax 松弛

```
generated_image_tokens = []
for position in range(1024):  # 32x32 positions
    logits = transformer(text_tokens + generated_image_tokens)
    next_token = sample(logits[position])
    generated_image_tokens.append(next_token)
```

**第三步：图像重建**

```
image = dVAE_decode(generated_image_tokens)
```

#### **3.3.2 CLIP 重排序机制**

DALL-E 只生成一张的效果并不一定好，为了保证效果，对每个文本 prompt 会**大批量生成** $N = 512$ 张候选图像，再使用预训练的 CLIP 模型计算文本-图像相似度：

$$
\mathrm{score}(text, image) = \cos(\mathrm{CLIP}_{\mathrm{text}}(text), \mathrm{CLIP}_{\mathrm{image}}(image))
$$

然后选择 CLIP 分数最高的 $k$ 张图像作为最终输出。根据论文中的分析，N=32 指标显著好于 N=1，但再提升到 N=512 收益有限。

某种程度上，这体现了大模型时代大资本的碾压性优势：他们不光有资源收集庞大的数据集、搭建昂贵的训练集群、训练小团队无法负担的大模型，还能在推理时采样抽奖，用钱换效果。

## 4. 总结

该总结主要由AI生成，只做了删减。

### 4.1 与动机的呼应

DALL-E 论文在其研究动机中提出了两个核心假设：**数据规模和模型规模是文本到图像生成的主要瓶颈**，以及**简单的自回归方法在充足资源下能够匹敌复杂的专用架构**。实验结果强有力地支持了这两个假设。

#### **4.1.1 规模效应的验证**

通过将数据规模从传统的数十万扩展到 2.5 亿图文对，模型参数从数百万增加到 120 亿，DALL-E 实现了质的飞跃：

- **零样本泛化**：在从未见过 MS-COCO 数据的情况下超越了所有在该数据集上专门训练的方法
- **涌现能力**：获得了训练时未明确优化的多种能力（概念组合、文字渲染、图像翻译等）
- **鲁棒性**：在多个评估维度上都表现出更高的稳定性

这验证了"规模是智能"这一现代 AI 的核心理念。

#### **4.1.2 简单方法的有效性**

相比于传统文本到图像方法的复杂设计（多尺度生成器、attention 机制、辅助损失、分割掩码等），DALL-E 采用了极其简洁的方案：

$$
\mathrm{离散化 Encoder (dVAE)} + \mathrm{自回归 Transformer} = \mathrm{强大的生成能力}
$$

这种简洁性的成功再次证明了"奥卡姆剃刀"原理在深度学习中的适用性：**在数据和计算充足的情况下，简单方法往往优于复杂方法**。

### 4.2 DALL-E 的局限性与后续工作方向

#### **4.2.1 当前局限性分析**

**计算效率问题**：

- 自回归生成的串行性导致推理速度慢（需要 1024 步顺序解码）
- 与扩散模型、GAN 等并行生成方法相比效率较低
- 大模型的存储和计算需求限制了部署场景

**图像质量天花板**：

- dVAE 的压缩损失导致细节缺失，特别是高频纹理
- 8×8 下采样比例的选择在压缩率和质量间做了妥协
- 在需要精细细节的应用场景中表现受限

**专业领域泛化问题**：

- 在 CUB 等专业数据集上表现不佳
- 对医学、科学等专业领域的概念理解不足
- 缺乏持续学习和领域适应机制

**可控性和可解释性**：

- 生成过程缺乏中间控制点
- 难以精确控制物体位置、大小、属性等
- 黑盒性质导致失败案例难以调试

#### **4.2.2 后续工作方向**

**DALL-E 2 的改进方向**：

- 采用 CLIP + Diffusion Model 架构，摆脱自回归的速度限制
- 使用更高分辨率的生成，提升细节质量
- 引入 inpainting、outpainting 等更多可控编辑功能

**更广泛的技术趋势**：

**1. 架构创新**：

- Diffusion Model 的兴起（Stable Diffusion, Imagen, Midjourney）
- 混合架构：结合 Transformer、CNN、扩散过程
- 更高效的注意力机制（Linear Attention, Flash Attention）

**2. 数据与训练**：

- 更大规模数据集（LAION-5B 等）
- 数据质量提升和 filtering 策略
- 指令跟随训练（instruction following）

**3. 可控性增强**：

- ControlNet、Adapter 等控制插件
- 多条件输入（文本+草图+pose 等）
- 更精细的语义控制

**4. 效率优化**：

- 模型压缩和量化
- 推理优化（并行解码、缓存机制）
- 边缘部署适配

#### **4.2.3 对 AI 发展的更广泛启示**

DALL-E 的成功体现了几个重要的 AI 发展趋势：

**大一统模型的崛起**：
从任务特定模型转向通用大模型，单一架构处理多种模态和任务成为主流。

**数据驱动的范式转换**：
从精心设计的算法转向数据驱动的端到端学习，数据质量和规模成为核心竞争力。

**涌现能力的重要性**：
大规模训练带来的涌现能力往往比专门优化的功能更有价值，预示着 AGI 发展方向。

**计算-数据-算法的协同**：
成功的 AI 系统需要在算力、数据、算法三个维度协同优化，单一维度的突破难以带来质变。

这些趋势在 DALL-E 之后的 GPT-4、Claude、Stable Diffusion 等模型中得到进一步验证，标志着 AI 进入了"大模型时代"的新阶段。

---

**总结**：DALL-E 不仅是一个成功的文本到图像生成模型，更是自回归建模思想演进的重要里程碑。它证明了简单方法在充足资源下的强大潜力，为后续的多模态大模型发展奠定了基础。通过与前期工作的对比分析，我们可以看到 AI 发展的清晰脉络：从专用到通用，从小规模到大规模，从单模态到多模态。
