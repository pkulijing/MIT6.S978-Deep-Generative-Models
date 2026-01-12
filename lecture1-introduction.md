# [MIT 6.S978 Deep Generative Models](https://mit-6s978.github.io/schedule.html)

## 课程笔记与详细解析 (Fall 2024)

## [第一讲：课程简介 Introduction](https://mit-6s978.github.io/assets/pdfs/lec1_intro.pdf)

本章主要介绍了生成式人工智能（GenAI）时代的背景，对比了判别式模型与生成式模型，并确立了概率建模作为本课程的核心框架。

### 📖 Slide 1-2: 课程标题与 GenAI 时代

本课程由**Kaiming He 教授**在 MIT EECS 系讲授，聚焦于**深度生成模型**（Deep Generative Models）。课程开篇就点出我们正处于"**GenAI 时代**"，这标志着生成式 AI 已经从学术研究走向大规模实际应用。

### 📖 Slide 2-7: GenAI 时代的典型应用

课程展示了生成式 AI 在多个领域的突破性应用：

- **聊天机器人与自然语言对话**：如 ChatGPT 等大语言模型，能够进行流畅的多轮对话
- **文本到图像生成**：使用 Stable Diffusion 3 Medium 生成的示例，提示词"teddy bear teaching a course, with 'generative models' written on blackboard"展示了模型理解复杂语义并生成高质量图像的能力
- **文本到视频生成**：Sora 等模型能够根据文本描述生成连贯的视频内容
- **代码生成 AI 助手**：帮助程序员编写和调试代码
- **蛋白质设计与生成**：Watson 等人 2023 年在 Nature 发表的 RFdiffusion 工作，展示了生成模型在科学研究中的应用
- **天气预报**：2021 年 Nature 论文展示使用深度生成模型进行降水预报（"Skilful precipitation nowcasting using deep generative models of radar"）

**分析**：这些应用横跨文本、图像、视频、代码、生物学、气象学等多个领域，充分说明生成模型的通用性和强大能力。

### 📖 Slide 8-9: GenAI 时代之前的生成模型

课程回顾了早期的生成模型技术：

- **2009 年 PatchMatch 算法**：用于 Photoshop 的 Content-aware Fill（内容感知填充）功能，发表于 SIGGRAPH 2009
- **1999 年 Efros-Leung 算法**：用于**纹理合成**（Texture Synthesis），发表于 ICCV 1999。有趣的是，课程指出这实际上是一个**Autoregressive 模型**（自回归模型），只是当时没有这样命名

**深刻见解**：许多现代生成模型的核心思想在几十年前就已经出现，只是受限于计算能力和数据规模。深度学习的发展使这些思想能够在更大规模上实现。

### 📖 Slide 10-11: 什么是生成模型？

课程提出一个核心问题：上述应用场景有什么共同点？

- 对于一个输入，存在**多个或无限个可能的预测结果**
- 某些预测比其他预测更"**合理**"（plausible）
- 训练数据可能**不包精确的解**
- 预测结果可能比输入**更复杂、更有信息量、维度更高**

课程列举了聊天机器人、视频生成、图像生成、蛋白质生成等多个例子来说明这些特点。

### 📖 Slide 12-15: 判别式模型 vs. 生成式模型

**判别式模型**（Discriminative Models）：

- 从样本 x 映射到标签 y：x → y
- 对于每个输入，有**一个期望的输出**
- 例如：图像分类，输入狗的图片，输出"dog"标签

**生成式模型**（Generative Models）：

- 从标签 y 映射到样本 x：y → x
- 对于每个输入，有**多个可能的输出**
- 例如：给定"dog"标签，生成各种狗的图片

**重要关系**：

- **生成模型可以做判别**：通过**贝叶斯规则**（Bayes' rule）：

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$

其中 $p(x)$ 对于给定 $x$ 是常数，$p(y)$ 是已知的先验

- **判别模型能做生成吗？** 理论上可以通过：

$$
p(x|y) = \frac{p(y|x)p(x)}{p(y)}
$$

但问题是需要建模 $x$ 的**先验分布** $p(x)$，其中 $p(y)$ 对于给定 $y$ 是常数

**核心挑战**：表示和预测分布（representing and predicting distributions）。这就是生成模型的本质挑战。

### 📖 Slide 16-21: 概率建模（Probabilistic Modeling）

**概率从何而来？** 课程提出两种视角：

**视角 1：假设底层的数据生成过程**

- 假设存在**隐变量** $z$（如姿态、光照、尺度、品种等）
- $z$ 服从简单分布
- 观测值 $x$ 通过"世界模型"（world model）或物理模型、渲染器从 $z$ 生成
- 因此 $x$ 具有复杂分布
- 课程引用了 Freeman 和 Tenenbaum 1996 年的工作作为例子

**视角 2：概率是建模的一部分**

- 可能并不存在真实的"底层"分布
- 即使存在，我们只能观察到**有限的数据点**
- 模型对观测进行**外推**以建模分布
- 面临**过拟合 vs.欠拟合**的权衡（类似判别模型）

**过拟合与欠拟合的可视化**：

- **欠拟合**：模型过于简单，分布过于平滑，无法捕捉数据的真实复杂性
- **合适拟合**：模型复杂度适中，能够很好地泛化到新数据
- **过拟合**：模型过于复杂，分布由许多尖峰组成。极端情况下，使用**delta 函数**相当于直接从训练数据采样（记忆训练数据）

### 📖 Slide 22-27: 生成模型的概率建模流程

课程通过可视化展示了生成模型的完整流程：

1. **数据**（data）：收集训练数据点
2. **数据分布**（distribution of data）：认识到这些数据点来自某个分布（这本身就是建模的一部分）
3. **估计分布**（estimated distribution of data）：通过优化损失函数，学习一个估计的数据分布
4. **生成新数据**（sample new "data"）：从学习到的分布中采样，生成新的"数据"
5. **估计概率密度**（estimate prob density）：可以查询特定点 $x$ 的概率密度 $p(x)$

**重要注意事项**（Notes）：

- 生成模型涉及**统计模型**，这些模型通常由人类设计和推导
- **概率建模不仅仅是神经网络的工作**
- 概率建模是一种流行的方法，但**不是唯一的方法**
- 引用 George Box 的经典名言："_所有模型都是错的，但有些是有用的_"（"All models are wrong, but some are useful"）

### 📖 Slide 28-30: 什么是深度生成模型？

**深度学习是表示学习**（Deep learning is representation learning）

**传统深度学习**：学习表示数据实例

- 将数据映射到特征：$f(x; \theta)$
- 用目标最小化损失：$\mathcal{L}(f(x), y)$

**深度生成模型**：学习表示概率分布

- 将简单分布（如**高斯分布/均匀分布**）映射到复杂分布
- 映射函数：$G(z; \theta)$，其中 $z$ 来自简单分布
- 用数据分布最小化损失：$\mathcal{L}(G(z), p_{\text{data}})$
- **通常同时进行**实例表示学习和分布表示学习

**关键区别**：从简单分布到复杂分布的映射，是深度生成模型的核心。

### 📖 Slide 31-35: 分布建模的设计与学习

**核心理念**：不是所有的分布建模都通过学习完成。深度生成模型结合了人类先验知识（设计）和数据驱动学习。

**案例研究 1：Autoregressive 模型**

- **依赖图是设计的**（不是学习的）：

$$
p(x) = p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)\cdots
$$

这种分解方式由人类指定

- **映射函数是学习的**：神经网络（如 Transformer）学习如何基于前面的 token 预测下一个 token

**案例研究 2：Diffusion 模型**

- **依赖图是设计的**：
  - 前向过程（noising）：逐步加噪的时间步关系
  - 反向过程（denoising）：逐步去噪的时间步关系
  - 这些依赖关系由人类设计
- **映射函数是学习的**：神经网络（如 U-Net）学习如何在每个时间步去噪

**深刻理解**：深度生成模型是**归纳偏置**（inductive bias，人类设计的结构）和**数据驱动学习**（神经网络参数）的优雅结合。

### 📖 Slide 36: 深度生成模型的完整框架

课程总结了深度生成模型涉及的所有关键组成部分：

- **1. 公式化（Formulation）**：

  - 将问题表述为概率建模
  - 将复杂分布分解为简单且可处理的分布

- **2. 表示（Representation）**：

  - 使用深度神经网络表示数据
  - 使用深度神经网络表示数据的分布

- **3. 目标函数（Objective Function）**：

  - 衡量预测分布的质量

- **4. 优化（Optimization）**：

  - 优化网络参数
  - 优化分解方式（在某些情况下）

- **5. 推理（Inference）**：

  - **采样器**（sampler）：产生新样本
  - **概率密度估计器**（可选）：估计 $p(x)$ 的值

这个框架将贯穿整个课程，不同的生成模型（VAE、GAN、Autoregressive、Diffusion 等）都可以从这个框架来理解。

### 📖 Slide 37-50: 将真实世界问题表述为生成模型

**核心框架**：生成模型本质上是建模条件概率 $p(x|y)$

- **y**：条件/约束/标签/属性（更抽象，信息量更少）
- **x**：数据/样本/观测/测量（更具体，信息量更多）

课程通过 11 个具体案例展示如何表述不同问题：

**案例 1：自然语言对话**

- y: 用户的 prompt（提示词）
- x: 聊天机器人的 response（响应）

**案例 2：文本到图像/视频生成**

- y: 文本 prompt（如"teddy bear teaching a course, with 'generative models' written on blackboard"）
- x: 生成的视觉内容（图像或视频）
- 实例：由 Stable Diffusion 3 Medium 生成的图像，由 Sora 生成的视频

**案例 3：文本到 3D 结构生成**

- y: 文本 prompt
- x: 生成的 3D 结构
- 引用：Tang 等人的 LGM（Large Multi-View Gaussian Model）工作，ECCV 2024

**案例 4：蛋白质结构生成**

- y: 条件/约束（如对称性 symmetry 等结构约束）
- x: 生成的蛋白质结构
- 引用：Watson 等人的 RFdiffusion（"De novo design of protein structure and function"），Nature 2023

**案例 5：类条件图像生成**

- y: 类别标签（如"red fox"）
- x: 生成的该类别图像
- 引用：Li 等人的"Autoregressive Image Generation without Vector Quantization"，2024

**案例 6："无条件"图像生成**

- y: 隐式条件"遵循 CIFAR10 分布的图像"
- x: 生成的 CIFAR10 风格图像
- **重要区分**：$p(x|y)$ 是给定某类别的图像分布，$p(x)$ 是所有图像的分布
- 引用：Karras 等人"Elucidating the Design Space of Diffusion-Based Generative Models"，NeurIPS 2022

**案例 7：分类（生成视角）**

- y: 图像（作为"条件"）
- x: 在该图像条件下，各类别的概率分布（cat, bird, horse, dog 等）
- **洞察**：分类也可以看作生成模型，生成的是类别的概率分布

**案例 8：开放词汇识别（Open-vocabulary Recognition）**

- y: 图像（作为"条件"）
- x: 多种可能的描述（bird, flamingo, red color, orange color, ...）
- 与传统分类不同，输出空间是开放的

**案例 9：图像标注（Image Captioning）**

- y: 图像（作为"条件"）
- x: 合理的描述性句子
- 与开放词汇识别类似，但通常输出完整的自然语言描述

**案例 10：多模态聊天机器人**

- y: 图像和文本 prompt 的组合
- x: 聊天机器人的响应
- 引用：GPT-4 Technical Report，2023

**案例 11：机器人策略学习（Policy Learning in Robotics）**

- y: 视觉和其他传感器观测（visual and other sensory observations）
- x: 策略（policies），即动作的概率分布
- 引用：Chi 等人的"Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"，RSS 2023

**关键洞察**：许多看似不同的问题都可以统一地表述为生成模型 $p(x|y)$。**关键在于识别**：什么是 $x$ ? 什么是 $y$?如何表示 $x$, $y$ 及其依赖关系？

### 💡 第一讲总结

Lecture 1 为整个课程奠定了坚实的基础。Kaiming He 教授清晰地定义了：

1. **生成模型的本质**：处理一对多映射，建模概率分布
2. **与判别模型的区别与联系**：通过贝叶斯规则相互转化，但生成模型面临更大的挑战
3. **概率建模的核心思想**：将复杂分布分解为简单分布，结合人类设计和数据学习
4. **统一的问题表述框架**：通过 $p(x|y)$ 统一理解各种应用
5. **深度生成模型的完整流程**：从公式化、表示、目标函数、优化到推理

这一讲展示了生成模型的强大通用性——从自然语言到图像、视频、3D、蛋白质、机器人控制，几乎所有 AI 问题都可以从生成模型的角度来理解和解决。这为后续深入学习 VAE、GAN、Autoregressive、Diffusion 等具体模型做好了铺垫。
