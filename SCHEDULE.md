# MIT 6.S978 Deep Generative Models — Course Schedule

课程主页：https://mit-6s978.github.io/schedule.html

---

## Lectures

| #   | 日期       | 标题                                                                      | 课件                                                                                   |
| --- | ---------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1   | 2024-09-05 | Introduction                                                              | [slides](https://mit-6s978.github.io/assets/pdfs/lec1_intro.pdf)                       |
| 2   | 2024-09-12 | Variational Autoencoder (VAE)                                             | [slides](https://mit-6s978.github.io/assets/pdfs/lec2_vae.pdf)                         |
| 3   | 2024-09-19 | Autoregressive (AR) Models                                                | [slides](https://mit-6s978.github.io/assets/pdfs/lec3_ar.pdf)                          |
| 4   | 2024-10-03 | Generative Adversarial Network (GAN)                                      | [slides](https://mit-6s978.github.io/assets/pdfs/lec4_gan.pdf)                         |
| 5   | 2024-10-17 | Energy-based Models, Score Matching, Diffusion Models                     | [slides](https://mit-6s978.github.io/assets/pdfs/lec5_diffusion.pdf)                   |
| G1  | 2024-11-07 | Guest Lecture: Jun-Yan Zhu — Ensuring Data Ownership in Generative Models | [slides](https://mit-6s978.github.io/assets/pdfs/data_ownership_MIT_guest_lecture.pdf) |
| G2  | 2024-11-21 | Guest Lecture: Yang Song — Consistency Models                             | [slides](https://mit-6s978.github.io/assets/pdfs/CM_lecture.pdf)                       |

---

## Reading Sessions

### Reading 1 — Modeling Image Prior（2024-09-10）

**必读：**

- Zoran & Weiss. "From Learning Models of Natural Image Patches to Whole Image Restoration." ICCV 2011.
- Zoran & Weiss. "Natural Images, Gaussian Mixtures and Dead Leaves." NeurIPS 2012.
- Ulyanov et al. "Deep Image Prior." CVPR 2018.

**可选：**

- Efros & Leung. "Texture Synthesis by Non-parametric Sampling." 1999.
- Barnes et al. "PatchMatch: A Randomized Correspondence Algorithm for Structural Image Editing." 2009.
- Freeman et al. "Example-Based Super-Resolution." 2002.

---

### Reading 2 — Normalizing Flows（2024-09-17）

**必读：**

- Rezende & Mohamed. "Variational Inference with Normalizing Flows." ICML 2015.
- Kingma & Dhariwal. "Glow: Generative Flow with Invertible 1×1 Convolutions." NeurIPS 2018.
- Behrmann et al. "Invertible Residual Networks." ICML 2019.

**可选：**

- Lilian Weng. "Flow-based Deep Generative Models." (blog post)

---

### Reading 3 — Autoregressive (AR) Models（2024-09-24）

**必读：**

- Bengio & Bengio. "Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks." NIPS 1999.
- Van Den Oord et al. "Pixel Recurrent Neural Networks." ICML 2016.
- Kingma et al. "Improved Variational Inference with Inverse Autoregressive Flow." NIPS 2016.

---

### Reading 4 — AR and Tokenizers（2024-09-26）

**必读：**

- Yu et al. "Language Model Beats Diffusion — Tokenizer is Key to Visual Generation." ICLR 2024.
- Tian et al. "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction." arXiv.
- Yu et al. "An Image is Worth 32 Tokens for Reconstruction and Generation." arXiv.

**可选：**

- Chen et al. "Generative Pretraining from Pixels." ICML 2020.
- Ramesh et al. "Zero-Shot Text-to-Image Generation (DALL-E)." ICML 2021.
- Mentzer et al. "Finite Scalar Quantization: VQ-VAE Made Simple." arXiv.

---

### Reading 5 — AR and Diffusion（2024-10-01）

**必读：**

- Li et al. "Autoregressive Image Generation without Vector Quantization." arXiv.
- Zhou et al. "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model." arXiv.
- Xie et al. "Show-o: One Single Transformer to Unify Multimodal Understanding and Generation." arXiv.

**可选：**

- Hoogeboom et al. "Autoregressive Diffusion Models." ICLR 2022.
- Chen et al. "Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion." arXiv.

---

### Reading 6 — GAN 1（2024-10-08）

**必读：**

- Sauer et al. "StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis." ICML 2023.
- Kang et al. "Scaling up GANs for Text-to-Image Synthesis." CVPR 2023.
- Kang et al. "Distilling Diffusion Models into Conditional GANs." ECCV 2024.

**可选：**

- Tu. "Learning Generative Models via Discriminative Approaches." CVPR 2007.

---

### Reading 7 — GAN 2（2024-10-10）

**必读：**

- Huang et al. "The GAN is Dead; Long Live the GAN! A Modern GAN Baseline." ICML 2024.
- Wang et al. "Diffusion-GAN: Training GANs with Diffusion." ICLR 2023.
- Asokan et al. "GANs Settle Scores!" arXiv.

---

### Reading 8 — Diffusion Models（2024-10-22）

**必读：**

- Ho & Salimans. "Classifier-Free Diffusion Guidance." NeurIPS 2021.
- Salimans & Ho. "Progressive Distillation for Fast Sampling of Diffusion Models." ICLR 2022.
- Hoogeboom et al. "Simple Diffusion: End-to-end Diffusion for High-Resolution Images." ICML 2023.

**可选：**

- Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
- Karras et al. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
- Ramesh et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents." arXiv.
- Chen, T. "On the Importance of Noise Scheduling for Diffusion Models." arXiv.
- Sander Dieleman. "Diffusion is Spectral Autoregression." (blog post)

---

### Reading 9 — Diffusion Beyond Denoising（2024-10-24）

**必读：**

- Bansal et al. "Cold Diffusion: Inverting Arbitrary Image Transforms without Noise." NeurIPS 2023.
- Rissanen et al. "Generative Modelling with Inverse Heat Dissipation." ICLR 2023.
- Delbracio et al. "Inversion by Direct Iteration: An Alternative to Denoising Diffusion for Image Restoration." TMLR 2023.

**可选：**

- Daras et al. "Soft Diffusion: Score Matching for General Corruptions." TMLR 2023.

---

### Reading 10 — Discrete Diffusion（2024-10-29）

**必读：**

- Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces." NeurIPS 2021.
- Gong et al. "DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models." ICLR 2023.
- Lou et al. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution." ICML 2024.

---

### Reading 11 — Flow Matching 1（2024-10-31）

**必读：**

- Lipman et al. "Flow Matching for Generative Modeling." ICLR 2023.
- Albergo et al. "Building Normalizing Flows with Stochastic Interpolants." ICLR 2023.
- Liu et al. "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow." ICLR 2023.

**可选：**

- Fjelde et al. "An Introduction to Flow Matching." (blog post)

---

### Reading 12 — Flow Matching 2（2024-11-05）

**必读：**

- Esser et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." ICML 2024.
- Ma et al. "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers." ECCV 2024.
- Gat et al. "Discrete Flow Matching." arXiv.

---

### Reading 13 — Application: Videos（2024-11-12）

**必读：**

- Bar-Tal et al. "Lumiere: A Space-Time Diffusion Model for Video Generation." arXiv.
- Bruce et al. "Genie: Generative Interactive Environments." arXiv.
- Meta Movie Gen team. "Movie Gen: A Cast of Media Foundation Models." arXiv.

---

### Reading 14 — Application: 3D and Geometry（2024-11-14）

**必读：**

- Poole et al. "DreamFusion: Text-to-3D using 2D Diffusion." ICLR 2023.
- Hong et al. "LRM: Large Reconstruction Model for Single Image to 3D." ICLR 2024.
- Siddiqui et al. "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers." CVPR 2024.

**可选：**

- Zhang et al. "CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets." SIGGRAPH 2024.
- Wei et al. "MeshLRM: Large Reconstruction Model for High-Quality Meshes." arXiv.
- Shen et al. "Flexible Isosurface Extraction for Gradient-Based Mesh Optimization." SIGGRAPH 2023.

---

### Reading 15 — Application: Robotics（2024-11-19）

**必读：**

- Janner et al. "Planning with Diffusion for Flexible Behavior Synthesis." ICML 2022.
- Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." RSS 2023.
- Yang et al. "UniSim: Learning Interactive Real-World Simulators." ICLR 2024.

**可选：**

- Driess et al. "PaLM-E: An Embodied Multimodal Language Model." ICML 2023.

---

### Reading 16 — Application: Material Science（2024-11-26）

**必读：**

- Jin et al. "Junction Tree Variational Autoencoder for Molecular Graph Generation." ICML 2018.
- Hoogeboom et al. "Equivariant Diffusion for Molecule Generation in 3D." ICML 2022.
- Zhou et al. "Uni-Mol: A Universal 3D Molecular Representation Learning Framework." ICLR 2023.

**可选：**

- Xu et al. "Geometric Latent Diffusion Models for 3D Molecule Generation." ICML 2023.
- Arts et al. "Two for One: Diffusion Models and Force Fields for Coarse-Grained Molecular Dynamics." arXiv.

---

### Reading 17 — Application: Protein and Biology（2024-12-03）

**必读：**

- Watson et al. "De novo design of protein structure and function with RFdiffusion." Nature.
- Abramson et al. "Accurate structure prediction of biomolecular interactions with AlphaFold 3." Nature.
- Ingraham et al. "Illuminating protein space with a programmable generative model." Nature.
