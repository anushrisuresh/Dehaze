# Rich Teacher Features for Efficient Single-Image Haze Removal

**PyTorch Implementation of**  
**_"Rich Teacher Features for Efficient Single-Image Haze Removal"_**

This repository provides an efficient and lightweight framework for single-image dehazing, leveraging **heterogeneous knowledge distillation** from a lightweight **super-resolution teacher model** to guide a compact **student dehazing network**.  
The method is simple yet effective, designed specifically for **on-the-edge deployment** where computational resources are limited.

<p align="center">
  <img src="figures/methodology.png" alt="Proposed Framework" width="600">
</p>

---

## 📄 Paper

> [Rich Teacher Features for Efficient Single-Image Haze Removal](https://arxiv.org/pdf/2207.11250)

**Abstract:**  
Single-image haze removal is a long-standing hurdle for computer vision applications. Previous methods relying on contrastive learning or traditional knowledge distillation often prove computationally expensive.  
This work introduces a simple, lightweight framework that exploits rich *dark knowledge* from a pre-trained super-resolution model through **heterogeneous knowledge distillation**. A novel **Feature Affinity Module** maximizes the transfer of rich feature semantics to the student dehazing network.  
Extensive experiments on the **RESIDE-Standard** dataset demonstrate that our framework achieves gains of up to **15% (PSNR)** while reducing the model size by approximately **20×**.
