# 🖼️ Text-to-Image Generator using Stable Diffusion

This repository showcases my work at Muonium AI, where I built a **Text-to-Image Generator** using the Stable Diffusion model. The application converts natural language prompts into high-quality, photorealistic images using deep learning.


## 🚀 Overview

Stable Diffusion is a powerful generative model capable of creating detailed images from textual descriptions. This project leverages pre-trained diffusion models to allow users to generate images simply by providing descriptive prompts.


## ✨ Features
### 🎨 Text-to-Image Generation
Generate high-quality images from textual input.
### 🧠 Customizable Prompts
Users can input unique prompts to create diverse outputs.
### ⚡ Pre-trained Model
Uses a fine-tuned Stable Diffusion model for optimal results.
### 🖥️ Easy CLI Interface
Simple and user-friendly command-line execution.

---

## ⚠️ Disclaimer

This project requires a **CUDA-compatible GPU** for efficient performance.
Without GPU acceleration:
- Image generation will be significantly slower
- Processing time may become impractical


## 🛠️ Installation
Install all required dependencies using pip:


```bash
pip install tqdm accelerate torch diffusers numpy pandas transformers matplotlib opencv-python
```
## 📦 Dependencies Explained

- **tqdm** → Progress bars for tracking execution  
- **accelerate** → Efficient multi-GPU / distributed execution  
- **torch (PyTorch)** → Core deep learning framework  
- **diffusers** → Stable Diffusion implementation  
- **numpy** → Numerical computations  
- **pandas** → Data manipulation  
- **transformers** → Pre-trained NLP models  
- **matplotlib** → Visualization tools  
- **opencv-python** → Image processing  

---
