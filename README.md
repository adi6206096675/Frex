# Frex AI Framework


![Typing SVG](https://readme-typing-svg.demolab.com?lines=Frex:+High-Performance+UVM+AI+Engine;AROM+Labs+Deep+Learning;Built+for+Efficiency+at+Scale)


[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


Frex is a high-performance, Unified Virtual Memory (UVM) Deep Learning framework designed for constrained hardware and efficient large-scale training. Developed under **AROM Labs**, Frex abstracts the complexity of memory management and allows models to scale beyond the physical limitations of GPU VRAM by utilizing intelligent page-spilling into system RAM.

## Detailed Description
Frex reimagines the deep learning stack by implementing a page-based memory virtualization system. Unlike traditional frameworks that rely on contiguous memory allocation, Frex manages memory in 1MB physical pages, allowing the engine to treat RAM and VRAM as a single, seamless computational pool. 

**Key Features:**
* **Unified Virtual Memory (UVM):** Automatically spills tensors to system RAM when VRAM is exhausted.
* **Operator Fusion:** Compiles fused math kernels (e.g., Linear + Bias + ReLU) to minimize memory bandwidth bottlenecks.
* **Dynamic Autograd:** A lightweight DAG-based automatic differentiation engine.
* **Native Extensibility:** Easily bind custom C++ kernels using PyBind11.

## Comparison Table

| Feature | PyTorch/TensorFlow | Frex (AROM Labs) |
| :--- | :--- | :--- |
| **Memory Strategy** | Contiguous / OOM-prone | Page-based / UVM |
| **Compute Overhead** | High (Read/Write Wall) | Low (Operator Fusion) |
| **Hardware Boundary** | Strict VRAM limits | Spillover to System RAM |
| **Engine Weight** | Heavyweight | Lightweight Core |

## Getting Started

### Prerequisites
* Python 3.12+
* C++17 Compiler (gcc/g++)
* Pybind11

### Installation
Clone the repository and install Frex in editable mode:
```bash
git clone [https://github.com/adi6206096675/Frex.git]
cd frex
pip install -e .
```




###LICENSE
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

###Credits
Built with ❤️ by Aditya under AROM Labs.


---

