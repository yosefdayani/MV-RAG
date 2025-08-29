# MV-RAG: Retrieval Augmented Multiview Diffusion
Yosef Dayani, Omer Benishu, Sagie Benaim

| [Project Page](https://yosefdayani.github.io/MV-RAG/) | [Paper](https://arxiv.org/pdf/2508.16577v1) | [HuggingFace](https://huggingface.co/papers/2508.16577) | [Weights](https://huggingface.co/yosepyossi/mvrag) | [Benchmark (OOD-Eval)](https://huggingface.co/datasets/yosepyossi/OOD-Eval) |

![teaser](https://yosefdayani.github.io/MV-RAG/static/images/teaser.jpg)

## 📌 Overview
MV-RAG is a text-to-3D generation method that retrieves 2D reference images to guide a multiview diffusion model. By conditioning on both text and multiple real-world 2D images, MV-RAG improves realism and consistency for rare/out-of-distribution or newly emerging objects.

---

## ⚙️ Installation

We recommend creating a fresh conda environment to run MV-RAG:

```bash
# Clone the repository
git clone https://github.com/yosefdayani/MV-RAG.git
cd MV-RAG

# Create new environment
conda create -n mvrag python=3.9 -y
conda activate mvrag

# Install PyTorch (adjust CUDA version as needed)
# Example: CUDA 12.4, PyTorch 2.5.1
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```
___
## 🏋️ Weights

MV-RAG weights are available on [Hugging Face](https://huggingface.co/yosepyossi/mvrag).
```bash
# Make sure git-lfs is installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/yosepyossi/mvrag
```

Then the model weights should appear as MV-RAG/mvrag/...

---
## 🚀 Usage Example
You could prompt the model on your retrieved local images by:
```bash
python main.py \
--prompt "Cadillac 341 automobile car" \
--retriever simple \
--folder_path "assets/Cadillac 341 automobile car" \
--seed 0 \
--k 4 \
--azimuth_start 45  # or 0 for front view
```

To see all command options run
```bash
python main.py --help
```

## 🙌 Acknowledgement
This repository is based on [MVDream](https://github.com/bytedance/MVDream) and adapted from [MVDream Diffusers](https://github.com/ashawkey/mvdream_diffusers). We would like to thank the authors of these works for publicly releasing their code.

## 📖 Citation
``` bibtex
@misc{dayani2025mvragretrievalaugmentedmultiview,
      title={MV-RAG: Retrieval Augmented Multiview Diffusion}, 
      author={Yosef Dayani and Omer Benishu and Sagie Benaim},
      year={2025},
      eprint={2508.16577},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.16577}, 
}
```