# Autoregressive vs. Masked Diffusion Transformers for NMT

<img width="2879" height="1766" alt="image" src="https://github.com/user-attachments/assets/290bedff-0610-48c0-946c-3baafaaf041b" />
[Check out the ARM vs Diffusion NMT Space](https://huggingface.co/spaces/sammyboi1801/arm_vs_diffusion_nmt_streamlit)

This repository contains a PyTorch implementation and comparative analysis of **Autoregressive Models (ARM)** versus **Masked Diffusion Models (MDM)** for English-to-French machine translation.

Using identical 44M-parameter architectures trained on the OPUS Books dataset, this project investigates whether diffusion models can compete with the standard autoregressive paradigm at smaller scales.

## 📄 Project Overview

Autoregressive transformers (like GPT) generate text sequentially, which creates an inference bottleneck. Diffusion models offer a promising alternative by generating tokens in parallel through iterative refinement.

This project implements both paradigms from scratch to compare:
1.  **Translation Quality:** BLEU, ROUGE, and BERTScore.
2.  **Inference Strategy:** Left-to-right (ARM) vs. Iterative Denoising (Diffusion).
3.  **Training Efficiency:** Convergence rates and loss landscapes.

## 🏗️ Architecture & Method

To ensure a fair comparison, both models utilize nearly identical backbones:

* **Parameters:** ~44 Million
* **Layers:** 8 Decoder Layers
* **Hidden Dimension:** 512 (`d_model`)
* **Attention Heads:** 8
* **Tokenizer:** Custom Byte-Pair Encoding (BPE) trained jointly on En-Fr (Vocab size: 20,000)
* **Dataset:** OPUS Books (127k sentence pairs)

### The Models
| Component | Autoregressive (ARM) | Masked Diffusion (MDM) |
| :--- | :--- | :--- |
| **Training Objective** | Next-token prediction (Causal Masking) | Masked Token Reconstruction (BERT-style) |
| **Inference** | Sequential Greedy Decoding (O(N)) | [Iterative Denoising / LLaDA-style sampling |
| **Context** | Unidirectional (Left-to-Right) | Bidirectional (Full Context) |

## 🚀 Getting Started

### Prerequisites
The project is implemented in a single self-contained Jupyter Notebook. You will need Python 3.8+ and a GPU (recommended).

### Installation
Clone the repository and run the ipynb file.

### Running the Project
The entire pipeline (Data processing -> Training -> Inference -> Evaluation) is contained within `Diffusion vs Transformers for NMT.ipynb`.

1.  **Launch Jupyter:**
    ```bash
    jupyter notebook "Diffusion vs Transformers for NMT.ipynb"
    ```
2.  **Execute Cells:** Run the cells sequentially. The notebook handles:
    * Downloading and tokenizing the OPUS Books dataset.
    * Initializing the ARM and Diffusion architectures.
    * Training loops (default: 50 epochs).
    * Generation and metric calculation.

*Note: If you wish to skip training and jump straight to inference, ensure you download the checkpoints below and place them in a `checkpoints/` directory.*

## 💾 Model Checkpoints

Pre-trained weights for both models (trained for 50 epochs) are available for download. 

| Model | Description | Link |
| :--- | :--- | :--- |
| **ARM_epoch_40.pt** | Autoregressive Model Weights | [Download Here](https://huggingface.co/sammyboi1801/Diffusion-vs-Transformers/tree/main) |
| **Diffusion_epoch_40.pt** | Masked Diffusion Model Weights | [Download Here](https://huggingface.co/sammyboi1801/Diffusion-vs-Transformers/tree/main) |

*Please create a folder named `checkpoints` in the root directory and place these `.pt` files inside to use the inference cells immediately.*

## 📊 Results

We evaluated both models on a held-out test set using standard NMT metrics.

| Metric | Autoregressive (ARM) | Masked Diffusion |
| :--- | :--- | :--- |
| **BLEU** | **30.42** | 13.77 |
| **ROUGE-L** | **0.6549** | 0.6068 |
| **BERTScore** | **0.8905** | 0.8013 |

### Key Findings
* **Performance:** At the 44M parameter scale, the Autoregressive model significantly outperforms the Diffusion model.
* **Convergence:** ARM converges much faster (perplexity ~50) compared to Diffusion, which requires significantly more training steps to resolve semantic inconsistencies.
* **Scale:** Our analysis suggests that diffusion models suffer from a lack of scale in this experiment. Literature (e.g., LLaDA) suggests emergent capabilities for diffusion primarily appear at 7B+ parameters.

## 👥 Authors

* **Sam Selvaraj** - Northeastern University | selvaraj.sam@northeastern.edu
* **Dhruv Puri** - Northeastern University | puri.dh@northeastern.edu
* **Jatan Patel** - Northeastern University | patel.jatan@northeastern.edu

## 📚 References

1.  *Opus Books Dataset* - [HuggingFace](https://huggingface.co/datasets/Helsinki-NLP/opus_books)
2.  *Nie et al. (2025). Large Language Diffusion Models (LLaDA)*
3.  *Ho et al. (2020). Denoising Diffusion Probabilistic Models*
