# [Chinese-Mixtral](https://github.com/ymcui/Chinese-Mixtral) MoE LLM has been released!

[**🇨🇳中文**](./README.md) | [**🌐English**](./README_EN.md) | [**📖文档/Docs**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki) | [**❓提问/Issues**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/issues) | [**💬讨论/Discussions**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/discussions) | [**⚔️竞技场/Arena**](http://chinese-alpaca-arena.ymcui.com/)

<p align="center">
    <br>
    <img src="./pics/banner.png" width="800"/>
    <br>
</p>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ymcui/Chinese-LLaMA-Alpaca-2.svg?color=blue&style=flat-square">
    <img alt="GitHub release (latest by date)" src="https://img.shields.io/github/v/release/ymcui/Chinese-LLaMA-Alpaca-2">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ymcui/Chinese-LLaMA-Alpaca-2">
    <a href="https://app.codacy.com/gh/ymcui/Chinese-LLaMA-Alpaca-2/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/1710faac5e634acaabfc26b0a778cdde"/></a>
</p>


This project is based on the Llama-2, released by Meta, and it is the second generation of the Chinese LLaMA & Alpaca LLM project. We open-source Chinese LLaMA-2 (foundation model) and Alpaca-2 (instruction-following model). These models have been expanded and optimized with Chinese vocabulary beyond the original Llama-2. We used large-scale Chinese data for incremental pre-training, which further improved the fundamental semantic understanding of the Chinese language, resulting in a significant performance improvement compared to the first-generation models. Standard version supports 4K context, and long context version supports 16K and 64K context. The RLHF models are fine-tuned for human preference alignment and have gained significant performance improvements in the representation of correct values compared to the standard version of the model.

#### Main Contents

- 🚀 New extended Chinese vocabulary beyond Llama-2, open-sourcing the Chinese LLaMA-2 and Alpaca-2 LLMs.
- 🚀 Open-sourced the pre-training and instruction finetuning (SFT) scripts for further tuning on user's data
- 🚀 Quickly deploy and experience the quantized LLMs on CPU/GPU of personal PC
- 🚀 Support for LLaMA ecosystems like [🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LangChain](https://github.com/hwchase17/langchain), [privateGPT](https://github.com/imartinez/privateGPT), [vLLM](https://github.com/vllm-project/vllm) etc.

#### Open-sourced Models

- Base model: Chinese-LLaMA-2 (1.3B, 7B, 13B)
- Instruction/chat model: Chinese-Alpaca-2 (1.3B, 7B, 13B)
- Long context model (16K/64K): 
  - Chinese-LLaMA-2-16K (7B, 13B) 、Chinese-Alpaca-2-16K (7B, 13B) 
  - Chinese-LLaMA-2-64K (7B)、Chinese-Alpaca-2-64K (7B)
- RLHF model：Chinese-Alpaca-2-RLHF (1.3B, 7B)

![](./pics/screencast.gif)

----

[Chinese LLaMA&Alpaca LLMs](https://github.com/ymcui/Chinese-LLaMA-Alpaca)| [Visual Chinese-LLaMA-Alpaca](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [Multi-modal VLE](https://github.com/iflytek/VLE) | [Chinese MiniRBT](https://github.com/iflytek/MiniRBT) | [Chinese LERT](https://github.com/ymcui/LERT) | [Chinese-English PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [Knowledge distillation tool TextBrewer](https://github.com/airaria/TextBrewer) | [Model pruning tool TextPruner](https://github.com/airaria/TextPruner)

## News

**[Jan 23, 2024] Add new GGUF models (with imatrix), AWQ models, support YaRN under vLLM. For details, see[📚 v4.1 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v4.1)**

[Dec 29, 2023] Release long context models: Chiense-LLaMA-2-7B-64K and Chinese-Alpaca-2-7B-64K. We also release RLHF-tuned Chinese-Alpaca-2-RLHF (1.3B/7B). For details, see [📚 v4.0 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v4.0)

[Sep 01, 2023] Release long context models: Chinese-Alpaca-2-7B-16K and Chinese-Alpaca-2-13B-16K, which can be directly used in downstream tasks, such as privateGPT. For details, see [📚 v3.1 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v3.1)

[Aug 25, 2023] Release long context models: Chinese-LLaMA-2-7B-16K and Chinese-LLaMA-2-13B-16K, which support 16K context and can be further extended up to 24K+ using NTK. For details, see [📚 v3.0 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v3.0)

[Aug 14, 2023] Release Chinese-LLaMA-2-13B and Chinese-Alpaca-2-13B. Add text-generation-webui/LangChain/privateGPT support. Add CFG sampling, etc. For details, see [📚 v2.0 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v2.0)

[Aug 02, 2023] Add FlashAttention-2 training support, vLLM-based inference acceleration support, a new system prompt that generates longer response, etc. For details, see [📚 v1.1 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.1)

[July 31, 2023] Release Chinese-LLaMA-2-7B (base model), trained with 120GB Chinese data. It was further fine-tuned using 5M instruction data, resulting in the Chinese-Alpaca-2-7B (instruction/chat model). For details, see [📚 v1.0 release notes](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.0)

[July 19, 2023] 🚀Launched the [Chinese LLaMA-2 and Alpaca-2 open-source LLM project](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)

## Content Guide
| Section                                                | Description                                                  |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| [💁🏻‍♂️Introduction](#introduction)                       | Briefly introduces the technical features of the models in this project |
| [⏬Download](#download)                                 | Download links for Chinese LLaMA-2 and Alpaca-2              |
| [💻Inference and Deployment](#inference-and-deployment) | Introduces how to quantify models and deploy and experience large models using a personal computer |
| [💯System Performance](#system-performance)             | Experimental results on several tasks                        |
| [📝Training and Fine-tuning](#training-and-fine-tuning) | Introduces how to perform further training and fine-tuning on Chinese LLaMA-2 and Alpaca-2 |
| [❓Frequently Asked Questions](#FAQ)                    | Responses to some common questions                           |

## Introduction

This project launches the Chinese LLaMA-2 and Alpaca-2 models based on Llama-2. Compared to the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), the main features include:

**📖 Optimized Chinese Vocabulary**

- In the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), we expanded Chinese words and characters for the first-generation Chinese LLaMA model (LLaMA: 49953, Alpaca: 49954) to improve the model's encoding and decoding efficiency of Chinese texts.
- In this project, we **redesigned the new vocabulary** (size: 55296) to further improve the coverage of Chinese words and characters. We also unified the LLaMA/Alpaca vocabulary to avoid problems due to mixed use.

**⚡ Efficient FlashAttention-2**

- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention) is an implementation of efficient attention mechanisms, offering **faster speed and optimized memory usage** compared to its first-generation.
- When the context length is longer, using efficient attention technology is essential to prevent explosive growth in memory usage.

**🚄 Adaptive Context Extension based on PI and YaRN**

- In the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), we implemented the [context extension based on NTK](https://github.com/ymcui/Chinese-LLaMA-Alpaca/pull/743), which can support longer contexts without further training the model.
- We release long context models, using [PI](https://arxiv.org/abs/2306.15595) and NTK methods, supporting 16K context, and can be further extended up to 24K-32K
- We further release long context models, using [YaRN](https://arxiv.org/abs/2309.00071), supporting 64K context
- Based on the above, we further designed a **convenient adaptive empirical formula** that does not require manually setting corresponding hyperparameters for different context lengths.

**🤖 Simplified Bilingual System Prompt**

- In the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), we use [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) template for our Chinese Alpaca models
- Through preliminary experiments, we found that the lengthy system prompt by Llama-2-Chat is not as effective as a simple one
- We use a very simple system prompt while keeping the Llama-2-Chat template to better adapt to relevant ecosystems

#### 👮 Human Preference Alignment

- In the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), the Chinese Alpaca models completed pre-training and instruction fine-tuning, and gained basic conversational ability
- Through reinforcement learning from human feedback (RLHF) experiments, we find that the ability of the model to convey correct values can be significantly improved
- This project introduces the Alpaca-2-RLHF series of models, which are used in the same way as the SFT models

The following figure depicts all open-sourced models for our projects (including the [first-gen project](https://github.com/ymcui/Chinese-LLaMA-Alpaca)).

![](./pics/models.png)

## Download

### Model Selection Guide

Below is a basic comparison between the Chinese LLaMA-2 and Alpaca-2 models, as well as recommended use cases. **Use Alpaca for ChatGPT-like interaction.**

| Comparison                    |                       Chinese LLaMA-2                        |                       Chinese Alpaca-2                       |
| :---------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Model Type                    |                        **Base Model**                        |          **Instruction/Chat Model (like ChatGPT)**           |
| Released Sizes                |                        1.3B, 7B, 13B                         |                        1.3B, 7B, 13B                         |
| Training Method               |                       Causal-LM (CLM)                        |                   Instruction fine-tuning                    |
| Training Parts                |      7B, 13B: LoRA + emb/lm-head</br> 1.3B: full params      |      7B, 13B: LoRA + emb/lm-head</br> 1.3B: full params      |
| Trained on                    | [Original Llama-2](https://github.com/facebookresearch/llama) (non-chat) |                       Chinese LLaMA-2                        |
| Training Corpus               |           Unlabeled general corpus (120G raw text)           |            Labeled instruction data (5M samples)             |
| Vocabulary Size<sup>[1]</sup> |                            55,296                            |                            55,296                            |
| Context Size<sup>[2]</sup>    | Standard: 4K (12K-18K)<br/>Long ctx(PI): 16K (24K-32K) <br/>Long ctx(YaRN): 64K | Standard: 4K (12K-18K)<br/>Long ctx(PI): 16K (24K-32K) <br/>Long ctx(YaRN): 64K |
| Input Template                |                         Not required                         |          Requires specific templates<sup>[3]</sup>           |
| Suitable Scenarios            | Text continuation: Given the context, the model generates the following text | Instruction understanding: Q&A, writing, chatting, interaction, etc. |
| Unsuitable Scenarios          |       Instruction understanding, multi-turn chat, etc.       |                 Unrestricted text generation                 |
| Preference Alignment          |                              No                              |                   RLHF version (1.3B, 7B)                    |

> [!NOTE]
> [1] *The vocabulary of the first and second generation models in this project are different, do not mix them. The vocabularies of the second generation LLaMA and Alpaca are the same.*</br> 
> [2] *Extended context size with NTK method is depicted in brackets.*</br>
> [3] *Alpaca-2 uses the Llama-2-chat series templates (different prompts), not the templates of the first-generation Alpaca, do not mix them.*</br>
> [4] *1.3B models are not intended for standalone use; instead, use it together with larger models (7B, 13B) through speculative sampling.*</br>

### Full Model Download

Below are the full models, which can be used directly afterwards, without additional merging steps. Recommended for users with sufficient network bandwidth.

| Model Name            |       Type        |  Size   |                        Download Link                         |                        GGUF                        |
| :-------------------- | :---------------: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-13B | Base model | 24.7 GB | [[Baidu]](https://pan.baidu.com/s/1T3RqEUSmyg6ZuBwMhwSmoQ?pwd=e9qy) [[Google]](https://drive.google.com/drive/folders/1YNa5qJ0x59OEOI7tNODxea-1YvMPoH05?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-13b) | [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-13b-gguf) |
| Chinese-LLaMA-2-7B | Base model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/1E5NI3nlQpx1j8z3eIzbIlg?pwd=n8k3) [[Google]](https://drive.google.com/drive/folders/18pp4I-mvQxRA7b8vF9gP-2cH_ocnXVKh?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-7b) | [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-7b-gguf) |
| Chinese-LLaMA-2-1.3B | Base model | 2.4 GB | [[Baidu]](https://pan.baidu.com/s/1hEuOCllnJJ5NMEZJf8OkRw?pwd=nwjg) [[Google]](https://drive.google.com/drive/folders/1Sd3PA_gs6JctXtBg5HwmHXh9GX93riMP?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-1.3b) | [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-1.3b-gguf) |
| Chinese-Alpaca-2-13B | Chat Model | 24.7 GB | [[Baidu]](https://pan.baidu.com/s/1MT_Zlap1OtdYMgoBNTS3dg?pwd=9xja) [[Google]](https://drive.google.com/drive/folders/1MTsKlzR61xmbTR4hBWzQas_MOpUZsogN?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-13b) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-13b-gguf) |
| Chinese-Alpaca-2-7B | Chat Model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/1wxx-CdgbMupXVRBcaN4Slw?pwd=kpn9) [[Google]](https://drive.google.com/drive/folders/1JsJDVs7tE2y31PBNleBlDPsB7S0ZrY8d?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-gguf) |
| Chinese-Alpaca-2-1.3B | Chat model | 2.4 GB | [[Baidu]](https://pan.baidu.com/s/1PD7Ng-ltOIdUGHNorveptA?pwd=ar1p) [[Google]](https://drive.google.com/drive/folders/1h6qOy-Unvqs1_CJ8uPp0eKC61Gbbn8n7?usp=share_link)[[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-1.3b) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-1.3b-gguf) |

#### Long Context Models

The followings are long context models, which are recommended for long context tasks. 

| Model Name                |    Type    |  Size   |                        Download Link                         |                        GGUF                        |
| :------------------------ | :--------: | :-----: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-7B-64K 🆕  | Base model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/1ShDQ2FG2QUJrvfnxCn4hwQ?pwd=xe5k) [[Google]](https://drive.google.com/drive/folders/17l9xJx55L2YNpqt7NiLVQzOZ6fV4rzJ-?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-7b-64k) | [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-7b-64k-gguf) |
| Chinese-Alpaca-2-7B-64K 🆕 | Chat model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/1KBAr9PCGvX2oQkYfCuLEjw?pwd=sgp6) [[Google]](https://drive.google.com/drive/folders/13G_d5xcDnhtaMOaulj1BFiZbVoVwJ-Cu?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-64k) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-64k-gguf) |
| Chinese-LLaMA-2-13B-16K   | Base model | 24.7 GB | [[Baidu]](https://pan.baidu.com/s/1XWrh3Ru9x4UI4-XmocVT2w?pwd=f7ik) [[Google]](https://drive.google.com/drive/folders/1nii6lF0DgB1u81CnsE4cCK2jD5oq_OW-?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-13b-16k) | [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-13b-16k-gguf) |
| Chinese-LLaMA-2-7B-16K    | Base model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/1ZH7T7KU_up61ugarSIXw2g?pwd=pquq) [[Google]](https://drive.google.com/drive/folders/1Zc6jI5bl3myQbQsY79dWJJ8mP_fyf3iF?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-7b-16k) | [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-7b-16k-gguf) |
| Chinese-Alpaca-2-13B-16K  | Chat model | 24.7 GB | [[Baidu]](https://pan.baidu.com/s/1gIzRM1eg-Xx1xV-3nXW27A?pwd=qi7c) [[Google]](https://drive.google.com/drive/folders/1mOkYQCvEqtGoZ9DaIpYFweSkSia2Q0vl?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-13b-16k) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-13b-16k-gguf) |
| Chinese-Alpaca-2-7B-16K   | Chat model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/1Qk3U1LyvMb1RSr5AbiatPw?pwd=bfis) [[Google]](https://drive.google.com/drive/folders/1KBRSd2xAhiVQmamfA5wpm5ovYFRKuMdr?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-16k) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-16k-gguf) |

#### RLHF Models

The following lists the RLHF models which exhibit a better value orientation than the standard version for issues involving law, ethics, etc.

| Model Name                |    Type    |  Size   |                        Download Link                         |                        GGUF                        |
| :------------------------ | :------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Chinese-Alpaca-2-7B-RLHF 🆕 | Chat Model | 12.9 GB | [[Baidu]](https://pan.baidu.com/s/17GJ1y4rpPDuvWlvPaWgnqw?pwd=4feb) [[Google]](https://drive.google.com/drive/folders/1OHZVVtwM5McVEIZzyOYgGYLAxcZNVK4D?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-rlhf) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-7b-rlhf-gguf) |
| Chinese-Alpaca-2-1.3B-RLHF 🆕 | Chat Model | 2.4 GB | [[Baidu]](https://pan.baidu.com/s/1cLKJKieNitWbOggUXXaamw?pwd=cprp) [[Google]](https://drive.google.com/drive/folders/1zcvPUPPkq69SgqRu6YBurAZ9ptcPSZNx?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-1.3b-rlhf) | [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-1.3b-rlhf-gguf) |

#### AWQ Models

AWQ (Activation-aware Weight Quantization) is an efficient quantization method, which can be used with 🤗transformers, llama.cpp, etc.

The pre-computed search results of our models are available: https://huggingface.co/hfl/chinese-llama-alpaca-2-awq

- Generate AWQ-quantized models (AWQ official repo): https://github.com/mit-han-lab/llm-awq#usage
- Using AWQ in llama.cpp: https://github.com/ggerganov/llama.cpp/tree/master/awq-py

### LoRA Model Download

Below are the LoRA models, **which cannot be used directly and must be merged with the refactored models according to the tutorial**. Recommended for users with insufficient network bandwidth, who already have the original Llama-2 and light-weight download.

| Model Name               |       Type        |                       Required Model for merging                       | Size  |                      LoRA Download Link                      |
| :----------------------- | :---------------: | :----------------------------------------------------------: | :---: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-LoRA-13B | Base model | [Llama-2-13B-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 1.5 GB | [[Baidu]](https://pan.baidu.com/s/1PFKTBn54GjAjzWeQISKruw?pwd=we6s) [[Google]](https://drive.google.com/file/d/10Z_k9A9N9D_6RHrMTmbHQRCuI6s1iMb1/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-lora-13b) |
| Chinese-LLaMA-2-LoRA-7B | Base model |        [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)        | 1.1 GB | [[Baidu]](https://pan.baidu.com/s/1bmgqdyRh9E3a2uqOGyNqiQ?pwd=7kvq) [[Google]](https://drive.google.com/file/d/1njJGSU_PRbzjYRNw5RSbC5-4fBOXTVY3/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-lora-7b) |
| Chinese-Alpaca-2-LoRA-13B | Chat Model | [Llama-2-13B-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 1.5 GB | [[Baidu]](https://pan.baidu.com/s/1Y5giIXOUUzI4Na6JOcviVA?pwd=tc2j) [[Google]](https://drive.google.com/file/d/1z2FIInsYJBTXipgztc-Mv7kkeqscx442/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-lora-13b) |
| Chinese-Alpaca-2-LoRA-7B | Chat Model | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1 GB | [[Baidu]](https://pan.baidu.com/s/1g0olPxkB_rlZ9UUVfOnbcw?pwd=5e7w) [[Google]](https://drive.google.com/file/d/1MzJL-ZIzdJW7MIcAiYIDIDJ5dlMi8Kkk/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-lora-7b) |

The followings are long context models, which are recommended for long context tasks. 

| Model Name                      |    Type    |                  Required Model for merging                  |  Size  |                        Download Link                         |
| :------------------------------ | :--------: | :----------------------------------------------------------: | :----: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-LoRA-7B-64K 🆕 | Base model | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1 GB | [[Baidu]](https://pan.baidu.com/s/1QjqKNM9Xez5g6koUrbII_w?pwd=94pk) [[Google]](https://drive.google.com/file/d/1-NuGqfduUZARRquFjGLpTmI5J-HlXYSR/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-lora-7b-64k) |
| Chinese-Alpaca-2-LoRA-7B-64K 🆕 | Chat model | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1 GB | [[Baidu]](https://pan.baidu.com/s/1t6bPpMlJCrs9Ce7LXs09-w?pwd=37it) [[Google]](https://drive.google.com/file/d/1qESorx2PHtIsnj53JJ7XBsdOGHuLNjoI/view?usp=sharing) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-lora-7b-64k) |
| Chinese-LLaMA-2-LoRA-13B-16K    | Base model | [Llama-2-13B-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 1.5 GB | [[Baidu]](https://pan.baidu.com/s/1VrfOJmhDnXxrXcdnfX00fA?pwd=4t2j) [[Google]](https://drive.google.com/file/d/1mSpigmHcN9YX1spa4QN3IPtx43Vfs55H/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-lora-13b-16k) |
| Chinese-LLaMA-2-LoRA-7B-16K     | Base model | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1 GB | [[Baidu]](https://pan.baidu.com/s/14Jnm7QmcDx3XsK_NHZz6Uw?pwd=5b7i) [[Google]](https://drive.google.com/file/d/1yUdyQuBMAmxmUEAvGiKbjKuxTYPPI-or/view?usp=sharing) [[🤗HF]](https://huggingface.co/hfl/chinese-llama-2-lora-7b-16k) |
| Chinese-Alpaca-2-LoRA-13B-16K | Chat Model | [Llama-2-13B-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf) | 1.5 GB | [[Baidu]](https://pan.baidu.com/s/1g42_X7Z0QWDyrrDqv2jifQ?pwd=bq7n) [[Google]](https://drive.google.com/file/d/1ppGNyMWnuLDcClXN7DBTbKxVehsn3Gd2/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-lora-13b-16k) |
| Chinese-Alpaca-2-LoRA-7B-16K  | Chat Model | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1 GB | [[Baidu]](https://pan.baidu.com/s/1E7GEZ6stp8EavhkhR06FwA?pwd=ewwy) [[Google]](https://drive.google.com/file/d/1GTgDNfMdcQhHEAfMPaP-EOEk_fwDvNEK/view?usp=share_link) [[🤗HF]](https://huggingface.co/hfl/chinese-alpaca-2-lora-7b-16k) |

> [!IMPORTANT] 
> As the LoRA models cannot be used separately, they must be merged with the original Llama-2 to form a complete model for model inference, quantization, or further training. Please choose one of the following methods to merge these models.
>
> - [**Online Conversion**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/online_conversion_en): Colab users can use the notebook provided by this project for online conversion and model quantization
> - [**Manual Conversion**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/manual_conversion_en): Offline method of conversion, generating different formats of models for quantization or further fine-tuning

## Inference and Deployment

The models in this project mainly support the following quantization, inference, and deployment methods.

| Tool                                                         | Features                                                | CPU  | GPU  | Quant | GUI  | API  | vLLM<sup>§</sup> | 16K<sup>‡</sup> | 64K<sup>‡</sup> |Speculative  Sampling |                           Tutorial                           |
| :----------------------------------------------------------- | ------------------------------------------------------- | :--: | :--: | :---: | :--: | :--:| :--: | :--: | :----------------------------------------------------------: | :----------------------------------------------------------: | ------------------------------------------------------------ |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp)      | Rich quantization options and efficient local inference |  ✅   |  ✅   |   ✅   |  ❌   |  ✅   |  ❌   |  ✅  |✅  | ✅ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_en) |
| [**🤗Transformers**](https://github.com/huggingface/transformers) | Native transformers inference interface                 |  ✅   |  ✅   |   ✅   |  ✅   |  ❌   |  ✅  |  ✅  |✅  | ✅ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/inference_with_transformers_en) |
| [**Colab Demo**](https://colab.research.google.com/drive/1yu0eZ3a66by8Zqm883LLtRQrguBAb9MR?usp=sharing) | Running a Gradio web demo in Colab | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅  |✅ | [link](https://colab.research.google.com/drive/1yu0eZ3a66by8Zqm883LLtRQrguBAb9MR?usp=sharing) |
| [**OpenAI API Calls**](https://platform.openai.com/docs/api-reference) | A server that implements OpenAI API |  ✅   |  ✅   |  ✅   |  ❌   |  ✅   |  ✅  |  ✅  |✅  | ❌ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/api_calls_en) |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) | A tool for deploying model as a web UI |  ✅   |  ✅   |  ✅   |  ✅   | ✅<sup>†</sup> | ❌  | ✅ | ❌ |❌ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/text-generation-webui_en) |
| [**LangChain**](https://github.com/hwchase17/langchain) | LLM application development framework, suitable for secondary development |  ✅<sup>†</sup>  |  ✅   |  ✅<sup>†</sup>   |  ❌   |  ❌   | ❌  | ✅ | ✅ |❌ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/langchain_en) |
| [**privateGPT**](https://github.com/imartinez/privateGPT) | LangChain-based multi-document QA framework | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/privategpt_en) |

> [!NOTE]
> <sup>†</sup>: Supported by this tool, but not implemented in the tutorial. Please refer to the official documentation for details. <br/>
> <sup>‡</sup>: Support long context or not (requires customized RoPE support)</br>
> <sup>§</sup>: vLLM backend does not support our long context models. </br>

## System Performance

### Generation Performance Evaluation

In order to intuitively understand the generation performance of the model, this project has launched an online model arena platform imitating [Fastchat Chatbot Arena](https://chat.lmsys.org/?arena), where you can browse and evaluate the quality of model responses. The arena platform provides evaluation indicators such as win rate and Elo score, and you can view the win rate of battles between two models. The question bank comes from [200 questions manually created in the first-generation project](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/f16-p7b-p13b-33b), and additional questions added on this basis. Generated replies are subject to randomness and are influenced by decoding hyperparameters, random seeds, etc., so the related evaluations are not absolutely rigorous. The results are only for reference, and you are welcome to experience it yourself. Please see the [examples directory](./examples) for some generated examples.

**⚔️ Online Chatbot Arena: [http://llm-arena.ymcui.com](http://llm-arena.ymcui.com/)**

| System                                                       | Win Rate (no tie)↓ | Elo Rating |
| ------------------------------------------------------------ | :----------------: | :--------: |
| **Chinese-Alpaca-2-13B-16K**                                 |        86.84%        |  1580   |
| **Chinese-Alpaca-2-13B**                                     |        72.01%        |  1579   |
| [Chinese-Alpaca-Pro-33B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |        64.87%        |  1548   |
| **Chinese-Alpaca-2-7B**                                      |        64.11%        |  1572   |
| [Chinese-Alpaca-Pro-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |        62.05%        |  1500   |
| **Chinese-Alpaca-2-7B-16K**                                  |        61.67%        |  1540   |
| [Chinese-Alpaca-Pro-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |        61.26%        |  1567   |
| [Chinese-Alpaca-Plus-33B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |        31.29%        |  1401   |
| [Chinese-Alpaca-Plus-13B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |        23.43%        |  1329   |
| [Chinese-Alpaca-Plus-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca) |        20.92%        |  1379   |

> [!NOTE]
> Results timestamp: Sep 1. 2023 . For the latest results, see [**⚔️Arena**](http://llm-arena.ymcui.com/).

### NLU Performance Evaluation: C-Eval

[C-Eval](https://cevalbenchmark.com/) is a comprehensive Chinese basic model evaluation suite. The validation set contains 1.3K multiple-choice questions, and the test set contains 12.3K multiple-choice questions, covering 52 subjects. The type of questions is multiple-choice. The experimental results are presented in the format of "zero-shot / 5-shot". For C-Eval inference code, please refer to this project's [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/ceval_en).

| LLaMA Models            |    Valid    |    Test     | Alpaca Models            |    Valid    |    Test     |
| ----------------------- | :---------: | :---------: | ------------------------ | :---------: | :---------: |
| **Chinese-LLaMA-2-13B** | 40.6 / 42.7 | 38.0 / 41.6 | **Chinese-Alpaca-2-13B** | 44.3 / 45.9 | 42.6 / 44.0 |
| **Chinese-LLaMA-2-7B**  | 28.2 / 36.0 | 30.3 / 34.2 | **Chinese-Alpaca-2-7B**  | 41.3 / 42.9 | 40.3 / 39.5 |
| Chinese-LLaMA-Plus-33B  | 37.4 / 40.0 | 35.7 / 38.3 | Chinese-Alpaca-Plus-33B  | 46.5 / 46.3 | 44.9 / 43.5 |
| Chinese-LLaMA-Plus-13B  | 27.3 / 34.0 | 27.8 / 33.3 | Chinese-Alpaca-Plus-13B  | 43.3 / 42.4 | 41.5 / 39.9 |
| Chinese-LLaMA-Plus-7B   | 27.3 / 28.3 | 26.9 / 28.4 | Chinese-Alpaca-Plus-7B   | 36.7 / 32.9 | 36.4 / 32.3 |

### NLU Performance Evaluation: CMMLU

[CMMLU](https://github.com/haonan-li/CMMLU) is another comprehensive Chinese evaluation dataset, specifically designed to evaluate the knowledge and reasoning abilities of language models in a Chinese context. It covers 67 topics ranging from basic subjects to advanced professional levels, with a total of 11.5K test cases. The type of questions is multiple-choice. For CMMLU inference code, please refer to this project's [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/cmmlu_en).

| LLaMA Models            | Test (0/few-shot) | Alpaca Models            | Test (0/few-shot) |
| ----------------------- | :---------------: | ------------------------ | :---------------: |
| **Chinese-LLaMA-2-13B** |    38.9 / 42.5    | **Chinese-Alpaca-2-13B** |    43.2 / 45.5    |
| **Chinese-LLaMA-2-7B**  |    27.9 / 34.1    | **Chinese-Alpaca-2-7B**  |    40.0 / 41.8    |
| Chinese-LLaMA-Plus-33B  |    35.2 / 38.8    | Chinese-Alpaca-Plus-33B  |    46.6 / 45.3    |
| Chinese-LLaMA-Plus-13B  |    29.6 / 34.0    | Chinese-Alpaca-Plus-13B  |    40.6 / 39.9    |
| Chinese-LLaMA-Plus-7B   |    25.4 / 26.3    | Chinese-Alpaca-Plus-7B   |    36.8 / 32.6    |

### Long Context Model Evaluation

[LongBench](https://github.com/THUDM/LongBench) is a benchmark for testing LLM's long context ability, consisting of 6 categories and 20 tasks. The average length of most of the task ranges from 5K to 15K. LongBench has 4.5K test samples in total. The followings are the results on Chinese subtasks. For LongBench inference code, please refer to this project's [📖GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/longbench_en)

| Models                      | Single-doc QA | Multi-doc QA | Summarization | Few-shot Learning | Code Completion | Synthetic Task | Avg  |
| --------------------------- | :-----------: | :----------: | :-----------: | :---------------: | :-------------: | :------------: | :--: |
| **Chinese-Alpaca-2-7B-64K** | 44.7  |  28.1 | 14.4 |  39.0   |  44.6  |   5.0  | 29.3|
| **Chinese-LLaMA-2-7B-64K** | 27.2  |  16.4 | 6.5 |  33.0   |  7.8  |   5.0  | 16.0|
| **Chinese-Alpaca-2-13B-16K** |   47.9  |   26.7 | 13.0 |     22.3    |   46.6   |   21.5   | 29.7 |
| Chinese-Alpaca-2-13B         |   38.4   |   20.0   | 11.9 |     17.3    |   46.5   |   8.0    | 23.7 |
| **Chinese-Alpaca-2-7B-16K**  |   46.4  |   23.3  | 14.3 |     29.0     |   49.6   |   9.0    | 28.6 |
| Chinese-Alpaca-2-7B          |   34.0   |   17.4   | 11.8 |     21.3    |   50.3  |   4.5    | 23.2 |
| **Chinese-LLaMA-2-13B-16K**  |   36.7   |   17.7  | 3.1 |     29.8     |   13.8   |   3.0    | 17.3 |
| Chinese-LLaMA-2-13B          |   28.3   |   14.4   | 4.6 |     16.3     |   10.4   |   5.4    | 13.2 |
| **Chinese-LLaMA-2-7B-16K**   |   33.2   |   15.9   | 6.5 |     23.5     |   10.3    |   5.3    | 15.8|
| Chinese-LLaMA-2-7B           |   19.0   |   13.9   | 6.4  |     11.0    |   11.0   |   4.7    | 11.0 |

### Quantization Evaluation

To understand the quality loss brought by quantization, taking Chinese-LLaMA-2-7B as an example, we report the model size, PPL, C-eval results under different quantization levels. PPL is calculated under 4K context, and we report zero-shot and 5-shot results on C-Eval valid set.

| Precision       | Model Size |  PPL   |   C-Eval    |
| :-------------- | :--------: | :----: | :---------: |
| FP16            |  12.9 GB   | 9.373  | 28.2 / 36.0 |
| 8-bit quantized |   6.8 GB   | 9.476  | 26.8 / 35.4 |
| 4-bit quantized |   3.7 GB   | 10.132 | 25.5 / 32.8 |

Specifically, the followings are the benchmark for different quantization methods in llama.cpp. The speed is presented with ms/tok. For details, see our [Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_en#quantization-method-and-inference-speed).

| llama.cpp |    F16 |   Q2_K |  Q3_K |  Q4_0 |  Q4_1 |  Q4_K |  Q5_0 |  Q5_1 |  Q5_K |  Q6_K |  Q8_0 |
| --------- | -----: | -----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| PPL       |  9.128 | 11.107 | 9.576 | 9.476 | 9.576 | 9.240 | 9.156 | 9.213 | 9.168 | 9.133 | 9.129 |
| Size      | 12.91G |  2.41G | 3.18G | 3.69G | 4.08G | 3.92G | 4.47G | 4.86G | 4.59G | 5.30G | 6.81G |
| CPU Speed |    117 |     42 |    51 |    39 |    44 |    43 |    48 |    51 |    50 |    54 |    65 |
| GPU Speed |     53 |     19 |    21 |    17 |    18 |    20 |     x |     x |    25 |    26 |     x |

### Speculative Sampling Evaluation

Using speculative sampling and leveraging Chinese-LLaMA-2-1.3B and Chinese-Alpaca-2-1.3B can accelerate the inference speed of 7B and 13B LLaMA and Alpaca models. The followings are the inference speeds (ms/token) evaluated on the questions in [Generation Performance Evaluation](#Generation-Performance-Evaluation) on 1*A40-48G. All the models are in fp16 format. For details, see our [Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/inference_with_transformers_en#Speculative-Sampling).

| Draft Model           | Draft Model Speed | Target Model         | Target Model Speed | Speculative Sampling Speed |
| :-------------------- | :---------------: | :------------------- | :----------------: | :------------------------: |
| Chinese-LLaMA-2-1.3B  |        7.6        | Chinese-LLaMA-2-7B   |        49.3        |       36.0（1.37x）        |
| Chinese-LLaMA-2-1.3B  |        7.6        | Chinese-LLaMA-2-13B  |        66.0        |       47.1（1.40x）        |
| Chinese-Alpaca-2-1.3B |        8.1        | Chinese-Alpaca-2-7B  |        50.2        |       34.9（1.44x）        |
| Chinese-Alpaca-2-1.3B |        8.2        | Chinese-Alpaca-2-13B |        67.0        |       41.6（1.61x）        |


### RLHF Models Evaluation

#### Alignment

To assess the degree of alignment of the RLHF models with human preferences, we constructed our own evaluation dataset, which covers a number of aspects that are the focus of human value preferences, such as morality, pornography, drugs, violence, and so on. The experimental results are presented in terms of the percentage of correct value embodiment (number of systematically correct value questions / total number of questions).

| Alpaca Models            | Accuracy |  Alpaca Models            | Accuracy |
| ------------------------ | :---------------: |------------------------ | :---------------: |
| Chinese-Alpaca-2-1.3B |   79.3%    | Chinese-Alpaca-2-7B  |    88.3%    |
| **Chinese-Alpaca-2-1.3B-RLHF** |    95.8%    | **Chinese-Alpaca-2-7B-RLHF** |    97.5%    |


#### NLU Performance Evaluation： C-Eval & CMMLU
| Alpaca Models            | C-Eval (0/few-shot) | CMMLU (0/few-shot) |
| ------------------------ | :---------------: | :---------------: |
| Chinese-Alpaca-2-1.3B |    23.8 / 26.8    |    24.8 / 25.1    |
| Chinese-Alpaca-2-7B  |    42.1 / 41.0    |    40.0 / 41.8    |
| **Chinese-Alpaca-2-1.3B-RLHF** |    23.6 / 27.1    |    24.9 / 25.0    |
| **Chinese-Alpaca-2-7B-RLHF** |    40.6 / 41.2    |    39.5 / 41.0    |


## Training and Fine-tuning

Please refer to the corresponding Wiki for information on pre-training (Chinese LLaMA-2 training) and instruction fine-tuning (Chinese Alpaca-2 training).

- **Pre-training**: The code is adapted from [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) in 🤗transformers. For usage, see the [Pre-training Script Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_en).
- **Instruction Fine-tuning**: The code refers to the relevant parts of dataset handling in the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) project. For usage, see the [Instruction Fine-tuning Script Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_en).
- **RLHF Fine-tuning**: Reinforcement learning from human feedback fine-tuning using preference data and PPO algorithm based on Chinese-Alpaca-2. For details, see the [📖Reward Modeling Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/rm_en)和[📖Reinforcement Learning Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/rl_en).


## FAQ

Please make sure to check if there is a solution in the FAQ before raising an Issue.

```
Question 1: What is the difference between this project and the first-gen project?
Question 2: Can the model be commercialized?
Question 3: Do you accept third-party Pull Requests?
Question 4: Why not perform full pre-training but use LoRA instead?
Question 5: Does Llama-2 series support tools that support the first-gen LLaMA?
Question 6: Is Chinese-Alpaca-2 trained from Llama-2-Chat?
Question 7: Why does training with 24GB VRAM lead to an OOM error when fine-tuning chinese-alpaca-2-7b?
Question 8: Can the 16K long-context version model replace the standard version model?
Question 9: How to interprete the results of third-party benchmarks?
Question 10: Will you release 34B or 70B models?
Question 11: Why the long-context model is 16K context, not 32K or 100K?
Question 12: Why does the Alpaca model reply that it is ChatGPT?
Question 13: Why is the adapter_model.bin in the pt_lora_model or sft_lora_model folder only a few hundred kb?
```

For specific questions and answers, please refer to the project >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/faq_en)

## Citation

If you use the resources related to this project, please refer to and cite this project's technical report: https://arxiv.org/abs/2304.08177
```
@article{Chinese-LLaMA-Alpaca,
    title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca},
    author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
    journal={arXiv preprint arXiv:2304.08177},
    url={https://arxiv.org/abs/2304.08177},
    year={2023}
}
```

## Acknowledgments

This project is mainly based on the following open-source projects, and we would like to express our gratitude to the related projects and research developers.

- [Llama-2 *by Meta*](https://github.com/facebookresearch/llama)
- [llama.cpp *by @ggerganov*](https://github.com/ggerganov/llama.cpp)
- [FlashAttention-2 by *Dao-AILab*](https://github.com/Dao-AILab/flash-attention)

We also appreciate the contributors of Chinese-LLaMA-Alpaca (the first-gen project) and [the associated projects and personnel](https://github.com/ymcui/Chinese-LLaMA-Alpaca#致谢).

## Disclaimer

This project is predicated on the utilization of the Llama-2 model, as released by Meta. As such, we respectfully request all users to adhere diligently to the provisions of the open-source license agreement pertinent to the Llama-2 model. In instances where third-party code is integrated, strict adherence to the appropriate open-source license agreement is also essential. Please be advised that the precision of the content generated by the model is subject to variability due to computational methodologies, random elements, and potential degradation of quantization accuracy. Consequently, this project makes no warranties, express or implied, regarding the accuracy of the model output. Furthermore, this project cannot be held accountable for any losses, whether direct or consequential, that may arise from the use of associated resources and the results derived therefrom. In cases where the models associated with this project are employed for commercial purposes, it is incumbent upon developers to act in accordance with local laws and regulations, thereby ensuring the legality of the content generated by the model. Finally, please note that this project does not accept any liability for products or services that may be developed based on its models.

<details>
<summary><b>Limitation Statement</b></summary>

Although the models in this project have significantly improved Chinese understanding and generation capabilities compared to the original LLaMA and Alpaca, there are also the following limitations:

- It may produce unpredictable harmful content and content that does not conform to human preferences and values.
- Due to computing power and data issues, the training of the related models is not sufficient, and the Chinese understanding ability needs to be further improved.
- There is no online interactive demo available for now (Note: users can still deploy it locally themselves).

</details>


## Feedback

If you have any questions, please submit them in GitHub Issues.

- Before submitting a question, please check if the FAQ can solve the problem and consult past issues to see if they can help.
- Please use our dedicated issue template for submitting.
- Duplicate and unrelated issues will be handled by [stable-bot](https://github.com/marketplace/stale); please understand.
- Raise questions politely and help build a harmonious discussion community.
