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


This project is based on the Llama-2, released by Meta, and it is the second generation of the Chinese LLaMA & Alpaca LLM project. We open-source Chinese LLaMA-2 (foundation model) and Alpaca-2 (instruction-following model). These models have been expanded and optimized with Chinese vocabulary beyond the original Llama-2. We used large-scale Chinese data for incremental pre-training, which further improved the fundamental semantic understanding of the Chinese language, resulting in a significant performance improvement compared to the first-generation models. The relevant models support a 4K context and can be expanded up to 18K+ using the NTK method.

**The main contents of this project include:**

- 🚀 New extended Chinese vocabulary beyond Llama-2, open-sourcing the Chinese LLaMA-2 and Alpaca-2 LLMs.
- 🚀 Open-sourced the pre-training and instruction finetuning (SFT) scripts for further tuning on user's data
- 🚀 Quickly deploy and experience the quantized LLMs on CPU/GPU of personal PC
- 🚀 Support for LLaMA ecosystems like [🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LangChain](https://github.com/hwchase17/langchain), [vLLM](https://github.com/vllm-project/vllm) etc.
- The currently open-source models are Chinese-LLaMA-2-7B and Chinese-Alpaca-2-7B (check our [first-gen project](https://github.com/ymcui/Chinese-LLaMA-Alpaca) for more models).

![](./pics/screencast.gif)

----

 [Visual Chinese-LLaMA-Alpaca](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [Multi-modal VLE](https://github.com/iflytek/VLE) | [Chinese MiniRBT](https://github.com/iflytek/MiniRBT) | [Chinese LERT](https://github.com/ymcui/LERT) | [Chinese-English PERT](https://github.com/ymcui/PERT) | [Chinese MacBERT](https://github.com/ymcui/MacBERT) | [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [Chinese XLNet](https://github.com/ymcui/Chinese-XLNet) | [Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [Knowledge distillation tool TextBrewer](https://github.com/airaria/TextBrewer) | [Model pruning tool TextPruner](https://github.com/airaria/TextPruner)

## News

**[Aug 02, 2023] Add FlashAttention-2 training support, vLLM-based inference acceleration support, a new system prompt that generates longer response, etc. For details, see [📚 v1.1 release note](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.1)**

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

**🚄 Adaptive Context Extension based on NTK**

- In the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), we implemented the [context extension based on NTK](https://github.com/ymcui/Chinese-LLaMA-Alpaca/pull/743), which can support longer contexts without further training the model.
- Based on the above, we further designed a **convenient adaptive empirical formula** that does not require manually setting corresponding hyperparameters for different context lengths.
- The models in this project natively support a 4K context, which can be extended to 12K with the above technology and up to 18K+ at the expense of some accuracy loss.

**🤖 Simplified Bilingual System Prompt**

- In the [first generation of the project](https://github.com/ymcui/Chinese-LLaMA-Alpaca), we use [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) template for our Chinese Alpaca models
- Through preliminary experiments, we found that the lengthy system prompt by Llama-2-Chat is not as effective as a simple one
- We use a very simple system prompt while keeping the Llama-2-Chat template to better adapt to relevant ecosystems

## Download

### Model Selection Guide

Below is a basic comparison between the Chinese LLaMA-2 and Alpaca-2 models, as well as recommended use cases.

| Comparison                    | Chinese LLaMA-2                                              | Chinese Alpaca-2                                             |
| :---------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Training Method               | Traditional CLM                                              | Instruction fine-tuning                                      |
| Model Type                    | **Base Model**                                               | **Instruction/Chat Model (like ChatGPT)**                    |
| Training Corpus               | Unlabeled general corpus                                     | Labeled instruction data                                     |
| Vocabulary Size<sup>[1]</sup> | 55,296                                                       | 55,296                                                       |
| Input Template                | Not required                                                 | Requires specific templates<sup>[2]</sup>                    |
| Suitable Scenarios            | Text continuation: Given the context, the model generates the following text | Instruction understanding: Q&A, writing, chatting, interaction, etc. |
| Unsuitable Scenarios          | Instruction understanding, multi-turn chat, etc.             | Unrestricted text generation                                 |

[1] *The vocabulary of the first and second generation models in this project are different, do not mix them. The vocabularies of the second generation LLaMA and Alpaca are the same.*</br>
[2] *Alpaca-2 uses the Llama-2-chat series templates (different prompts), not the templates of the first-generation Alpaca, do not mix them.*</br>

### Full Model Download

Below are the full models, which can be used directly afterwards, without additional merging steps. Recommended for users with sufficient network bandwidth.

| Model Name          |       Type        |   Training Data   | Size |                        Download Link                         |
| :------------------ | :---------------: | :---------------: | :--: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-7B  |    Base Model     | 120G General Text | 13GB | [[Baidu Disk]](https://pan.baidu.com/s/1E5NI3nlQpx1j8z3eIzbIlg?pwd=n8k3)<br/>[[Google Drive]](https://drive.google.com/drive/folders/18pp4I-mvQxRA7b8vF9gP-2cH_ocnXVKh?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-llama-2-7b) |
| Chinese-Alpaca-2-7B | Instruction Model |  5M Instructions  | 13GB | [[Baidu Disk]](https://pan.baidu.com/s/1wxx-CdgbMupXVRBcaN4Slw?pwd=kpn9)<br/>[[Google Drive]](https://drive.google.com/drive/folders/1JsJDVs7tE2y31PBNleBlDPsB7S0ZrY8d?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b) |

### LoRA Model Download

Below are the LoRA models, **which cannot be used directly and must be merged with the refactored models according to the tutorial**. Recommended for users with insufficient network bandwidth, who already have the original Llama-2 and light-weight download.

| Model Name               |       Type        |   Training Data   |                       Refactored Model                       | Size  |                      LoRA Download Link                      |
| :----------------------- | :---------------: | :---------------: | :----------------------------------------------------------: | :---: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-7B-LoRA  |    Base Model     | 120G General Text | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1GB | [[Baidu Disk]](https://pan.baidu.com/s/1bmgqdyRh9E3a2uqOGyNqiQ?pwd=7kvq)<br/>[[Google Drive]](https://drive.google.com/file/d/1njJGSU_PRbzjYRNw5RSbC5-4fBOXTVY3/view?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-llama-2-lora-7b) |
| Chinese-Alpaca-2-7B-LoRA | Instruction Model |  5M Instructions  | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1GB | [[Baidu Disk]](https://pan.baidu.com/s/1g0olPxkB_rlZ9UUVfOnbcw?pwd=5e7w)<br/>[[Google Drive]](https://drive.google.com/file/d/1MzJL-ZIzdJW7MIcAiYIDIDJ5dlMi8Kkk/view?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-alpaca-2-lora-7b) |

As the LoRA models cannot be used separately, they must be merged with the original Llama-2 to form a complete model for model inference, quantization, or further training. Please choose one of the following methods to merge these models.

- [**Online Conversion**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/online_conversion_en): Colab users can use the notebook provided by this project for online conversion and model quantization
- [**Manual Conversion**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/manual_conversion_en): Offline method of conversion, generating different formats of models for quantization or further fine-tuning

## Inference and Deployment

The models in this project mainly support the following quantization, inference, and deployment methods.

| Tool                                                         | Features                                                | CPU  | GPU  | Quant | GUI  | API  | vLLM |                           Tutorial                           |
| :----------------------------------------------------------- | ------------------------------------------------------- | :--: | :--: | :---: | :--: | :--: | :--: | :----------------------------------------------------------: |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp)      | Rich quantization options and efficient local inference |  ✅   |  ✅   |   ✅   |  ❌   |  ✅   |  ❌   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_en) |
| [**🤗Transformers**](https://github.com/huggingface/transformers) | Native transformers inference interface                 |  ✅   |  ✅   |   ✅   |  ✅   |  ❌   |  ✅  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/inference_with_transformers_en) |
| [**Colab Demo**](https://colab.research.google.com/drive/1yu0eZ3a66by8Zqm883LLtRQrguBAb9MR?usp=sharing) | Running a Gradio web demo in Colab | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | [link](https://colab.research.google.com/drive/1yu0eZ3a66by8Zqm883LLtRQrguBAb9MR?usp=sharing) |
| [**OpenAI API Calls**](https://platform.openai.com/docs/api-reference) | A server that implements OpenAI API |  ✅   |  ✅   |  ✅   |  ❌   |  ✅   |  ✅  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/api_calls_en) |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) | A tool for deploying model as a web UI |  ✅   |  ✅   |  ✅   |  ✅   |  ❌   | ❌  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/text-generation-webui_en) |

⚠️ Inference and deployment support related to the first-generation model will be gradually migrated to this project, and relevant tutorials will be updated later.

## System Performance

### Generation Performance Evaluation

In order to intuitively understand the generation performance of the model, this project has launched an online model arena platform imitating [Fastchat Chatbot Arena](https://chat.lmsys.org/?arena), where you can browse and evaluate the quality of model responses. The arena platform provides evaluation indicators such as win rate and Elo score, and you can view the win rate of battles between two models. The question bank comes from [200 questions manually created in the first-generation project](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/f16-p7b-p13b-33b), and additional questions added on this basis. Generated replies are subject to randomness and are influenced by decoding hyperparameters, random seeds, etc., so the related evaluations are not absolutely rigorous. The results are only for reference, and you are welcome to experience it yourself. Please see the [examples directory](./examples) for some generated examples.

Tested models include:

- [**First-gen model**](https://github.com/ymcui/Chinese-LLaMA-Alpaca): Chinese-Alpaca-Pro series (7B/13B/33B), Chinese-Alpaca-Plus series (7B/13B/33B)
- [**Second-gen model (this project)**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2): Chinese-Alpaca-2 (7B)

**📊 Online ChatBot Arena**: [http://chinese-alpaca-arena.ymcui.com](http://chinese-alpaca-arena.ymcui.com/)

### NLU Performance Evaluation

This project also tested related models on the NLU datasets. The results of this type of evaluation are objective and only require the output of given labels, so they can provide insights into the capabilities of large models from another perspective. In the recently launched [C-Eval dataset](https://cevalbenchmark.com/), this project tested the performance of the relevant models. The test set contains 12.3K multiple-choice questions covering 52 subjects. The following are the evaluation results (average) of some models on the validation and test sets.

Comparisons between LLaMA models:

| Model                  | Valid (zero-shot) | Valid (5-shot) | Test (zero-shot) | Test (5-shot) |
| ---------------------- | :---------------: | :------------: | :--------------: | :-----------: |
| **Chinese-LLaMA-2-7B** |     **28.2**      |    **36.0**    |     **30.3**     |   **34.2**    |
| Chinese-LLaMA-Plus-13B |       27.3        |      34.0      |       27.8       |     33.3      |
| Chinese-LLaMA-Plus-7B  |       27.3        |      28.3      |       26.9       |     28.4      |

Comparisons between Alpaca models:

| Model                   | Valid (zero-shot) | Valid (5-shot) | Test (zero-shot) | Test (5-shot) |
| ----------------------- | :---------------: | :------------: | :--------------: | :-----------: |
| **Chinese-Alpaca-2-7B** |       41.3        |    **42.9**    |       40.3       |     39.5      |
| Chinese-Alpaca-Plus-13B |     **43.3**      |      42.4      |     **41.5**     |   **39.9**    |
| Chinese-Alpaca-Plus-7B  |       36.7        |      32.9      |       36.4       |     32.3      |

It is important to note that the comprehensive assessment of the capabilities of large models is still an urgent and significant topic to address. It is beneficial to approach the various evaluation results of large models in a rational and balanced manner to promote the healthy development of large-scale model technology. It is recommended for users to conduct tests on their own tasks and choose models that are suitable for the relevant tasks.

For C-Eval inference code, please refer to >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/ceval_en)

### Quantization Evaluation

To understand the quality loss brought by quantization, taking Chinese-LLaMA-2-7B as an example, we report the model size, PPL, C-eval results under different quantization levels. PPL is calculated under 4K context, and we report zero-shot and 5-shot results on C-Eval valid set.

| Precision | Model Size |  PPL   |   C-Eval    |
| :-------- | :--------: | :----: | :---------: |
| FP16      |  12.9 GB   | 8.1797 | 28.2 / 36.0 |
| 8-bit     |   6.8 GB   | 8.2884 | 26.8 / 35.4 |
| 4-bit     |   3.7 GB   | 8.8581 | 25.5 / 32.8 |

Specifically, the followings are the benchmark for different quantization methods in llama.cpp. The speed is presented with ms/tok. For details, see our [Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_en#quantization-method-and-inference-speed).

|           |    F16 |  Q4_0 |  Q4_1 |  Q4_K |  Q5_0 |  Q5_1 |  Q5_K |  Q6_K |  Q8_0 |
| --------- | -----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| PPL       |  8.640 | 8.987 | 9.175 | 8.836 | 8.730 | 8.776 | 8.707 | 8.671 | 8.640 |
| Size      | 12.91G | 3.69G | 4.08G | 3.92G | 4.47G | 4.86G | 4.59G | 5.30G | 6.81G |
| CPU Speed |    117 |    39 |    44 |    43 |    48 |    51 |    50 |    54 |    65 |
| GPU Speed |     53 |    17 |    18 |    20 |   n/a |   n/a |    25 |    26 |   n/a |

## Training and Fine-tuning

Please refer to the corresponding Wiki for information on pre-training (Chinese LLaMA-2 training) and instruction fine-tuning (Chinese Alpaca-2 training).

- **Pre-training**: The code is adapted from [run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) in 🤗transformers. For usage, see the [Pre-training Script Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_en).
- **Instruction Fine-tuning**: The code refers to the relevant parts of dataset handling in the [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) project. For usage, see the [Instruction Fine-tuning Script Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_en).

## FAQ

Please make sure to check if there is a solution in the FAQ before raising an Issue.

```
Question 1: What is the difference between this project and the first-gen project?
Question 2: Can the model be commercialized?
Question 3: Do you accept third-party Pull Requests?
Question 4: Why not perform full pre-training but use LoRA instead?
Question 5: Does Llama-2 series support tools that support the first-gen LLaMA?
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
