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


本项目基于Meta发布的可商用大模型[Llama-2](https://github.com/facebookresearch/llama)开发，是[中文LLaMA&Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的第二期项目，开源了**中文LLaMA-2基座模型和Alpaca-2指令精调大模型**。这些模型**在原版Llama-2的基础上扩充并优化了中文词表**，使用了大规模中文数据进行增量预训练，进一步提升了中文基础语义和指令理解能力，相比一代相关模型获得了显著性能提升。相关模型**支持4K上下文并可通过NTK方法最高扩展至18K+。**

**本项目主要内容：**

- 🚀 针对Llama-2模型扩充了**新版中文词表**，开源了中文LLaMA-2和Alpaca-2大模型
- 🚀 开源了预训练脚本、指令精调脚本，用户可根据需要进一步训练模型
- 🚀 使用个人电脑的CPU/GPU快速在本地进行大模型量化和部署体验
- 🚀 支持[🤗transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp), [text-generation-webui](https://github.com/oobabooga/text-generation-webui), [LangChain](https://github.com/hwchase17/langchain), [privateGPT](https://github.com/imartinez/privateGPT), [vLLM](https://github.com/vllm-project/vllm)等LLaMA生态
- 目前已开源的模型：Chinese-LLaMA-2-7B, Chinese-Alpaca-2-7B (更大的模型可先参考[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca))

![](./pics/screencast.gif)

----

[多模态中文LLaMA&Alpaca大模型](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner) | [蒸馏裁剪一体化GRAIN](https://github.com/airaria/GRAIN)


## 新闻

**[2023/08/02] 添加FlashAttention-2训练支持，基于vLLM的推理加速支持，提供长回复系统提示语模板等。详情查看[📚 v1.1版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.1)**

[2023/07/31] 正式发布Chinese-LLaMA-2-7B（基座模型），使用120G中文语料增量训练（与一代Plus系列相同）；进一步通过5M条指令数据精调（相比一代略微增加），得到Chinese-Alpaca-2-7B（指令/chat模型）。详情查看[📚 v1.0版本发布日志](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/releases/tag/v1.0)

[2023/07/19] 🚀启动[中文LLaMA-2、Alpaca-2开源大模型项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)


## 内容导引
| 章节                                  | 描述                                                         |
| ------------------------------------- | ------------------------------------------------------------ |
| [💁🏻‍♂️模型简介](#模型简介) | 简要介绍本项目相关模型的技术特点 |
| [⏬模型下载](#模型下载)        | 中文LLaMA-2、Alpaca-2大模型下载地址          |
| [💻推理与部署](#推理与部署) | 介绍了如何对模型进行量化并使用个人电脑部署并体验大模型 |
| [💯系统效果](#系统效果) | 介绍了模型在部分任务上的效果    |
| [📝训练与精调](#训练与精调) | 介绍了如何训练和精调中文LLaMA-2、Alpaca-2大模型 |
| [❓常见问题](#常见问题) | 一些常见问题的回复 |


## 模型简介

本项目推出了基于Llama-2的中文LLaMA-2以及Alpaca-2系列模型，相比[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)其主要特点如下：

**📖 经过优化的中文词表**

- 在[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)中，我们针对一代LLaMA模型的32K词表扩展了中文字词（LLaMA：49953，Alpaca：49954），以期进一步提升模型对中文文本的编解码效率
- 在本项目中，我们**重新设计了新词表**（大小：55296），进一步提升了中文字词的覆盖程度，同时统一了LLaMA/Alpaca的词表，避免了因混用词表带来的问题

**⚡ 基于FlashAttention-2的高效注意力**

- [FlashAttention-2](https://github.com/Dao-AILab/flash-attention)是高效注意力机制的一种实现，相比其一代技术具有**更快的速度和更优化的显存占用**
- 当上下文长度更长时，为了避免显存爆炸式的增长，使用此类高效注意力技术尤为重要
- 本项目的所有模型均使用了FlashAttention-2技术进行训练

**🚄 基于NTK的自适应上下文扩展技术**

- 在[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)中，我们实现了[基于NTK的上下文扩展技术](https://github.com/ymcui/Chinese-LLaMA-Alpaca/pull/743)，可在不继续训练模型的情况下支持更长的上下文
- 在上述基础上，我们进一步设计了**方便的自适应经验公式**，无需针对不同的上下文长度设置相应超参
- 本项目模型原生支持4K上下文，利用上述技术可扩展至12K，并最高支持扩展至18K+（精度有一定损失）

**🤖 简化的中英双语系统提示语**

- 在[一期项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca)中，中文Alpaca系列模型使用了[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)的指令模板和系统提示语
- 初步实验发现，Llama-2-Chat系列模型的默认系统提示语未能带来统计显著的性能提升，且其内容过于冗长
- 本项目中的Alpaca-2系列模型简化了系统提示语，同时遵循Llama-2-Chat指令模板，以便更好地适配相关生态


## 模型下载

### 模型选择指引

下面是中文LLaMA-2和Alpaca-2模型的基本对比以及建议使用场景。

| 对比项                | 中文LLaMA-2                                            | 中文Alpaca-2                                                 |
| :-------------------- | :----------------------------------------------------- | :----------------------------------------------------------- |
| 训练方式              | 传统CLM                            | 指令精调                                                     |
| 模型类型 | **基座模型** | **指令/Chat模型（类ChatGPT）** |
| 训练语料 | 无标注通用语料 | 有标注指令数据 |
| 词表大小<sup>[1]</sup> | 55,296 | 55,296 |
| 输入模板              | 不需要                                                 | 需要套用特定模板<sup>[2]</sup>，类似Llama-2-Chat |
| 适用场景            | 文本续写：给定上文，让模型生成下文            | 指令理解：问答、写作、聊天、交互等 |
| 不适用场景          | 指令理解 、多轮聊天等                                  |  文本无限制自由生成                                                       |

[1] *本项目一代模型和二代模型的词表不同，请勿混用。二代LLaMA和Alpaca的词表相同。*</br>
[2] *Alpaca-2采用了Llama-2-chat系列模板（格式相同，提示语不同），而不是一代Alpaca的模板，请勿混用。*</br>

### 完整模型下载

以下是完整版模型，直接下载即可使用，无需其他合并步骤。推荐网络带宽充足的用户。

| 模型名称                  |   类型   | 训练数据 | 大小 |                    下载地址                    |
| :------------------------ | :------: | :------: | :----------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-7B | 基座模型 | 120G通用文本 | 13GB | [[百度网盘]](https://pan.baidu.com/s/1E5NI3nlQpx1j8z3eIzbIlg?pwd=n8k3)<br/>[[Google Drive]](https://drive.google.com/drive/folders/18pp4I-mvQxRA7b8vF9gP-2cH_ocnXVKh?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-llama-2-7b) |
| Chinese-Alpaca-2-7B | 指令模型 | 5M条指令 | 13GB | [[百度网盘]](https://pan.baidu.com/s/1wxx-CdgbMupXVRBcaN4Slw?pwd=kpn9)<br/>[[Google Drive]](https://drive.google.com/drive/folders/1JsJDVs7tE2y31PBNleBlDPsB7S0ZrY8d?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-alpaca-2-7b) |

### LoRA模型下载

以下是LoRA模型，与上述完整模型一一对应。需要注意的是**LoRA模型无法直接使用**，必须按照教程与重构模型进行合并。推荐网络带宽不足，手头有原版Llama-2且需要轻量下载的用户。

| 模型名称                  |   类型   | 训练数据 |                   重构模型                   | 大小 |                    LoRA下载地址                    |
| :------------------------ | :------: | :------: | :--------------------------------------------------------: | :----------------: | :----------------------------------------------------------: |
| Chinese-LLaMA-2-LoRA-7B | 基座模型 | 120G通用文本 |        [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)        | 1.1GB | [[百度网盘]](https://pan.baidu.com/s/1bmgqdyRh9E3a2uqOGyNqiQ?pwd=7kvq)<br/>[[Google Drive]](https://drive.google.com/file/d/1njJGSU_PRbzjYRNw5RSbC5-4fBOXTVY3/view?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-llama-2-lora-7b) |
| Chinese-Alpaca-2-LoRA-7B | 指令模型 | 5M条指令 | [Llama-2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | 1.1GB | [[百度网盘]](https://pan.baidu.com/s/1g0olPxkB_rlZ9UUVfOnbcw?pwd=5e7w)<br/>[[Google Drive]](https://drive.google.com/file/d/1MzJL-ZIzdJW7MIcAiYIDIDJ5dlMi8Kkk/view?usp=share_link)<br/>[[HuggingFace]](https://huggingface.co/ziqingyang/chinese-alpaca-2-lora-7b) |

由于LoRA模型无法单独使用，必须与原版Llama-2进行合并才能转为完整模型，以便进行模型推理、量化或者进一步训练。请选择以下方法对模型进行转换合并。

- [**在线转换**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/online_conversion_zh)：Colab用户可利用本项目提供的notebook进行在线转换并量化模型
- [**手动转换**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/manual_conversion_zh)：离线方式转换，生成不同格式的模型，以便进行量化或进一步精调


## 推理与部署

本项目中的相关模型主要支持以下量化、推理和部署方式。

| 工具                                                         | 特点                         | CPU  | GPU  | 量化 | GUI  | API  | vLLM |                             教程                             |
| :----------------------------------------------------------- | ---------------------------- | :--: | :--: | :--: | :--: | :--: | :--: | :----------------------------------------------------------: |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp)      | 丰富的量化选项和高效本地推理 |  ✅   |  ✅   |  ✅   |  ❌   |  ✅   |  ❌   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_zh) |
| [**🤗Transformers**](https://github.com/huggingface/transformers) | 原生transformers推理接口     |  ✅   |  ✅   |  ✅   |  ✅   |  ❌   |  ✅  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/inference_with_transformers_zh) |
| [**Colab Demo**](https://colab.research.google.com/drive/1yu0eZ3a66by8Zqm883LLtRQrguBAb9MR?usp=sharing) | 在Colab中启动交互界面 | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | [link](https://colab.research.google.com/drive/1yu0eZ3a66by8Zqm883LLtRQrguBAb9MR?usp=sharing) |
| [**仿OpenAI API调用**](https://platform.openai.com/docs/api-reference) | 仿OpenAI API接口的服务器Demo |  ✅   |  ✅   |  ✅   |  ❌   |  ✅   |  ✅  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/api_calls_zh) |
| [**text-generation-webui**](https://github.com/oobabooga/text-generation-webui) | 前端Web UI界面的部署方式 |  ✅   |  ✅   |  ✅   |  ✅   |  ✅   | ❌  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/text-generation-webui_zh) |
| [**LangChain**](https://github.com/hwchase17/langchain) | 适合二次开发的大模型应用开源框架 |  ✅<sup>†</sup>  |  ✅   |  ✅<sup>†</sup>   |  ❌   |  ❌   | ❌  | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/langchain_zh) |
| [**privateGPT**](https://github.com/imartinez/privateGPT) | 基于LangChain的多文档本地问答框架 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/privategpt_zh) |

<sup>†</sup>: LangChain框架支持，但教程中未实现；详细说明请参考LangChain官方文档。


## 系统效果

### 生成效果评测

为了更加直观地了解模型的生成效果，本项目仿照[Fastchat Chatbot Arena](https://chat.lmsys.org/?arena)推出了模型在线对战平台，可浏览和评测模型回复质量。对战平台提供了胜率、Elo评分等评测指标，并且可以查看两两模型的对战胜率等结果。题库来自于[一期项目人工制作的200题](https://github.com/ymcui/Chinese-LLaMA-Alpaca/tree/main/examples/f16-p7b-p13b-33b)，以及在此基础上额外增加的题目。生成回复具有随机性，受解码超参、随机种子等因素影响，因此相关评测并非绝对严谨，结果仅供晾晒参考，欢迎自行体验。部分生成样例请查看[examples目录](./examples)。

测试模型包括：

- [**一期模型**](https://github.com/ymcui/Chinese-LLaMA-Alpaca)：Chinese-Alpaca-Pro系列（7B/13B/33B）、Chinese-Alpaca-Plus系列（7B/13B/33B）
- [**二期模型（本项目）**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)：Chinese-Alpaca-2（7B）

**📊 模型在线对战**：[http://chinese-alpaca-arena.ymcui.com](http://chinese-alpaca-arena.ymcui.com/)

### 客观效果评测

本项目还在“NLU”类客观评测集合上对相关模型进行了测试。这类评测的结果不具有主观性，只需要输出给定标签（需要设计标签mapping策略），因此可以评测大模型的部分NLU能力。本项目在[C-Eval评测数据集](https://cevalbenchmark.com)上测试了相关模型效果，其中验证集包含1.3K个选择题，测试集包含12.3K个选择题，涵盖52个学科。从以下结果可以看出本项目推出的模型相比一期模型具有显著性能优势，甚至在大部分指标上超越了之前的Plus-13B系列模型。

LLaMA系列模型之间对比：

| 模型                   | Valid (zero-shot) | Valid (5-shot) | Test (zero-shot) | Test (5-shot) |
| ---------------------- | :---------------: | :------------: | :--------------: | :-----------: |
| **Chinese-LLaMA-2-7B** |     **28.2**      |    **36.0**    |     **30.3**     |   **34.2**    |
| Chinese-LLaMA-Plus-13B |       27.3        |      34.0      |       27.8       |     33.3      |
| Chinese-LLaMA-Plus-7B  |       27.3        |      28.3      |       26.9       |     28.4      |

Alpaca系列模型之间对比：

| 模型                    | Valid (zero-shot) | Valid (5-shot) | Test (zero-shot) | Test (5-shot) |
| ----------------------- | :---------------: | :------------: | :--------------: | :-----------: |
| **Chinese-Alpaca-2-7B** |       41.3        |    **42.9**    |       40.3       |     39.5      |
| Chinese-Alpaca-Plus-13B |     **43.3**      |      42.4      |     **41.5**     |   **39.9**    |
| Chinese-Alpaca-Plus-7B  |       36.7        |      32.9      |       36.4       |     32.3      |

需要注意的是，综合评估大模型能力仍然是亟待解决的重要课题，单个数据集的结果并不能综合评估模型性能。合理辩证地看待大模型相关评测结果有助于大模型技术的良性发展。推荐用户在自己关注的任务上进行测试，选择适配相关任务的模型。

C-Eval推理代码请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/ceval_zh)

### 量化效果评测

以Chinese-LLaMA-2-7B为例，对比不同精度下的模型大小、PPL（困惑度）、C-Eval效果，方便用户了解量化精度损失。PPL以4K上下文大小计算，C-Eval汇报的是valid集合上zero-shot和5-shot结果。

| 精度      | 模型大小 |  PPL   |   C-Eval    |
| :-------- | :------: | :----: | :---------: |
| FP16      | 12.9 GB  | 8.1797 | 28.2 / 36.0 |
| 8-bit量化 |  6.8 GB  | 8.2884 | 26.8 / 35.4 |
| 4-bit量化 |  3.7 GB  | 8.8581 | 25.5 / 32.8 |

特别地，以下是在llama.cpp下不同量化方法的评测数据，供用户参考，速度以ms/tok计。具体细节见[Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llamacpp_zh#关于量化方法选择及推理速度)。

| | F16       | Q4_0   | Q4_1  | Q4_K  | Q5_0  | Q5_1  | Q5_K  | Q6_K  | Q8_0  |
| --------- | -----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| PPL       | 8.640  | 8.987 | 9.175 | 8.836 | 8.730 | 8.776 | 8.707 | 8.671 | 8.640 |
| Size      | 12.91G | 3.69G | 4.08G | 3.92G | 4.47G | 4.86G | 4.59G | 5.30G | 6.81G |
| CPU Speed | 117    | 39    | 44    | 43    | 48    | 51    | 50    | 54    | 65    |
| GPU Speed | 53     | 17    | 18    | 20    | n/a   | n/a  | 25    | 26    | n/a  |

## 训练与精调

预训练（中文LLaMA-2训练）和指令精调（中文Alpaca-2训练）相关内容请参考对应Wiki。

- **预训练**：代码参考了🤗transformers中的[run_clm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py)，使用方法见[预训练脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/pt_scripts_zh)
- **指令精调**：代码参考了[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)项目中数据集处理的相关部分，使用方法见[指令精调脚本Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/sft_scripts_zh)


## 常见问题

请在提Issue前务必先查看FAQ中是否已存在解决方案。

```
问题1：本项目和一期项目的区别？
问题2：模型能否商用？
问题3：接受第三方Pull Request吗？
问题4：为什么不对模型做全量预训练而是用LoRA？
问题5：二代模型支不支持某些支持一代LLaMA的工具？
```

具体问题和解答请参考本项目 >>> [📚 GitHub Wiki](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/faq_zh)


## 引用

如果您使用了本项目的相关资源，请参考引用本项目的技术报告：https://arxiv.org/abs/2304.08177
```
@article{Chinese-LLaMA-Alpaca,
    title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca},
    author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
    journal={arXiv preprint arXiv:2304.08177},
    url={https://arxiv.org/abs/2304.08177},
    year={2023}
}
```


## 致谢

本项目主要基于以下开源项目二次开发，在此对相关项目和研究开发人员表示感谢。

- [Llama-2 *by Meta*](https://github.com/facebookresearch/llama)
- [llama.cpp *by @ggerganov*](https://github.com/ggerganov/llama.cpp)
- [FlashAttention-2 by *Dao-AILab*](https://github.com/Dao-AILab/flash-attention)

同时感谢Chinese-LLaMA-Alpaca（一期项目）的contributor以及[关联项目和人员](https://github.com/ymcui/Chinese-LLaMA-Alpaca#致谢)。


## 免责声明

本项目基于由Meta发布的Llama-2模型进行开发，使用过程中请严格遵守Llama-2的开源许可协议。如果涉及使用第三方代码，请务必遵从相关的开源许可协议。模型生成的内容可能会因为计算方法、随机因素以及量化精度损失等影响其准确性，因此，本项目不对模型输出的准确性提供任何保证，也不会对任何因使用相关资源和输出结果产生的损失承担责任。如果将本项目的相关模型用于商业用途，开发者应遵守当地的法律法规，确保模型输出内容的合规性，本项目不对任何由此衍生的产品或服务承担责任。

<details>
<summary><b>局限性声明</b></summary>

虽然本项目中的模型具备一定的中文理解和生成能力，但也存在局限性，包括但不限于：

- 可能会产生不可预测的有害内容以及不符合人类偏好和价值观的内容
- 由于算力和数据问题，相关模型的训练并不充分，中文理解能力有待进一步提升
- 暂时没有在线可互动的demo（注：用户仍然可以自行在本地部署）

</details>


## 问题反馈
如有疑问，请在GitHub Issue中提交。礼貌地提出问题，构建和谐的讨论社区。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 提交问题请使用本项目设置的Issue模板，以帮助快速定位具体问题。
- 重复以及与本项目无关的issue会被[stable-bot](https://github.com/marketplace/stale)处理，敬请谅解。
