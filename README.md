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


本项目基于Meta发布的可商用大模型[Llama-2](https://github.com/facebookresearch/llama)开发，是[中文LLaMA&Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca)的第二期项目。

**本项目主要内容：**TBA

![](./pics/screencast.gif)

----

[中文LLaMA&Alpaca大模型](https://github.com/ymcui/Chinese-LLaMA-Alpaca) | [多模态中文LLaMA&Alpaca大模型](https://github.com/airaria/Visual-Chinese-LLaMA-Alpaca) | [多模态VLE](https://github.com/iflytek/VLE) | [中文MiniRBT](https://github.com/iflytek/MiniRBT) | [中文LERT](https://github.com/ymcui/LERT) | [中英文PERT](https://github.com/ymcui/PERT) | [中文MacBERT](https://github.com/ymcui/MacBERT) | [中文ELECTRA](https://github.com/ymcui/Chinese-ELECTRA) | [中文XLNet](https://github.com/ymcui/Chinese-XLNet) | [中文BERT](https://github.com/ymcui/Chinese-BERT-wwm) | [知识蒸馏工具TextBrewer](https://github.com/airaria/TextBrewer) | [模型裁剪工具TextPruner](https://github.com/airaria/TextPruner) | [蒸馏裁剪一体化GRAIN](https://github.com/airaria/GRAIN)


## 新闻

[2023/07/19] 🚀 正式启动[中文LLaMA-2、Alpaca-2开源大模型项目](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)


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

TBA


## 模型下载

### 模型选择指引

下面是中文LLaMA-2和Alpaca-2模型的基本对比以及建议使用场景。

### 完整模型下载

以下是完整版模型，直接下载即可使用，无需其他合并步骤。推荐网络带宽充足的用户。

### LoRA模型下载

以下是LoRA模型，**无法直接使用，必须按照教程与重构模型进行合并**。推荐网络带宽不足，手头有原版Llama-2且需要轻量下载的用户。

由于LoRA模型无法单独使用，必须与原版LLaMA-2进行合并才能转为完整模型，以便进行模型推理、量化或者进一步训练。请选择以下方法对模型进行转换合并。

- [**在线转换**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/在线模型合并与转换)：Colab用户可利用本项目提供的notebook进行在线转换并量化模型
- [**手动转换**](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/手动模型合并与转换)：离线方式转换，生成不同格式的模型，以便进行量化或进一步精调

以下是完整模型在FP16精度和4-bit量化后的大小。如果选择手动合并，请确保本机有足够的内存和磁盘空间。


## 推理与部署

本项目中的模型主要支持以下量化、推理和部署方式。

| 工具                                                         | 特点                         | CPU  | GPU  | 量化 | GUI  | API  |                             教程                             |
| :----------------------------------------------------------- | ---------------------------- | :--: | :--: | :--: | :--: | :--: | :----------------------------------------------------------: |
| [**llama.cpp**](https://github.com/ggerganov/llama.cpp)      | 丰富的量化选项和高效本地推理 |  ✅   |  ✅   |  ✅   |  ❌   |  ✅   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/llama.cpp量化部署) |
| [**🤗Transformers**](https://github.com/huggingface/transformers) | 原生transformers推理接口     |  ✅   |  ✅   |  ✅   |  ✅   |  ❌   | [link](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/wiki/使用Transformers推理) |

一代模型相关推理与部署支持将陆续迁移到本项目，届时将同步更新相关教程。


## 系统效果

### 生成效果评测

**📊 模型在线对战**：[http://chinese-alpaca-arena.ymcui.com](http://chinese-alpaca-arena.ymcui.com/)

### 客观效果评测

TBA


## 训练与精调

TBA


## 常见问题

TBA


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

TBA


## 免责声明

TBA


## 问题反馈
如有疑问，请在GitHub Issue中提交。礼貌地提出问题，构建和谐的讨论社区。

- 在提交问题之前，请先查看FAQ能否解决问题，同时建议查阅以往的issue是否能解决你的问题。
- 提交问题请使用本项目设置的Issue模板，以帮助快速定位具体问题。
- 重复以及与本项目无关的issue会被[stable-bot](https://github.com/marketplace/stale)处理，敬请谅解。
