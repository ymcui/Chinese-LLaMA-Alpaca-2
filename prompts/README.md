## 系统指令 System Prompts

### alpaca-2.txt (default)

这个文件是训练时采用的默认系统指令，内容极简，因此回复长度上略短于一代Pro系列模型。

This file is the default system prompt used in the SFT phase, which is simple. Thus, the length of the response may be shorter than 1st-gen Pro series models.

### alpaca-2-long.txt

这个文件是增加模型回复内容长度的系统指令示例，用户可根据实际情况自行参照修改。但建议保留最原始的`alpaca-2.txt`中的内容，在此基础上进行自定义系统指令的编写。

This file is an improved system prompt sample to extend the response length. The users can modify this prompt if necessary. However, we suggest keep the original content in `alpaca-2.txt` and add your customized prompt based on this.
