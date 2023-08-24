# OPENAI API DEMO

> 更加详细的OPENAI API信息：<https://platform.openai.com/docs/api-reference>

这是一个使用fastapi实现的简易的仿OPENAI API风格的服务器DEMO，您可以使用这个API DEMO来快速搭建基于中文大模型的个人网站以及其他有趣的WEB DEMO。

本实现基于vLLM部署LLM后端服务，暂不支持加载LoRA模型、仅CPU部署和使用8bit推理。

## 部署方式

安装依赖
``` shell
pip install fastapi uvicorn shortuuid vllm fschat
```

启动脚本
``` shell
python scripts/openai_server_demo/openai_api_server_vllm.py --model /path/to/base_model --tokenizer-mode slow --served-model-name chinese-llama-alpaca-2
```

### 参数说明

`--model {base_model}`: 存放HF格式的LLaMA-2模型权重和配置文件的目录，可以是合并后的中文Alpaca-2模型

`--tokenizer {tokenizer_path}`: 存放对应tokenizer的目录。若不提供此参数，则其默认值与`--base_model`相同

`--tokenizer-mode {tokenizer-mode}`: tokenizer的模式。使用基于LLaMA/LLaMa-2的模型时，固定为`slow`

`--tensor-parallel-size {tensor_parallel_size}`: 使用的GPU数量。默认为1

`--served-model-name {served-model-name}`: API中使用的模型名。若使用中文Alpaca-2系列模型，模型名中务必包含`chinese-llama-alpaca-2`

`--host {host_name}`: 部署服务的host name。默认值是`localhost`

`--port {port}`: 部署服务的端口号。默认值是`8000`

## API文档

### 文字接龙（completion）

> 有关completion的中文翻译，李宏毅教授将其翻译为文字接龙 <https://www.youtube.com/watch?v=yiY4nPOzJEg>

最基础的API接口，输入prompt，输出语言大模型的文字接龙（completion）结果。

API DEMO内置有prompt模板，prompt将被套入instruction模板中，这里输入的prompt应更像指令而非对话。

#### 快速体验completion接口

请求command：

``` shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chinese-llama-alpaca-2",
    "prompt": "告诉我中国的首都在哪里"
  }'
```

json返回体：

``` json
{
    "id": "cmpl-41234d71fa034ec3ae90bbf6b5be7",
    "object": "text_completion",
    "created": 1690870733,
    "model": "chinese-llama-alpaca-2",
    "choices": [
        {
            "index": 0,
            "text": "中国的首都是北京。"
        }
    ]
}
```

#### completion接口高级参数

请求command：

``` shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chinese-llama-alpaca-2",
    "prompt": "告诉我中国和美国分别各有哪些优点缺点",
    "max_tokens": 90,
    "temperature": 0.7,
    "num_beams": 4,
    "top_k": 40
  }'
```

json返回体：

``` json
{
    "id": "cmpl-ceca9906bf0a429989e850368cc3f893",
    "object": "text_completion",
    "created": 1690870952,
    "model": "chinese-llama-alpaca-2",
    "choices": [
        {
            "index": 0,
            "text": "中国的优点是拥有丰富的文化和历史，而美国的优点是拥有先进的科技和经济体系。"
        }
    ]
}
```

#### completion接口高级参数说明

> 有关Decoding策略，更加详细的细节可以参考 <https://towardsdatascience.com/the-three-decoding-methods-for-nlp-23ca59cb1e9d> 该文章详细讲述了三种LLaMA会用到的Decoding策略：Greedy Decoding、Random Sampling 和 Beam Search，Decoding策略是top_k、top_p、temperature等高级参数的基础。

`prompt`: 生成文字接龙（completion）的提示。

`max_tokens`: 新生成的句子的token长度。

`temperature`: 在0和2之间选择的采样温度。较高的值如0.8会使输出更加随机，而较低的值如0.2则会使其输出更具有确定性。temperature越高，使用随机采样最为decoding的概率越大。

`use_beam_search`: 使用束搜索（beam search）。默认为`false`，即启用随机采样策略（random sampling）

`n`: 输出序列的数量，默认为1

`best_of`: 当搜索策略为束搜索（beam search）时，该参数为在束搜索（beam search）中所使用的束个数。默认和`n`相同

`top_k`: 在随机采样（random sampling）时，前top_k高概率的token将作为候选token被随机采样。

`top_p`: 在随机采样（random sampling）时，累积概率超过top_p的token将作为候选token被随机采样，越低随机性越大，举个例子，当top_p设定为0.6时，概率前5的token概率分别为{0.23, 0.20, 0.18, 0.11, 0.10}时，前三个token的累积概率为0.61，那么第4个token将被过滤掉，只有前三的token将作为候选token被随机采样。

`presence_penalty`: 重复惩罚，取值范围-2 ~ 2，默认值为0。值大于0表示鼓励模型使用新的token，反之鼓励重复。

`stream`: 设置为`true`时，按流式输出的形式返回。默认为`false`。


### 聊天（chat completion）

聊天接口支持多轮对话

#### 快速体验聊天接口

请求command：

``` shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chinese-llama-alpaca-2",
    "messages": [
      {"role": "user","content": "给我讲一些有关杭州的故事吧"}
    ]
  }'
```

json返回体：

``` json
{
    "id": "cmpl-8fc1b6356cf64681a41a8739445a8cf8",
    "object": "chat.completion",
    "created": 1690872695,
    "model": "chinese-llama-alpaca-2",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "好的，请问您对杭州有什么特别的偏好吗？"
            }
        }
    ]
}
```

#### 多轮对话

请求command：

``` shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chinese-llama-alpaca-2",
    "messages": [
      {"role": "user","content": "给我讲一些有关杭州的故事吧"},
      {"role": "assistant","content": "好的，请问您对杭州有什么特别的偏好吗？"},
      {"role": "user","content": "我比较喜欢和西湖，可以给我讲一下西湖吗"}
    ],
    "repetition_penalty": 1.0
  }'
```

json返回体：

``` json
{
    "id": "cmpl-02bf36497d3543c980ca2ae8cc4feb63",
    "object": "chat.completion",
    "created": 1690872676,
    "model": "chinese-llama-alpaca-2",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "是的，西湖是杭州最著名的景点之一，它被誉为“人间天堂”。 <\\s>"
            }
        }
    ]
}
```

#### 聊天接口高级参数说明

`prompt`: 生成文字接龙（completion）的提示。

`max_tokens`: 新生成的句子的token长度。

`temperature`: 在0和2之间选择的采样温度。较高的值如0.8会使输出更加随机，而较低的值如0.2则会使其输出更具有确定性。temperature越高，使用随机采样最为decoding的概率越大。

`use_beam_search`: 使用束搜索（beam search）。默认为`false`，即启用随机采样策略（random sampling）

`n`: 输出序列的数量，默认为1

`best_of`: 当搜索策略为束搜索（beam search）时，该参数为在束搜索（beam search）中所使用的束个数。默认和`n`相同

`top_k`: 在随机采样（random sampling）时，前top_k高概率的token将作为候选token被随机采样。

`top_p`: 在随机采样（random sampling）时，累积概率超过top_p的token将作为候选token被随机采样，越低随机性越大，举个例子，当top_p设定为0.6时，概率前5的token概率分别为{0.23, 0.20, 0.18, 0.11, 0.10}时，前三个token的累积概率为0.61，那么第4个token将被过滤掉，只有前三的token将作为候选token被随机采样。

`presence_penalty`: 重复惩罚，取值范围-2 ~ 2，默认值为0。值大于0表示鼓励模型使用新的token，反之鼓励重复。

`stream`: 设置为`true`时，按流式输出的形式返回。默认为`false`。
