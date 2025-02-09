# LLM Search Plus

让LLM支持联网搜索

[中文](./README_CN.md) | [English](./README.md)

目前支持搜索引擎：

- Google
- SearxNG

本地模型服务：

- LM Studio

或者其他兼容OpenAI API的服务。

### 使用方法：

前提：本地已有Python环境，或者自行安装：https://www.anaconda.com/download/success，建议下载Miniconda

**拷贝项目到本地：**

    git clone git@github.com:nocmt/LLMSearchPlus.git

**安装环境：**

    pip install -r requirements.txt

**将.env.template 复制或者重命名成 .env，修改其中的配置。**

**运行：**

    python main.py


在Chat客户端使用，URL填写：http://127.0.0.1:8100，可能要加/v1，和使用LM Studio并没有什么区别，请自行测试确定。


### 联网搜索逻辑：

发送内容中携带 #search或者/search 强制开启联网搜索，否则由模型自行判断是否要联网搜索。