# LLM Search Plus

让LLM支持联网搜索

效果：

![演示](./演示.gif)

## 联网搜索逻辑：

Chat客户端发送内容中携带 #search、/search、/ss 或 #ss 强制开启联网搜索，否则由模型自行判断。

## 目前支持搜索引擎：

- Google ：[google custom search api 申请注册 cx key](https://blog.csdn.net/whatday/article/details/113750998)
- SearxNG

## 本地模型服务：

- LM Studio

或者其他兼容OpenAI API的服务。

## 使用方法


如果之前已经部署Searxng或者申请过谷歌搜索引擎的相关CX和KEY，那建议修改.env环境变量直接本地运行（本地运行需要python环境，建议去官网安装），否则直接用Docker Compose部署，简单快捷。


### 1. 本地运行

前提：本地已有Python环境，或者自行安装：[https://www.anaconda.com/download/success](https://www.anaconda.com/download/success)，建议下载Miniconda。

**拷贝项目到本地：**

    git clone https://github.com/nocmt/LLMSearchPlus.git

**安装环境：**

    cd LLMSearchPlus
    pip install -r requirements.txt
    cp .env.template .env

修改.env其中的配置。

环境变量的含义：

|  kEY   | 默认值  | 含义  |
|  ----  | ----  | ----  |
| SEARCH_ENGINE  | searxng |目前只支持google、searxng  |
| OPENAI_BASE_URL  | http://127.0.0.1:1234 | 使用LM-Studio启动服务，并且打开`允许在局域网内提供服务`、`启用 CORS`。  |
| NUM_RESULTS  | 5 |搜索返回条数，尽量避免过多内容导致超时  |
| GOOGLE_CX  | "" |谷歌搜索CX值，SEARCH_ENGINE=google时必填  |
| GOOGLE_API_KEY  | "" |谷歌搜索API KEY值，SEARCH_ENGINE=google时必填  |
| SEARXNG_URL  | http://127.0.0.1:8101 |SEARCH_ENGINE=searxng时必填  |


**运行：**

    python main.py


在Chat客户端使用，URL填写：[http://127.0.0.1:8100](http://127.0.0.1:8100)，可能要加/v1，和使用LM Studio并没有什么区别，请自行测试确定。


![配置](配置.png)


#### 2. 使用Docker Compose部署

**修改配置文件**

docker-compose.yml 文件中，需要修改LM Studio的局域网IP地址(使用LM-Studio启动服务，并且打开`允许在局域网内提供服务`、`启用 CORS`)，建议和SEARXNG一起部署，这样SEARXNG的地址就是[http://searxng:8080](http://searxng:8080)，否则需要自行处理容器网络问题。


searxng 的配置文件，具体在 `.searxng` 目录下，有一个`secret_key`需要手动生成，根据系统的不同执行命令不一样，只需要生成1次。


Windows 用户可以使用以下 powershell 命令生成密钥：


```powershell
$randomBytes = New-Object byte[] 32
(New-Object Security.Cryptography.RNGCryptoServiceProvider).GetBytes($randomBytes)
$secretKey = -join ($randomBytes | ForEach-Object { "{0:x2}" -f $_ })
(Get-Content .searxng/settings.yml) -replace 'ultrasecretkey', $secretKey | Set-Content .searxng/settings.yml
```

Linux、Mac 用户可以使用以下 bash 命令生成密钥：

```bash

sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|g" .searxng/settings.yml

```


这样它会生成一个32位的密钥，并自动替换掉ultrasecretkey。


全部修改好后，启动服务：


**启动服务：**

    docker compose up -d


如果返回ERROR: failed to authorize之类的错误，那就先pull python镜像下来再执行 `docker compose up -d`。

    docker pull python:3.11-slim


**查看日志：**

    docker-compose logs -f

**停止删除服务：**

    docker-compose down


## 其他


### 自行构建镜像

#### 构建镜像

    docker build -t llm-search-plus .

#### 启动容器

    docker run -d -p 8100:8100 --name llm-search-plus llm-search-plus


### Searxng优化

如果发现联网搜索的内容不对或者不全面，建议修改它使用的搜索引擎，打开[http://localhost:8101/preferences](http://localhost:8101/preferences) 把一些搜索引擎关闭掉，比如我就只保留了Bing和Google，返回的结果就是正确的了。