import json
from fastapi.responses import StreamingResponse, Response
import httpx
from fastapi import FastAPI, HTTPException, Request
import requests
import uvicorn
import traceback
import os
from dotenv import load_dotenv
from gne import GeneralNewsExtractor
from bs4 import BeautifulSoup
import asyncio
from urllib.parse import urlparse

load_dotenv() # 加载环境变量

app = FastAPI()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:1234")
GOOGLE_CX = os.getenv("GOOGLE_CX", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8101")
DEFAULT_SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "google")   # 默认使用 google，可选 "google" 或 "searxng"
NUM_RESULTS = int(os.getenv("NUM_RESULTS", 5))


# print(f"当前环境变量：{OPENAI_BASE_URL},{GOOGLE_CX},{GOOGLE_API_KEY},{SEARXNG_URL},{DEFAULT_SEARCH_ENGINE},{NUM_RESULTS}")

@app.api_route("/v1/models", methods=["GET", "OPTIONS"])
async def get_models(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            # 转发原始请求头
            headers = dict(request.headers)
            headers.pop("host", None)  # 移除 host 头，避免冲突
            
            response = await client.get(
                f"{OPENAI_BASE_URL}/v1/models",
                headers=headers,
                timeout=30.0
            )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/v1/completions", methods=["POST","OPTIONS"])
async def post_completions(request: Request):
    try:
        body = await request.json()
        async with httpx.AsyncClient() as client:
            headers = dict(request.headers)
            headers.pop("host", None)
            
            response = await client.post(
                f"{OPENAI_BASE_URL}/v1/completions",
                headers=headers,
                json=body,
                timeout=30.0
            )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.api_route("/v1/embeddings", methods=["POST","OPTIONS"])
async def post_embeddings(request: Request):
    try:
        body = await request.json()
        async with httpx.AsyncClient() as client:
            headers = dict(request.headers)
            headers.pop("host", None)
            
            response = await client.post(
                f"{OPENAI_BASE_URL}/v1/embeddings",
                headers=headers,
                json=body,
                timeout=30.0
            )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LLM API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def perform_search(query, engine=None):
    """
    统一的搜索函数，根据指定的引擎选择搜索方式
    """
    search_engine = engine or DEFAULT_SEARCH_ENGINE
    if search_engine.lower() == "google":
        return await google_search(query)
    elif search_engine.lower() == "searxng":
        return await searxng_search(query)
    else:
        raise ValueError(f"不支持的搜索引擎: {search_engine}")

@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    client = httpx.AsyncClient()
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    
    try:
        body = await request.json()
        headers = dict(request.headers)
        headers.pop("host", None)
        headers.pop("content-length", None)
        
        stream = body.get("stream", False)
        messages = body.get("messages", [])
        
        # 检查是否是强制搜索
        force_search = False
        search_query = ""
        if messages and messages[-1]["role"] == "user":
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                if any(tag in content.lower() for tag in ["#search", "/search", "#ss", "/ss"]):
                    force_search = True
                    search_query = content
                    for tag in ["#search", "/search", "#ss", "/ss"]:
                        search_query = search_query.replace(tag, "").strip()
                    messages[-1]["content"] = search_query

        async def force_search_stream():
            # 创建新的 httpx 客户端
            async with httpx.AsyncClient() as client:
                # 1. 先用大模型优化搜索关键词
                optimize_messages = messages.copy()
                optimize_messages.append({
                    "role": "system",
                    "content": "请帮我优化以下搜索关键词，使其更容易获得准确的搜索结果。只需要返回优化后的关键词，不需要任何解释。"
                })
                
                nonlocal search_query
                response = await client.post(
                    url,
                    json={"messages": optimize_messages, "stream": False},
                    headers=headers,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and result["choices"]:
                        optimized_query = result["choices"][0]["message"]["content"].strip()
                        search_query = optimized_query
                
                # 2. 执行搜索
                yield "data: {\"choices\":[{\"delta\":{\"content\":\"> 正在联网搜索相关信息...\\n\\n\"},\"finish_reason\":null,\"index\":0}]}\n\n".encode('utf-8')
                
                search_results = await perform_search(search_query)
                
                # 3. 构建最终提示
                final_messages = messages.copy()
                final_messages.insert(-1, {
                    "role": "system",
                    "content": "请基于搜索结果提供详细的回答。在回答的最后，请列出参考资料清单。如果有思考过程，请直接执行工具且不要显示工具调用的相关内容。"
                })
                final_messages.extend([
                    {
                        "role": "system",
                        "content": f"搜索结果：{search_results}"
                    }
                ])
                
                # 4. 发送最终请求并流式返回结果
                async with client.stream(
                    "POST",
                    url,
                    json={"messages": final_messages, "stream": True},
                    headers=headers,
                    timeout=60.0
                ) as response:
                    async for chunk in response.aiter_bytes():
                        yield chunk

        if force_search:
            # 情况1：强制搜索流程
            if stream:
                return StreamingResponse(
                    force_search_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                # 非流式强制搜索处理
                async with httpx.AsyncClient() as client:
                    # ... 非流式处理代码 ...
                    pass
        else:
            # 情况2：自动判断是否需要搜索
            modified_body = inject_tool_definitions(body)
            
            # 直接执行搜索而不等待模型响应
            if stream:
                # 获取用户原始问题
                user_messages = [msg for msg in modified_body["messages"] if msg["role"] == "user"]
                if not user_messages:
                    raise ValueError("未找到用户问题")
                
                query = user_messages[-1]["content"]
                
                # 直接执行搜索流程
                return StreamingResponse(
                    force_search_stream(),  # 重用强制搜索的流程
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                final_response = await client.post(
                    url,
                    json=modified_body,
                    headers=headers,
                    timeout=60.0
                )
                return final_response.json()
                
    except Exception as e:
        print(f"发生异常: {str(e)}")
        print(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.aclose()

async def needs_tool_call(response_data):
    """检查响应是否需要工具调用"""
    if "choices" not in response_data or not response_data["choices"]:
        return False
        
    choice = response_data["choices"][0]
    
    # 检查标准工具调用格式
    if "message" in choice:
        if "tool_calls" in choice["message"]:
            return True
            
        # 检查消息内容中是否包含工具调用标记
        if "content" in choice["message"]:
            content = choice["message"]["content"].lower()
            return "[tool_request" in content
                
    return False

async def handle_tool_call(response_data, original_body):
    """处理工具调用并返回新的消息"""
    try:
        # 获取用户原始问题作为查询
        user_messages = [msg for msg in original_body["messages"] if msg["role"] == "user"]
        if not user_messages:
            raise ValueError("未找到用户问题")
            
        query = user_messages[-1]["content"]
        tool_call_id = f"call_{str(hash(query))[:8]}"
        search_engine = DEFAULT_SEARCH_ENGINE
        
        print(f"执行搜索，引擎：{search_engine}，查询：{query}")
        search_results = await perform_search(query, search_engine)
        
        # 构建工具调用消息
        tool_messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": f"{search_engine}_search",
                        "arguments": json.dumps({"query": query}, ensure_ascii=False)
                    }
                }]
            },
            {
                "role": "tool",
                "content": search_results,
                "tool_call_id": tool_call_id,
                "name": f"{search_engine}_search"
            }
        ]
        
        return tool_messages
        
    except Exception as e:
        print(f"处理工具调用时出错: {str(e)}")
        print(f"响应数据: {json.dumps(response_data, ensure_ascii=False, indent=2)}")
        raise

async def construct_messages_with_search(messages, query, search_results):
    """构建包含搜索结果的消息列表"""
    tool_call_id = f"call_{str(hash(query))[:8]}"
    new_messages = messages.copy()
    new_messages.extend([
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": f"{DEFAULT_SEARCH_ENGINE}_search",
                    "arguments": json.dumps({"query": query}, ensure_ascii=False)
                }
            }]
        },
        {
            "role": "tool",
            "content": search_results,
            "tool_call_id": tool_call_id,
            "name": f"{DEFAULT_SEARCH_ENGINE}_search"
        }
    ])
    return new_messages

def inject_tool_definitions(original_body):
    tools = []
    
    # 根据搜索引擎类型添加对应的工具
    if DEFAULT_SEARCH_ENGINE.lower() == "google":
        tools.append({
            "type": "function",
            "function": {
                "name": "google_search",
                "description": "使用谷歌搜索获取实时信息，如当前日期、天气、新闻、产品信息等",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        })
    elif DEFAULT_SEARCH_ENGINE.lower() == "searxng":
        tools.append({
            "type": "function",
            "function": {
                "name": "searxng_search",
                "description": "使用 SearXNG 搜索引擎获取信息，如当前日期、天气、新闻、产品信息等",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"}
                    },
                    "required": ["query"]
                }
            }
        })
    
    original_body.setdefault("tools", [])
    original_body["tools"].extend(tools)
    original_body["tool_choice"] = "auto"
    return original_body

async def fetch_and_parse_url(url):
    """
    抓取并解析单个URL的内容
    """
    async with httpx.AsyncClient(verify=False, follow_redirects=True) as client:  # 在这里创建新的客户端实例
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0'
            }
            response = await client.get(url, headers=headers, timeout=10.0)
            
            if response.status_code != 200:
                return None

            # 使用 GNE 提取正文
            extractor = GeneralNewsExtractor()
            html = response.text
            result = extractor.extract(html)
            
            # 提取正文内容
            content = result.get('content', '')
            if not content:
                # 如果GNE提取失败，尝试使用BeautifulSoup提取所有正文
                soup = BeautifulSoup(html, 'html.parser')
                # 移除脚本和样式元素
                for script in soup(["script", "style"]):
                    script.decompose()
                content = soup.get_text(separator='\n', strip=True)

            # 清理和限制内容长度
            content = ' '.join(content.split())
            content = content[:2000]  # 限制长度
            print(f"抓取{url} 内容完成")
            return {
                'url': url,
                'content': content,
                'title': result.get('title', '')
            }
            
        except Exception as e:
            print(f"抓取URL {url} 时出错: {str(e)}")
            return None

async def enrich_search_results(results):
    """
    增强搜索结果，添加网页正文内容
    """
    if isinstance(results, str):
        results = json.loads(results)
    tasks = []
    for result in results:
        url = result.get('link')
        if url and is_valid_url(url):
            tasks.append(fetch_and_parse_url(url))
        
    # 并发抓取所有URL
    enriched_contents = await asyncio.gather(*tasks)
        
    # 合并结果
    for i, content in enumerate(enriched_contents):
        if content:
            results[i]['extracted_content'] = content.get('content', '')
             
   
    return json.dumps(results, ensure_ascii=False)

def is_valid_url(url):
    """
    检查URL是否有效且适合抓取
    """
    try:
        parsed = urlparse(url)
        # 排除不需要的文件类型
        excluded_extensions = ('.pdf', '.doc', '.docx', '.xls', '.xlsx', '.zip', '.rar')
        return all([
            parsed.scheme in ('http', 'https'),
            not any(url.lower().endswith(ext) for ext in excluded_extensions)
        ])
    except:
        return False

# 修改 google_search 函数
async def google_search(query: str) -> str:
    """执行 Google 搜索"""
    try:
        print(f"开始执行 Google 搜索，查询词：{query}")
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": GOOGLE_CX,
            "q": query,
            "num": NUM_RESULTS
        }
        
        print(f"请求 URL（已隐藏敏感信息）: {url}?key=***&q={query}&cx=***&num={NUM_RESULTS}")
        
        # 配置 SSL 验证和重试策略
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,  # 总重试次数
            backoff_factor=1,  # 重试间隔
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码
        )
        adapter = requests.adapters.HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=100,
            pool_maxsize=100
        )
        session.mount("https://", adapter)
        
        # 发送请求
        response = session.get(
            url,
            params=params,
            timeout=30,
            verify=True  # 使用系统的证书验证
        )
        
        response.raise_for_status()
        results = response.json()
        
        if "items" not in results:
            return "未找到相关搜索结果。"
        
        formatted_results = []
        for item in results["items"]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            formatted_results.append(f"标题：{title}\n摘要：{snippet}\n链接：{link}\n")
        
        return "\n".join(formatted_results)
        
    except requests.exceptions.SSLError as e:
        print(f"SSL错误：{str(e)}")
        # 尝试不验证 SSL 证书重试一次
        try:
            session = requests.Session()
            response = session.get(
                url,
                params=params,
                timeout=30,
                verify=False  # 禁用 SSL 验证
            )
            response.raise_for_status()
            results = response.json()
            
            if "items" not in results:
                return "未找到相关搜索结果。"
            
            formatted_results = []
            for item in results["items"]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                link = item.get("link", "")
                formatted_results.append(f"标题：{title}\n摘要：{snippet}\n链接：{link}\n")
            
            return "\n".join(formatted_results)
            
        except Exception as retry_e:
            print(f"重试失败：{str(retry_e)}")
            return f"搜索过程发生异常：{str(e)}"
            
    except Exception as e:
        print(f"错误：搜索过程发生异常：{str(e)}")
        print(f"异常堆栈：{traceback.format_exc()}")
        return f"搜索过程发生异常：{str(e)}"

# 同样修改 searxng_search 函数
async def searxng_search(query):
    print(f"开始执行 SearXNG 搜索，查询词：{query}")
    try:
        params = {
            'q': query,
            'format': 'json',
            'pageno': 1,
            'num_results': NUM_RESULTS
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(SEARXNG_URL + "/search", params=params)
            print(f"API 响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(data)
                results = []
                
                if 'results' in data:
                    print(f"找到搜索结果数量: {len(data['results'])}")
                    if len(data['results']) <= NUM_RESULTS: # 避免过多内容导致超时
                        num = -1
                    else:
                        num = NUM_RESULTS
                    for i, item in enumerate(data['results'][:num], 1):
                        result = {
                            "title": item.get("title"),
                            "link": item.get("url"),
                            "snippet": item.get("content")
                        }
                        results.append(result)
                        print(f"处理第 {i} 条结果: {result['title']}")
                
                # 增强搜索结果
                enriched_results = await enrich_search_results(results)
                print(f"完成内容增强，返回结果")
                return enriched_results
            else:
                error_msg = f"搜索请求失败，状态码: {response.status_code}"
                print(error_msg)
                return error_msg
    except Exception as e:
        error_msg = f"搜索过程发生异常：{str(e)}"
        print(f"错误：{error_msg}")
        print(f"异常堆栈：{traceback.format_exc()}")
        return error_msg
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8100, reload=True)