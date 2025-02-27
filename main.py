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
    url = f"{OPENAI_BASE_URL}/v1/chat/completions"
    
    try:
        # 添加错误处理的日志
        request_body = await request.body()
        try:
            body = json.loads(request_body)
        except json.JSONDecodeError as e:
            print(f"JSON 解析错误: {str(e)}")
            print(f"原始请求体: {request_body}")
            raise HTTPException(status_code=400, detail=f"无效的 JSON 格式: {str(e)}")

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

        if force_search:
            # 情况1：强制搜索流程
            if stream:
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
                        yield "data: {\"choices\":[{\"delta\":{\"content\":\"搜索中...\\n\\n\"},\"finish_reason\":null,\"index\":0}]}\n\n".encode('utf-8')
                        
                        search_results = await perform_search(search_query)
                        
                        # 3. 构建最终提示
                        final_messages = messages.copy()
                        final_messages.extend([
                                            {
                                                "role": "system",
                                                "content": f"搜索结果：{search_results}，请基于搜索结果提供清晰明确的回答。回复内容最后必须按照markdown格式完整列出所有参考资料，格式要求：\n1. 必须包含所有搜索结果\n2. 每行一个链接\n3. 格式为：[序号. 标题](链接)\n4. 标题中如果有序号，请去除序号"
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
            # 情况2：让模型判断是否需要搜索
            modified_body = inject_tool_definitions(body)
            
            if stream:
                async def process_stream():
                    client = httpx.AsyncClient()
                    try:
                        async with client.stream(
                            "POST",
                            url,
                            json=modified_body,
                            headers=headers,
                            timeout=60.0
                        ) as response:
                            need_search = False
                            tool_calls_buffer = []
                            content_buffer = []
                            
                            async for chunk in response.aiter_bytes():
                                try:
                                    chunk_str = chunk.decode('utf-8')
                                    if not chunk_str.strip() or chunk_str.strip() == "data: [DONE]":
                                        continue
                                        
                                    if chunk_str.startswith('data: '):
                                        data = json.loads(chunk_str[6:])
                                        if "choices" in data and data["choices"]:
                                            choice = data["choices"][0]
                                            
                                            # 检查是否有工具调用
                                            if "delta" in choice:
                                                delta = choice["delta"]
                                                if "tool_calls" in delta:
                                                    need_search = True
                                                    tool_calls_buffer.append(chunk)
                                                    continue
                                                elif "content" in delta:
                                                    content_buffer.append(chunk)
                                                    
                                except Exception as e:
                                    print(f"处理流数据时出错: {str(e)}")
                                    continue
                            
                            # 如果需要搜索
                            if need_search:
                                user_messages = [msg for msg in modified_body["messages"] if msg["role"] == "user"]
                                if user_messages:
                                    query = user_messages[-1]["content"]
                                    # 发送搜索提示
                                    yield "data: {\"choices\":[{\"delta\":{\"content\":\"Searching...\\n\\n\"},\"finish_reason\":null,\"index\":0}]}\n\n".encode('utf-8')
                                    
                                    search_results = await perform_search(query)
                                    final_messages = modified_body["messages"].copy()
                                    final_messages.extend([
                                            {
                                                "role": "system",
                                                "content": f"搜索结果：{search_results}，请基于搜索结果提供清晰明确的回答。回复内容最后必须按照markdown格式完整列出所有参考资料，格式要求：\n1. 必须包含所有搜索结果\n2. 每行一个链接\n3. 格式为：[序号. 标题](链接)\n4. 标题中如果有序号，请去除序号"
                                            }
                                    ])
                                    # 发送带有搜索结果的最终请求
                                    async with client.stream(
                                        "POST",
                                        url,
                                        json={"messages": final_messages, "stream": True},
                                        headers=headers,
                                        timeout=60.0
                                    ) as final_response:
                                        async for final_chunk in final_response.aiter_bytes():
                                            yield final_chunk
                            else:
                                # 如果不需要搜索，返回原始内容
                                for chunk in content_buffer:
                                    yield chunk
                                yield b"data: [DONE]\n\n"
                                    
                    finally:
                        await client.aclose()
                
                return StreamingResponse(
                    process_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    }
                )
            else:
                # 非流式请求处理
                async with httpx.AsyncClient() as client:  # 非流式请求使用上下文管理器
                    response = await client.post(
                        url,
                        json=modified_body,
                        headers=headers,
                        timeout=60.0
                    )
                    
                    response_data = response.json()
                    
                    if await needs_tool_call(response_data):
                        tool_messages = await handle_tool_call(response_data, modified_body)
                        
                        final_messages = modified_body["messages"].copy()
                        final_messages.extend(tool_messages)
                        
                        final_response = await client.post(
                            url,
                            json={"messages": final_messages},
                            headers=headers,
                            timeout=60.0
                        )
                        return final_response.json()
                    
                    return response_data

    except Exception as e:
        print(f"发生异常: {str(e)}")
        print(f"异常堆栈: {traceback.format_exc()}")

        raise HTTPException(status_code=500, detail=str(e))

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
            print(f"抓取{url} 内容完成，返回结果")
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
async def google_search(query):
    print(f"开始执行 Google 搜索，查询词：{query}")
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&q={query}&cx={GOOGLE_CX}&num={NUM_RESULTS}"
        print(f"请求 URL（已隐藏敏感信息）: https://www.googleapis.com/customsearch/v1?key=***&q={query}&cx=***&num={NUM_RESULTS}")
        
        response = requests.get(url)
        print(f"API 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = []
            if 'items' in data:
                print(f"找到搜索结果数量: {len(data['items'])}")
                for i, item in enumerate(data['items'], 1):
                    result = {
                        "title": item.get("title"),
                        "link": item.get("link"),
                        "snippet": item.get("snippet")
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