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
NUM_RESULTS = os.getenv("NUM_RESULTS", 5) 


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
        raise HTTPException(status_code=504, detail="LM Studio API timeout")
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
        raise HTTPException(status_code=504, detail="LM Studio API timeout")
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
        raise HTTPException(status_code=504, detail="LM Studio API timeout")
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
        
        # 检查是否包含强制搜索的关键词
        force_search = False
        if "messages" in body and body["messages"]:
            last_message = body["messages"][-1]
            if last_message.get("role") == "user":
                content = last_message.get("content", "")
                if isinstance(content, str):
                    # 检查是否包含强制搜索标记（比如 #search 或 /search）
                    if "#search" in content.lower() or "/search" in content.lower() or "#ss" in content.lower() or "/ss" in content.lower():
                        force_search = True
                        # 移除搜索标记
                        content = content.replace("#search", "").replace("/search", "").replace("#ss", "").replace("/ss", "").strip()
                        body["messages"][-1]["content"] = content
        
        modified_body = inject_tool_definitions(body)
        
        # 如果强制搜索，直接构造搜索请求
        if force_search:
            search_query = body["messages"][-1]["content"]
            search_results = await perform_search(search_query)
            
            # 构建包含搜索结果的新消息列表
            new_messages = body["messages"].copy()
            tool_call_id = f"call_{str(hash(search_query))[:8]}"
            
            # 添加工具调用消息
            new_messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": f"{DEFAULT_SEARCH_ENGINE}_search",
                        "arguments": json.dumps({"query": search_query}, ensure_ascii=False)
                    }
                }]
            })
            
            # 添加搜索结果消息
            new_messages.append({
                "role": "tool",
                "content": search_results,
                "tool_call_id": tool_call_id,
                "name": f"{DEFAULT_SEARCH_ENGINE}_search"
            })
            
            modified_body["messages"] = new_messages
        
        stream = modified_body.get("stream", False)
        
        print("发送到 LM Studio 的请求体:", json.dumps(modified_body, ensure_ascii=False, indent=2))
        
        # 如果是流式请求，先关闭它以处理工具调用
        if stream:
            modified_body["stream"] = False
        
        response = await client.post(url, json=modified_body, timeout=60.0)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LM Studio API error: {response.text}"
            )
        
        response_data = response.json()
        print("LM Studio 原始返回数据:", json.dumps(response_data, ensure_ascii=False, indent=2))
        
        # 检查是否包含工具调用
        has_tool_calls = (
            "choices" in response_data 
            and response_data["choices"] 
            and "message" in response_data["choices"][0] 
            and "tool_calls" in response_data["choices"][0]["message"]
        )
        
        has_function_calling = (
            "choices" in response_data 
            and response_data["choices"] 
            and "message" in response_data["choices"][0] 
            and "content" in response_data["choices"][0]["message"] 
            and "<function_calling>" in response_data["choices"][0]["message"]["content"]
        )
        
        if has_tool_calls or has_function_calling:
            try:
                query = None
                tool_call_id = None
                search_engine = None
                
                if has_tool_calls:
                    print("检测到标准工具调用格式")
                    tool_call = response_data["choices"][0]["message"]["tool_calls"][0]
                    tool_call_id = tool_call["id"]
                    function_name = tool_call["function"]["name"]
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args["query"]
                    search_engine = function_name.replace("_search", "")
                elif has_function_calling:
                    print("检测到特殊函数调用格式")
                    content = response_data["choices"][0]["message"]["content"]
                    json_str = content.split("<function_calling>")[1].strip()
                    function_data = json.loads(json_str)
                    query = function_data["params"]["query"]
                    tool_call_id = f"call_{str(hash(content))[:8]}"
                    search_engine = DEFAULT_SEARCH_ENGINE
                
                print(f"使用搜索引擎: {search_engine}")  # 添加调试日志
                
                if query:
                    search_results = await perform_search(query, search_engine)
                    print(f"搜索结果: {search_results}")
                    
                    # 构建新的消息列表
                    new_messages = modified_body["messages"].copy()
                    tool_call = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": f"{search_engine}_search",
                            "arguments": json.dumps({"query": query}, ensure_ascii=False)
                        }
                    }
                    
                    new_messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    })
                    new_messages.append({
                        "role": "tool",
                        "content": search_results,
                        "tool_call_id": tool_call_id,
                        "name": f"{search_engine}_search"
                    })
                    
                    # 构建新的请求
                    new_request = {
                        "model": modified_body.get("model", "gpt-3.5-turbo"),
                        "messages": new_messages,
                        "stream": stream
                    }
                    
                    print("发送最终请求到 LM Studio...")
                    final_response = await client.post(
                        f"{OPENAI_BASE_URL}/v1/chat/completions",
                        json=new_request,
                        timeout=60.0
                    )
                    
                    if final_response.status_code == 200:
                        if stream:
                            return StreamingResponse(
                                final_response.aiter_bytes(),
                                media_type="text/event-stream"
                            )
                        else:
                            final_result = final_response.json()
                            print("最终响应:", json.dumps(final_result, ensure_ascii=False, indent=2))
                            return final_result
                    else:
                        raise HTTPException(
                            status_code=final_response.status_code,
                            detail=f"LM Studio final response error: {final_response.text}"
                        )
            except Exception as e:
                print(f"处理工具调用时发生错误: {str(e)}")
                print(f"错误堆栈: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # 修改这部分：确保在没有工具调用时也能正确返回响应
        if not (has_tool_calls or has_function_calling):
            if stream:
                # 重新发送流式请求
                stream_response = await client.post(
                    url,
                    json={**modified_body, "stream": True},
                    timeout=60.0
                )
                return StreamingResponse(
                    stream_response.aiter_bytes(),
                    media_type="text/event-stream"
                )
            return response_data
        
    except Exception as e:
        print(f"发生异常: {str(e)}")
        print(f"异常堆栈: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await client.aclose()


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

async def handle_tool_calls(response_data, original_request):
    print("开始处理工具调用...")
    if "choices" not in response_data:
        return response_data

    choice = response_data["choices"][0]
    if not choice.get("message", {}).get("tool_calls"):
        return response_data

    # 处理每个工具调用
    for tool_call in choice["message"]["tool_calls"]:
        if tool_call["function"]["name"] == "google_search":
            try:
                print("执行 Google 搜索工具调用...")
                args = json.loads(tool_call["function"]["arguments"])
                search_results = await google_search(args["query"])
                print(f"搜索结果: {search_results}")
                
                # 构建新的请求
                new_messages = original_request["messages"].copy()
                # 添加助手的工具调用消息
                new_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                # 添加工具的响应消息
                new_messages.append({
                    "role": "tool",
                    "content": search_results,
                    "tool_call_id": tool_call["id"],
                    "name": "google_search"
                })
                
                # 创建新的请求体，移除工具定义以避免循环调用
                new_request = {
                    "model": original_request.get("model", "gpt-3.5-turbo"),
                    "messages": new_messages,
                    "temperature": original_request.get("temperature", 0.7)
                }
                
                print("发送最终请求到 LM Studio...")
                # 发送最终请求
                async with httpx.AsyncClient() as client:
                    final_response = await client.post(
                        f"{OPENAI_BASE_URL}/v1/chat/completions",
                        json=new_request,
                        timeout=60.0
                    )
                    
                    if final_response.status_code == 200:
                        result = final_response.json()
                        print(f"LM Studio 最终响应: {json.dumps(result, ensure_ascii=False)}")
                        return result
                    else:
                        print(f"LM Studio 响应错误: {final_response.status_code}")
                        return {"error": f"LM Studio error: {final_response.text}"}
                        
            except Exception as e:
                print(f"工具调用处理错误: {str(e)}")
                print(f"错误堆栈: {traceback.format_exc()}")
                return {"error": f"Tool call error: {str(e)}"}

    return response_data

async def fetch_and_parse_url(url, client):
    """
    抓取并解析单个URL的内容
    """
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
    
    async with httpx.AsyncClient(verify=False, follow_redirects=True) as client:
        tasks = []
        for result in results:
            url = result.get('link')
            if url and is_valid_url(url):
                tasks.append(fetch_and_parse_url(url, client))
        
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
                    for i, item in enumerate(data['results'][:-1], 1):
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