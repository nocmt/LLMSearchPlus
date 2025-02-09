import json
from fastapi.responses import StreamingResponse, Response
import httpx
from fastapi import FastAPI, HTTPException, Request
import requests
import uvicorn
import traceback
import os
from dotenv import load_dotenv
load_dotenv() # 加载环境变量

app = FastAPI()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
GOOGLE_CX = os.getenv("GOOGLE_CX")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


@app.api_route("/v1/models", methods=["GET","OPTIONS"])
async def get_models(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            # 转发原始请求头
            headers = dict(request.headers)
            headers.pop("host", None)  # 移除 host 头，避免冲突
            
            response = await client.get(
                f"{OPENAI_BASE_URL}/models",
                headers=headers,
                timeout=30.0
            )
            
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers)
            )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="LM Studio Studio Studio Studio API timeout")
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
                f"{OPENAI_BASE_URL}/completions",
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
        raise HTTPException(status_code=504, detail="LM Studio Studio Studio Studio API timeout")
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
                f"{OPENAI_BASE_URL}/embeddings",
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
        raise HTTPException(status_code=504, detail="LM Studio Studio Studio Studio API timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    client = httpx.AsyncClient()
    url = f"{OPENAI_BASE_URL}/chat/completions"
    
    try:
        body = await request.json()
        modified_body = inject_tool_definitions(body)
        stream = modified_body.get("stream", False)
        
        print("发送到 LM Studio Studio Studio Studio 的请求体:", json.dumps(modified_body, ensure_ascii=False, indent=2))
        
        # 如果是流式请求，先关闭它以处理工具调用
        if stream:
            modified_body["stream"] = False
        
        response = await client.post(url, json=modified_body, timeout=60.0)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"LM Studio Studio Studio API error: {response.text}"
            )
        
        response_data = response.json()
        print("LM Studio Studio Studio 原始返回数据:", json.dumps(response_data, ensure_ascii=False, indent=2))
        
        # 检查是否包含工具调用（标准格式）
        has_tool_calls = (
            "choices" in response_data 
            and response_data["choices"] 
            and "message" in response_data["choices"][0] 
            and "tool_calls" in response_data["choices"][0]["message"]
        )
        
        # 检查是否包含函数调用标记（特殊格式）
        has_function_calling = (
            "choices" in response_data 
            and response_data["choices"] 
            and "message" in response_data["choices"][0] 
            and "content" in response_data["choices"][0]["message"] 
            and "<function_calling>" in response_data["choices"][0]["message"]["content"]
        )
        
        if has_tool_calls or has_function_calling:
            try:
                # 提取查询参数
                query = None
                tool_call_id = None
                
                if has_tool_calls:
                    print("检测到标准工具调用格式")
                    tool_call = response_data["choices"][0]["message"]["tool_calls"][0]
                    tool_call_id = tool_call["id"]
                    args = json.loads(tool_call["function"]["arguments"])
                    query = args["query"]
                elif has_function_calling:
                    print("检测到特殊函数调用格式")
                    content = response_data["choices"][0]["message"]["content"]
                    json_str = content.split("<function_calling>")[1].strip()
                    function_data = json.loads(json_str)
                    query = function_data["params"]["query"]
                    tool_call_id = f"call_{str(hash(content))[:8]}"
                
                if query:
                    search_results = await google_search(query)
                    print(f"搜索结果: {search_results}")
                    
                    # 构建新的消息列表
                    new_messages = modified_body["messages"].copy()
                    tool_call = {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "google_search",
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
                        "name": "google_search"
                    })
                    
                    # 构建新的请求
                    new_request = {
                        "model": modified_body.get("model", "gpt-3.5-turbo"),
                        "messages": new_messages,
                        "stream": stream  # 恢复原始的流式设置
                    }
                    
                    print("发送最终请求到 LM Studio Studio Studio...")
                    final_response = await client.post(
                        f"{OPENAI_BASE_URL}/chat/completions",
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
                            detail=f"LM Studio Studio Studio final response error: {final_response.text}"
                        )
            except Exception as e:
                print(f"处理工具调用时发生错误: {str(e)}")
                print(f"错误堆栈: {traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # 如果不是工具调用，且是流式请求，返回流式响应
        if stream:
            return StreamingResponse(
                response.aiter_bytes(),
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
    tools = [{
        "type": "function",
        "function": {
            "name": "google_search",
            "description": "当需要获取实时信息，如当前日期、天气、新闻、产品信息等时使用谷歌搜索",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }]
    
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
                
                print("发送最终请求到 LM Studio Studio Studio...")
                # 发送最终请求
                async with httpx.AsyncClient() as client:
                    final_response = await client.post(
                        f"{OPENAI_BASE_URL}/chat/completions",
                        json=new_request,
                        timeout=60.0
                    )
                    
                    if final_response.status_code == 200:
                        result = final_response.json()
                        print(f"LM Studio Studio Studio 最终响应: {json.dumps(result, ensure_ascii=False)}")
                        return result
                    else:
                        print(f"LM Studio Studio Studio 响应错误: {final_response.status_code}")
                        return {"error": f"LM Studio Studio Studio error: {final_response.text}"}
                        
            except Exception as e:
                print(f"工具调用处理错误: {str(e)}")
                print(f"错误堆栈: {traceback.format_exc()}")
                return {"error": f"Tool call error: {str(e)}"}

    return response_data

async def google_search(query, num=3):
    print(f"开始执行 Google 搜索，查询词：{query}")
    try:
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&q={query}&cx={GOOGLE_CX}&num={num}"
        print(f"请求 URL（已隐藏敏感信息）: https://www.googleapis.com/customsearch/v1?key=***&q={query}&cx=***&num={num}")
        
        response = requests.get(url)
        print(f"API 响应状态码: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"API 返回数据类型: {type(data)}")
            print(f"API 返回数据键值: {list(data.keys())}")
            
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
            else:
                print("警告：API 响应中没有找到 'items' 键")
            
            final_result = json.dumps(results, ensure_ascii=False)
            print(f"最终返回结果长度: {len(final_result)} 字符")
            return final_result
        else:
            error_msg = f"搜索请求失败，状态码: {response.status_code}"
            print(error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"搜索过程发生异常：{str(e)}"
        print(f"错误：{error_msg}")
        print(f"异常类型：{type(e)}")
        print(f"异常详情：{traceback.format_exc()}")
        return error_msg
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8100, reload=True)