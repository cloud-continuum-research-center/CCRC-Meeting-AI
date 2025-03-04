import requests
import time
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import json
import re

app = FastAPI()

# 사용할 고정 모델
MODEL_NAME = "llama3.2-korea"

class QueryRequest(BaseModel):
    script: str  # 스크립트 파라미터


def query_ollama(prompt, script=""):
    url = "http://127.0.0.1:11434/api/generate"  # Ollama API 엔드포인트 (로컬 서버)
    headers = {
        "Content-Type": "application/json"
    }
    # 프롬프트와 스크립트를 합침
    full_prompt = f"{prompt} {script}"

    data = {
        "model": MODEL_NAME,  # 사용할 모델 이름
        "prompt": full_prompt,  # 프롬프트와 스크립트 결합
        "stream": False       # 스트리밍 여부
    }

    # 요청 시작 시간 기록
    start_time = time.time()

    try:
        # POST 요청 보내기
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 요청에 실패하면 예외가 발생

        # 응답 JSON 데이터 처리
        response_data = response.json()

        # 요청 소요 시간 출력
        print(f"Request time: {time.time() - start_time:.2f} seconds")
        
        return response_data
    
    except requests.exceptions.HTTPError as e:
        # HTTP 오류 처리
        return {"error": f"HTTP Error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.RequestException as e:
        # 기타 요청 오류 처리
        return {"error": f"Request Error: {str(e)}"}
    except ValueError as e:
        # JSON 파싱 오류 처리
        return {"error": f"Error decoding JSON: {str(e)}"}


@app.post("/api/positive")
async def positive_response(query: QueryRequest):
    prompt = "긍정적이고 격려하는 태도로 응답해줘. 꼭 한 줄!로 간결하게 대답해 스크립트 내용은 말하지마: \n\n"
    result = query_ollama(prompt, query.script)  
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})


@app.post("/api/negative")
async def negative_response(query: QueryRequest):
    prompt = "현실감 있는 리액션을 해줘. 응원을 하면서도 잘못된 내용이나 객관적인 부가 사실을 더 알려줘. 꼭 한 줄!로 간결하게 대답해 스크립트 내용은 말하지마:\n\n"
    result = query_ollama(prompt, query.script)  
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})

@app.post("/api/summary")
async def summary_response(query: QueryRequest):
    prompt = "스크립트를 보고 요약해줘. 참여자가 더 잘 회의를 이끌어갈 수 있도록 응원하는 말을 해줘. 꼭 한 줄!로 간결하게 대답해 스크립트 내용은 말하지마:\n\n"
    result = query_ollama(prompt, query.script)  
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)