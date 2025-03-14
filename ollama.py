import os
import requests
import time
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import re

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 요청 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

load_dotenv() 

# 사용할 고정 모델
MODEL_NAME = os.getenv("MODEL_NAME")

# 벡터DB 저장 위치
VECTORDB_PATH = os.getenv("VECTORDB_URL")

# 데이터 파일이 저장된 폴더 (회의록 txt 저장 위치)
DATA_DIR = "./example"  # 실제 경로에 맞게 변경

# 순환할 파일 리스트
FILE_LIST = ["1", "2", "3"]

# 순서를 기억할 카운터
file_counter = 0  # 0부터 시작

class QueryRequest(BaseModel):
    script: str  # 스크립트 파라미터

OLLAMA_IP = os.getenv("OLLAMA_IP")

def query_ollama(prompt, script=""):
    url = f"{OLLAMA_IP}/api/generate"  # Ollama API 엔드포인트 (로컬 서버)
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

# 벡터DB 생성 함수
def load_documents_and_create_vectorstore():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), encoding="utf-8") as f:
                file_text = f.read()
                chunks = text_splitter.split_text(file_text)
                for chunk in chunks:
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"filename": filename}
                    ))

    vectorstore = FAISS.from_documents(documents, embedding=FastEmbedEmbeddings())
    vectorstore.save_local(VECTORDB_PATH)
    return vectorstore

# 벡터DB 불러오기
def get_vectorstore():
    if os.path.exists(VECTORDB_PATH):
        return FAISS.load_local(VECTORDB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return load_documents_and_create_vectorstore()

# 순서대로 파일명 반환 함수 (순환 방식)
def get_next_filename():
    global file_counter
    filename = FILE_LIST[file_counter]
    file_counter += 1
    if file_counter >= len(FILE_LIST):
        file_counter = 0
    return filename

@app.post("/api/bot/endmeeting")
async def end_meeting_summary(query: QueryRequest):
    prompt = f"""
    주어진 회의 내용을 참고하여 아래 형식으로 회의록을 생성해줘. 
    제공된 스크립트 내용을 반드시 반영해서 회의록을 작성해야 해.

    ---
    ### 회의록: 자동 생성된 제목

    #### 목차
    1. 개요
    2. 주요 논의 사항
        1. 첫 번째 논의 내용
        2. 두 번째 논의 내용
        3. …
    3. 다음 단계

    #### 개요
    - 회의에서 논의된 주요 내용을 요약해서 작성해줘.

    ### 1. 주요 논의 사항
    - **주요 토픽 1**: 논의된 내용을 정리
    - **주요 토픽 2**: 논의된 내용을 정리
    - **추가 논의 사항**: 중요하게 언급된 내용이 있다면 포함

    ### 2. 다음 단계
    - 회의에서 결정된 액션 플랜을 정리

    ---
    
    **참고 회의 스크립트:**  
    \"\"\"  
    {query.script}  
    \"\"\"  

    회의 내용을 바탕으로 위 형식에 맞춰 회의록을 작성해줘.
    """

    result = query_ollama(prompt, query.script)  # LLM 요청
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')

    return JSONResponse(content={"response": response_text})



@app.post("/api/bot/positive")
async def positive_response(query: QueryRequest):
    prompt = "긍정적이고 격려하는 태도로 응답해줘. 꼭 한 줄!로 간결하게 대답해 스크립트 내용은 말하지마: \n\n"
    result = query_ollama(prompt, query.script)  
    print(f"{result}")
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    print(f"{response_text}")
    return JSONResponse(content={"response": response_text})


@app.post("/api/bot/negative")
async def negative_response(query: QueryRequest):
    prompt = "현실감 있는 리액션을 해줘. 응원을 하면서도 잘못된 내용이나 객관적인 부가 사실을 더 알려줘. 꼭 한 줄!로 간결하게 대답해 스크립트 내용은 말하지마:\n\n"
    result = query_ollama(prompt, query.script)  
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})

@app.post("/api/bot/summary")
async def summary_response(query: QueryRequest):
    prompt = "스크립트를 보고 요약해줘. 참여자가 더 잘 회의를 이끌어갈 수 있도록 응원하는 말을 해줘. 꼭 한 줄!로 간결하게 대답해 스크립트 내용은 말하지마:\n\n"
    result = query_ollama(prompt, query.script)  
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})

@app.post("/api/bot/loader")
async def loader_response(query: QueryRequest):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query.script)

    # 유사도 검색 결과 기반 문서 내용 합치기
    context = "\n\n".join([doc.page_content for doc in docs])

    # 순환 방식으로 파일명 선택 (강제 note_id 부여)
    note_id = get_next_filename()

    prompt = f"""
    다음은 회의록에서 찾은 관련 내용입니다:{context}
    참고 문서: {note_id}
    사용자 질문: {query.script}
    위 참고 문서를 기반으로 사용자 질문에 대해 정확하고 구체적으로 답변해줘. 또한, 짧고 질문에 대한 핵심적인 내용만 답변해줘.
    """

    result = query_ollama(prompt)
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')

    return JSONResponse(content={
        "note_ids": [note_id],  # 강제로 순환 방식으로 note_id 반환
        "response": response_text
    })


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="debug")