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
import shutil

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
MBTI_VECTOR_DB_PATH = os.getenv("MBTI_VECTOR_DB_PATH")

class LoaderRequest(BaseModel):
    stt_text: str
    scripts: list

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
    }import os
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
import shutil

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
MBTI_DATA_PATH = "./mbti_data.txt"

# 순환할 파일 리스트
FILE_LIST = ["1", "2", "3"]

# 순서를 기억할 카운터
file_counter = 0  # 0부터 시작

class QueryRequest(BaseModel):
    script: str  # 스크립트 파라미터

OLLAMA_IP = os.getenv("OLLAMA_IP")
MBTI_VECTOR_DB_PATH = os.getenv("MBTI_VECTOR_DB_PATH")

class LoaderRequest(BaseModel):
    stt_text: str
    scripts: list

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
    
# 벡터DB를 강제로 초기화하고 scripts만 저장
def update_vectorstore(scripts):
    # 벡터DB가 존재하면 로드, 없으면 새로 생성
    if os.path.exists(VECTORDB_PATH):
        print("[FAISS] 기존 벡터DB 로드 중...")
        vectorstore = FAISS.load_local(VECTORDB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("[FAISS] 새로운 벡터DB 생성 중...")
        vectorstore = None

    # 새로 들어온 scripts를 벡터화
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    new_documents = [Document(page_content=chunk) for script in scripts for chunk in text_splitter.split_text(script)]

    # 벡터DB가 있으면 기존 데이터에 추가, 없으면 새로 생성
    if vectorstore:
        print("[FAISS] 기존 벡터DB에 새 문서 추가 중...")
        vectorstore.add_documents(new_documents)  # 기존 벡터DB에 새로운 문서 추가
    else:
        print("[FAISS] 새로운 벡터DB 생성 중...")
        vectorstore = FAISS.from_documents(new_documents, embedding=FastEmbedEmbeddings())

    # 변경된 벡터DB 저장
    vectorstore.save_local(VECTORDB_PATH)
    print("[FAISS] 벡터DB 업데이트 완료!")

    return vectorstore

# 벡터DB 불러오기 (이제 scripts만 처리)
def get_vectorstore(scripts):
    if os.path.exists(VECTORDB_PATH):
        return FAISS.load_local(VECTORDB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return update_vectorstore(scripts)
    
# Ollama에게 질의하는 함수 (stt_text + similar_context 포함)
def query_ollama_loader(stt_text, similar_context):
    url = f"{OLLAMA_IP}/api/generate"
    headers = {"Content-Type": "application/json"}

    # 프롬프트 수정: 이전 회의 내용을 참고하여 한 줄 피드백 제공
    prompt = f"""
    현재 회의 내용:
    "{stt_text}"

    이전 회의에서 유사한 논의가 있었습니다:
    "{similar_context}"

    이전 회의 내용을 참고하여 현재 회의에 대한 짧고 명확한 피드백을 제공해주세요.  
    반드시 한 문장으로 요약하며, 핵심적인 인사이트만 전달하세요.

    예시 출력:
    "이전 회의에서는 X에 대한 논의가 있었으니, Y 방향으로 진행하면 더욱 효과적일 것 같습니다."
    """

    data = {"prompt": prompt, "model": MODEL_NAME, "stream": False}

    try:
        print(f"[OLLAMA] Sending to LLM: {data}")  # 요청 로그 추가
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        print(f"[OLLAMA] LLM Raw Response: {response_data}")  # 응답 로그 추가

        return response_data.get("response", "응답을 가져올 수 없습니다.")

    except requests.exceptions.RequestException as e:
        print(f"[OLLAMA] LLM Request Error: {e}")
        return {"error": str(e)}



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
async def loader_response(query: LoaderRequest):
    print("[API] /api/bot/loader 호출됨")
    start_time = time.time()

    # 벡터DB 생성 (기존 데이터 삭제 후 scripts만 벡터화)
    vectorstore = update_vectorstore(query.scripts)

    # STT 결과를 벡터DB에서 검색
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query.stt_text)

    # 유사한 회의 내용 추출
    similar_context = "\n\n".join([doc.page_content for doc in docs])

    # 디버깅 로그 추가
    print(f"[OLLAMA] Received stt_text={query.stt_text[:100]}")
    print(f"[OLLAMA] Found similar context: {similar_context[:300]}")  # 300자까지만 출력
    print(f"vectorDB 처리 시간: {time.time() - start_time:.2f}초")

    # LLM에게 질의 수행
    llm_response = query_ollama_loader(query.stt_text, similar_context)

    print(f"[OLLAMA] LLM Response: {llm_response[:300]}")  # 응답 로그 추가

    return JSONResponse(content={"response": llm_response})

@app.post("/api/bot/mbti")
async def mbti_response(query: QueryRequest):
    print("[API] /api/bot/mbti 호출됨")

    # 벡터DB 로딩 또는 생성
    if os.path.exists(MBTI_VECTOR_DB_PATH):
        vectorstore = FAISS.load_local(MBTI_VECTOR_DB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("[MBTI] 최초 벡터 DB 생성")
        loader = TextLoader(MBTI_DATA_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(split_docs, FastEmbedEmbeddings())
        vectorstore.save_local(MBTI_VECTOR_DB_PATH)

    # 유사 문서 검색
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query.script)
    similar_context = "\n\n".join([doc.page_content for doc in docs])

    # 프롬프트 생성
    prompt = f"""
    당신은 MBTI 성격 유형 분석 전문가입니다.

    다음은 사용자가 음성으로 응답한 STT 대화 내용입니다:

    \"\"\"{query.script}\"\"\"

    그리고 아래는 MBTI를 추론할 수 있는 질문-응답 규칙과 각 유형에 대한 설명입니다:

    {similar_context}

    이 내용을 바탕으로 사용자의 MBTI를 4글자로 추론하고,
    아래 예시처럼 한 문장으로만 대답해주세요.

    예시: 당신의 MBTI는 xxxx입니다! xxxx는 이런 특성을 가지고 있습니다!

    *주의*: 사용자의 답변(script)이나 규칙(context)은 답변에 포함하지 마세요.
    """

    result = query_ollama(prompt, query.script)
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="debug")

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
    
# 벡터DB를 강제로 초기화하고 scripts만 저장
def update_vectorstore(scripts):
    # 벡터DB가 존재하면 로드, 없으면 새로 생성
    if os.path.exists(VECTORDB_PATH):
        print("[FAISS] 기존 벡터DB 로드 중...")
        vectorstore = FAISS.load_local(VECTORDB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("[FAISS] 새로운 벡터DB 생성 중...")
        vectorstore = None

    # 새로 들어온 scripts를 벡터화
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    new_documents = [Document(page_content=chunk) for script in scripts for chunk in text_splitter.split_text(script)]

    # 벡터DB가 있으면 기존 데이터에 추가, 없으면 새로 생성
    if vectorstore:
        print("[FAISS] 기존 벡터DB에 새 문서 추가 중...")
        vectorstore.add_documents(new_documents)  # 기존 벡터DB에 새로운 문서 추가
    else:
        print("[FAISS] 새로운 벡터DB 생성 중...")
        vectorstore = FAISS.from_documents(new_documents, embedding=FastEmbedEmbeddings())

    # 변경된 벡터DB 저장
    vectorstore.save_local(VECTORDB_PATH)
    print("[FAISS] 벡터DB 업데이트 완료!")

    return vectorstore

# 벡터DB 불러오기 (이제 scripts만 처리)
def get_vectorstore(scripts):
    if os.path.exists(VECTORDB_PATH):
        return FAISS.load_local(VECTORDB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        return update_vectorstore(scripts)
    
# Ollama에게 질의하는 함수 (stt_text + similar_context 포함)
def query_ollama_loader(stt_text, similar_context):
    url = f"{OLLAMA_IP}/api/generate"
    headers = {"Content-Type": "application/json"}

    # 프롬프트 수정: 이전 회의 내용을 참고하여 한 줄 피드백 제공
    prompt = f"""
    현재 회의 내용:
    "{stt_text}"

    이전 회의에서 유사한 논의가 있었습니다:
    "{similar_context}"

    이전 회의 내용을 참고하여 현재 회의에 대한 짧고 명확한 피드백을 제공해주세요.  
    반드시 한 문장으로 요약하며, 핵심적인 인사이트만 전달하세요.

    예시 출력:
    "이전 회의에서는 X에 대한 논의가 있었으니, Y 방향으로 진행하면 더욱 효과적일 것 같습니다."
    """

    data = {"prompt": prompt, "model": MODEL_NAME, "stream": False}

    try:
        print(f"[OLLAMA] Sending to LLM: {data}")  # 요청 로그 추가
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        print(f"[OLLAMA] LLM Raw Response: {response_data}")  # 응답 로그 추가

        return response_data.get("response", "응답을 가져올 수 없습니다.")

    except requests.exceptions.RequestException as e:
        print(f"[OLLAMA] LLM Request Error: {e}")
        return {"error": str(e)}



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
async def loader_response(query: LoaderRequest):
    print("[API] /api/bot/loader 호출됨")
    start_time = time.time()

    # 벡터DB 생성 (기존 데이터 삭제 후 scripts만 벡터화)
    vectorstore = update_vectorstore(query.scripts)

    # STT 결과를 벡터DB에서 검색
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query.stt_text)

    # 유사한 회의 내용 추출
    similar_context = "\n\n".join([doc.page_content for doc in docs])

    # 디버깅 로그 추가
    print(f"[OLLAMA] Received stt_text={query.stt_text[:100]}")
    print(f"[OLLAMA] Found similar context: {similar_context[:300]}")  # 300자까지만 출력
    print(f"vectorDB 처리 시간: {time.time() - start_time:.2f}초")

    # LLM에게 질의 수행
    llm_response = query_ollama_loader(query.stt_text, similar_context)

    print(f"[OLLAMA] LLM Response: {llm_response[:300]}")  # 응답 로그 추가

    return JSONResponse(content={"response": llm_response})

@app.post("/api/bot/mbti")
async def mbti_response(query: QueryRequest):
    print("[API] /api/bot/mbti 호출됨")

    # 벡터DB 로딩 또는 생성
    if os.path.exists(MBTI_VECTOR_DB_PATH):
        vectorstore = FAISS.load_local(MBTI_VECTOR_DB_PATH, FastEmbedEmbeddings(), allow_dangerous_deserialization=True)
    else:
        print("[MBTI] 최초 벡터 DB 생성")
        loader = TextLoader(MBTI_DATA_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        vectorstore = FAISS.from_documents(split_docs, FastEmbedEmbeddings())
        vectorstore.save_local(MBTI_VECTOR_DB_PATH)

    # 유사 문서 검색
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(query.script)
    similar_context = "\n\n".join([doc.page_content for doc in docs])

    # 프롬프트 생성
    prompt = f"""
    당신은 MBTI 성격 유형 분석 전문가입니다.

    다음은 사용자가 음성으로 응답한 STT 대화 내용입니다:

    \"\"\"{query.script}\"\"\"

    그리고 아래는 MBTI를 추론할 수 있는 질문-응답 규칙과 각 유형에 대한 설명입니다:

    {similar_context}

    이 내용을 바탕으로 사용자의 MBTI를 4글자로 추론하고,
    아래 예시처럼 한 문장으로만 대답해주세요.

    예시: 당신의 MBTI는 xxxx입니다! xxxx는 이런 특성을 가지고 있습니다!

    *주의*: 사용자의 답변(script)이나 규칙(context)은 답변에 포함하지 마세요.
    """

    result = query_ollama(prompt, query.script)
    response_text = result.get("response", "응답을 가져올 수 없습니다.").strip('"')
    return JSONResponse(content={"response": response_text})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="debug")
