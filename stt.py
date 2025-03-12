from fastapi import FastAPI, UploadFile, File, Form, Depends
import shutil
import whisper
import os
import requests
import time
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base
from mysql import get_db

app = FastAPI()

load_dotenv() 

# STT 모델 로드
model = whisper.load_model("turbo")

# 디렉토리 설정
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

Base = declarative_base()

# Bot 테이블 모델 정의
class Bot(Base):
    __tablename__ = "bots"
    bot_id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(Integer, nullable=False)
    type = Column(String, nullable=True)  # type 컬럼 추가
    content = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(6), server_default=func.now())

LLM_IP = os.getenv("LLM_IP")

# LLM 서버 API URL
LLM_API_URLS = {
    "POSITIVE": f"{LLM_IP}/api/bot/positive",
    "NEGATIVE": f"{LLM_IP}/api/bot/negative",
    "SUMMARY": f"{LLM_IP}/api/bot/summary",
    "LOADER": f"{LLM_IP}/api/bot/loader",
}

# LLM 서버에 STT 결과 전달하는 함수
def send_to_llm(llm_url, text):
    payload = {"script": text}
    headers = {"Content-Type": "application/json"}
    response = requests.post(llm_url, json=payload, headers=headers)
    return response.json().get("response", "응답을 가져올 수 없습니다.")

@app.post("/api/positive")
async def transcribe_positive(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(INPUT_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="POSITIVE",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    result = model.transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
    
    text = result["text"]
    
    llm_response = send_to_llm(LLM_API_URLS["POSITIVE"], text)
    
    new_bot_entry.content = llm_response
    db.commit()
    
    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/negative")
async def transcribe_negative(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(INPUT_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="NEGATIVE",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    result = model.transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
    
    text = result["text"]
    
    llm_response = send_to_llm(LLM_API_URLS["NEGATIVE"], text)
    
    new_bot_entry.content = llm_response
    db.commit()
    
    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/summary")
async def transcribe_summary(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(INPUT_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="SUMMARY",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    result = model.transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
    
    text = result["text"]
    
    llm_response = send_to_llm(LLM_API_URLS["SUMMARY"], text)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/loader")
async def transcribe_loader(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(INPUT_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="LOADER",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    result = model.transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
    
    text = result["text"]
    
    llm_response = send_to_llm(LLM_API_URLS["LOADER"], text)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}
