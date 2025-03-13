from fastapi import FastAPI, UploadFile, File, Form, Depends
import shutil
import whisper
import os
import requests
import time
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Text, func
from sqlalchemy.ext.declarative import declarative_base
from mysql import get_db
from datetime import datetime

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
    
# user_meetings 테이블 엔티티
class UserMeeting(Base):
    __tablename__ = "user_meetings"

    user_meeting_id = Column(Integer, primary_key=True, autoincrement=True)
    entry_time = Column(TIMESTAMP(6), nullable=False)
    exit_time = Column(TIMESTAMP(6), nullable=True)
    meeting_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    user_team_id = Column(Integer, nullable=False)
    
# note 테이블 엔티티
class Note(Base):
    __tablename__ = "note"

    note_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP(6), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(6), onupdate=func.now(), nullable=True)
    members = Column(String(255), nullable=True)
    audio_url = Column(String(1000), nullable=True)
    script = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    title = Column(String(255), nullable=True)
    meeting_id = Column(Integer, ForeignKey("user_meetings.meeting_id"), nullable=False)

# users 테이블 엔티티
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP(6), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(6), onupdate=func.now(), nullable=True)
    email = Column(String(255), unique=True, nullable=False)
    nickname = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)
    profile = Column(String(2048), nullable=True)


    

LLM_IP = os.getenv("LLM_IP")

# LLM 서버 API URL
LLM_API_URLS = {
    "POSITIVE": f"{LLM_IP}/api/bot/positive",
    "NEGATIVE": f"{LLM_IP}/api/bot/negative",
    "SUMMARY": f"{LLM_IP}/api/bot/summary",
    "LOADER": f"{LLM_IP}/api/bot/loader",
    "END": f"{LLM_IP}/api/bot/endmeeting",
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

@app.post("/api/v1/endmeeting")
async def end_meeting(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    # 📌 1️⃣ 파일 저장 (mp3)
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 📌 2️⃣ userMeetings 테이블에서 meeting_id에 해당하는 user_id 조회
    user_ids = db.query(UserMeeting.user_id).filter(UserMeeting.meeting_id == meeting_id).all()
    user_ids = [user_id[0] for user_id in user_ids]

    # 📌 3️⃣ users 테이블에서 nickname 조회하여 members 필드에 저장
    nicknames = db.query(User.nickname).filter(User.user_id.in_(user_ids)).all()
    members = ", ".join([nickname[0] for nickname in nicknames])

    print(f"📌 members 값: {members}")

    # 📌 4️⃣ STT 변환 수행 (script 저장)
    result = model.transcribe(file_path)
    text = result["text"]

    print(f"📌 STT 변환 완료. script 내용: {text[:50]}...")  # 일부만 출력

    # 📌 5️⃣ Ollama로 요약 요청 (summary 저장)
    llm_response = send_to_llm(LLM_API_URLS["END"], text)

    print(f"📌 Ollama 요약 완료. summary 내용: {llm_response[:50]}...")  # 일부만 출력

    # 📌 6️⃣ 현재 날짜 (YYYY-MM-DD 형식)
    current_date = datetime.now().strftime("%Y-%m-%d")
    print(f"📌 저장할 title 값: {current_date}")

    # 📌 7️⃣ note 테이블에 데이터 추가
    new_note = Note(
        created_at=func.now(),
        updated_at=func.now(),
        members=members,
        audio_url=file_path,   # MP3 파일 저장 경로
        script=text,    # 원본 STT 결과
        summary=llm_response,  # 요약 결과
        title=current_date,    # YYYY-MM-DD 형식 날짜
        meeting_id=meeting_id  # 프론트에서 받은 meeting_id 그대로 저장
    )

    db.add(new_note)
    db.commit()

    print("📌 DB 저장 완료!")
    return  # 🚀 응답 필요 없음 (DB 저장만 수행)
