from fastapi import FastAPI, UploadFile, File, Form, Depends
import shutil
import os
import requests
import time
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Text, func
from sqlalchemy.ext.declarative import declarative_base
from mysql import get_db
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import openai

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

# STT 모델 로드
openai.api_key = os.getenv("OPENAI_API_KEY")

# 디렉토리 설정
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

Base = declarative_base()

MBTI_VECTOR_DB_PATH = os.getenv("MBTI_VECTOR_DB_PATH")
MBTI_DATA_PATH = "./mbti_data.txt"



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
    team_id = Column(Integer, ForeignKey("meetings.team_id"), nullable=False) 

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

# meetings 테이블 엔티티 추가
class Meeting(Base):
    __tablename__ = "meetings"

    meeting_id = Column(Integer, primary_key=True, autoincrement=True)
    duration = Column(Integer, nullable=True)  # NULL 가능
    ended_at = Column(TIMESTAMP(6), nullable=True)
    started_at = Column(TIMESTAMP(6), nullable=False)
    title = Column(String(255), nullable=True)
    team_id = Column(Integer, nullable=False)

    

LLM_IP = os.getenv("LLM_IP")

# LLM 서버 API URL
LLM_API_URLS = {
    "POSITIVE": f"{LLM_IP}/api/bot/positive",
    "NEGATIVE": f"{LLM_IP}/api/bot/negative",
    "SUMMARY": f"{LLM_IP}/api/bot/summary",
    "LOADER": f"{LLM_IP}/api/bot/loader",
    "END": f"{LLM_IP}/api/bot/endmeeting",
    "MBTI": f"{LLM_IP}/api/bot/mbti",
    "SAJU": f"{LLM_IP}/api/bot/saju",
}

# LLM 서버에 STT 결과 전달하는 함수
def send_to_llm(llm_url, text, meeting_id):
    payload = {"script": text,
               "meeting_id": meeting_id}
    headers = {"Content-Type": "application/json"}
    response = requests.post(llm_url, json=payload, headers=headers)
    return response.json().get("response", "응답을 가져올 수 없습니다.")

# RAG(LOADER) 전용 LLM 서버 요청 함수 (stt_text + scripts 리스트 전송)
def send_to_llm_loader(llm_url, stt_text, scripts, meeting_id):
    payload = {"stt_text": stt_text, "scripts": scripts, "meeting_id": meeting_id}
    headers = {"Content-Type": "application/json"}
    response = requests.post(llm_url, json=payload, headers=headers)
    return response.json().get("response", "응답을 가져올 수 없습니다.")

def whisper_api_transcribe(file_path):
    with open(file_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="json"
        )
    return response["text"]

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
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
    
    llm_response = send_to_llm(LLM_API_URLS["POSITIVE"], text, meeting_id)
    
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
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
        
    llm_response = send_to_llm(LLM_API_URLS["NEGATIVE"], text, meeting_id)
    
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
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
        
    llm_response = send_to_llm(LLM_API_URLS["SUMMARY"], text, meeting_id)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/loader")
async def transcribe_loader(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    # 파일 저장
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # bot 테이블에 기록
    new_bot_entry = Bot(meeting_id=meeting_id, type="LOADER", content="분석중...", created_at=func.now())
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)

    # STT 변환 수행
    start_time = time.time()
    stt_text = whisper_api_transcribe(file_path)
    print(f"STT 처리 시간: {time.time() - start_time:.2f}초")

    # meetingId로 teamId 조회
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        return {"error": "Meeting not found"}

    team_id = meeting.team_id

    # teamId에 매칭되는 noteId 및 script 조회
    notes = db.query(Note).filter(Note.team_id == team_id).all()
    note_ids = [note.note_id for note in notes]
    scripts = [note.script for note in notes]

    # RAG 수행을 위한 새로운 send_to_llm_loader 사용
    print(f"[STT] Sending to LLM: stt_text={stt_text[:100]}, scripts={scripts[:2]}")  # 로그 추가
    llm_response = send_to_llm_loader(LLM_API_URLS["LOADER"], stt_text, scripts, meeting_id)

    # bot 테이블에 LLM 응답 저장
    new_bot_entry.content = llm_response
    db.commit()

    return {"note_ids": note_ids[0], "response": llm_response}

@app.post("/api/mbti")
async def transcribe_mbti(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="MBTI",
        content="MBTI 분석 중...",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)

    stt_text = whisper_api_transcribe(file_path)

    # MBTI 분석 요청
    llm_response = send_to_llm(LLM_API_URLS["MBTI"], stt_text, meeting_id)

    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": stt_text, "llm_response": llm_response}

@app.post("/api/saju")
async def transcribe_saju(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="SAJU",
        content="사주 분석중...",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)

    stt_text = whisper_api_transcribe(file_path)

    llm_response = send_to_llm(LLM_API_URLS["SAJU"], stt_text, meeting_id)

    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": stt_text, "llm_response": llm_response}

@app.post("/api/v1/endmeeting")
async def end_meeting(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    # 파일 저장 (MP3)
    file_path = os.path.join(INPUT_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # userMeetings 테이블에서 meeting_id에 해당하는 user_id 조회
    user_ids = db.query(UserMeeting.user_id).filter(UserMeeting.meeting_id == meeting_id).all()
    user_ids = [user_id[0] for user_id in user_ids]

    # users 테이블에서 nickname 조회하여 members 필드에 저장
    nicknames = db.query(User.nickname).filter(User.user_id.in_(user_ids)).all()
    members = ", ".join([nickname[0] for nickname in nicknames])

    # meetings 테이블에서 team_id 조회
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        return {"error": "Meeting not found"}

    team_id = meeting.team_id  # team_id 값 저장

    # STT 변환 수행 (script 저장)
    text = whisper_api_transcribe(file_path)

    # Ollama로 요약 요청 (summary 저장)
    llm_response = send_to_llm(LLM_API_URLS["END"], text, meeting_id)

    print(f"LLM 응답: {llm_response}...")  # 처음 100자만 출력하여 확인

    # 현재 날짜 (YYYY-MM-DD 형식)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # note 테이블에 데이터 추가 (team_id 포함)
    new_note = Note(
        created_at=func.now(),
        updated_at=func.now(),
        members=members,
        audio_url=file_path,
        script=text,
        summary=llm_response,
        title=current_date,
        meeting_id=meeting_id,
        team_id=team_id  # team_id 추가
    )
    db.add(new_note)

    # meetings 테이블 업데이트 (ended_at & duration 추가)
    ended_at = datetime.now()
    meeting.ended_at = ended_at

    if meeting.started_at:
        if ended_at < meeting.started_at:
            meeting.duration = 0  # 잘못된 경우 0 설정
        else:
            duration_seconds = (ended_at - meeting.started_at).total_seconds()
            duration_minutes = round(duration_seconds / 60)  # 분 단위 변환 (반올림)
            meeting.duration = duration_minutes
    else:
        meeting.duration = 0  # started_at이 없으면 0

    db.commit()

    return
