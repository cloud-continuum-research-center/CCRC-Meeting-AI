import os
import openai

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# API_KEY 설정
API_KEY = os.getenv("API_KEY")
openai.api_key = API_KEY

# 사용자 질문 받기
query = input("질문을 입력하세요: ")

# 언어모델(LLM) 생성
model = ChatOpenAI(
    temperature=0.2,
    max_tokens=2048,
    model_name='gpt-4o-mini',
)

# 사용자 선택 받기
print("\n[모드 선택]")
print("0: 기본 모드 (단순 질의응답)")
print("1: 긍정적인 답변 (positive)")
print("2: 현실적인 답변 (realistic)")
print("3: 요약 제공 (summary)")
print("4: RAG 기반 답변 (rag)")

choice_map = {"0": None, "1": "positive", "2": "realistic", "3": "summary", "4": "rag"}
choice = input("원하는 답변 모드를 선택하세요 (0~4): ")
choice = choice_map.get(choice, None)  # 기본값은 단순 질의응답

# RAG 모드 선택 시만 문서 로드 및 임베딩 수행
if choice == "rag":
    loader = TextLoader("./rag/icns.txt")
    docs = loader.load()
    with open("./rag/icns.txt", "r") as f:
        text = f.read()
    
    recursive_text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    splits = recursive_text_splitter.split_text(text)
    vectorstore = FAISS.from_texts(texts=splits, embedding=FastEmbedEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity")
    search_result = retriever.get_relevant_documents(query)
else:
    search_result = None

# 프롬프트 생성 함수
def generate_prompt(choice, question, docs=None):
    if choice == "positive":
        return f"""
        당신은 긍정적인 답변을 하는 AI입니다. 
        사용자 질문: {question}
        긍정적인 답변을 제공하세요.
        """
    elif choice == "realistic":
        return f"""
        당신은 현실적인 답변을 제공하는 AI입니다. 
        사용자 질문: {question}
        잘못된 내용이나 객관적인 부가 사실을 현실적 답변으로 제공하세요.
        """
    elif choice == "summary":
        return f"""
        당신은 요약을 제공하는 AI입니다. 
        사용자 질문: {question}
        간결한 요약을 제공하세요.
        """
    elif choice == "rag":
        context = "\n".join([doc.page_content for doc in docs]) if docs else ""
        return f"""
        당신은 RAG(검색 증강 생성) 기반으로 답변하는 AI입니다. 
        사용자 질문: {question}
        참고할 내용: {context}
        검색 내용을 기반으로 정확한 답변을 제공하세요.
        """
    else:
        return question  # 기본 모드는 프롬프트 없이 질문 그대로 전달

# 최종 프롬프트 생성
final_prompt = generate_prompt(choice, query, search_result)

# 체인 실행
chain = load_qa_chain(model, chain_type="stuff", verbose=True)
answer = chain.run(input_documents=search_result if search_result else [], question=final_prompt)

print("\n[질문에 대한 답변]")
print(answer)
