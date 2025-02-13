import os
import json
import streamlit as st
import numpy as np
import re
from datetime import datetime
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

# ✅ OpenAI GPT 모델 초기화
chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# ✅ Streamlit UI 설정 (여백을 최소화)
st.set_page_config(page_title="전주대학교 비교과 챗봇 💬", page_icon="🤖", layout="wide")

# ✅ 스타일 적용 (말풍선 UI + 여백 최소화)
st.markdown("""
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .chat-container { display: flex; flex-direction: column; gap: 5px; padding: 5px; margin-top: -10px; }
        .chat-message { 
            padding: 10px; border-radius: 15px; max-width: 70%;
            animation: fadeIn 0.3s ease-in-out;
            margin-bottom: 3px;
        }
        .user { background-color: #4CAF50; color: white; align-self: flex-end; text-align: right; }
        .assistant { background-color: #E0E0E0; color: black; align-self: flex-start; text-align: left; }
        .stTextInput input { border-radius: 8px; padding: 8px; height: 40px; }
        .send-btn { background-color: #007AFF; color: white; padding: 8px 12px; border-radius: 5px; border: none; cursor: pointer; font-size: 14px; }
    </style>
""", unsafe_allow_html=True)

# ✅ 세션 상태 초기화 (대화 내역 저장)
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="안녕하세요! 저는 전주대학교 비교과 챗봇입니다. 😊 궁금한 점을 물어보세요!")
    ]

# ✅ JSON 데이터 로드 함수 (파일 경로 직접 입력)
@st.cache_data
def load_program_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"❌ JSON 파일을 찾을 수 없습니다: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        possible_keys = ["프로그램_정보", "비교과_프로그램", "프로그램"]
        for key in possible_keys:
            if key in data:
                return data[key]

        st.error("❌ JSON 데이터에서 프로그램 정보를 찾을 수 없습니다.")
        return []
    except Exception as e:
        st.error(f"❌ JSON 로드 오류: {str(e)}")
        return []

# ✅ JSON 파일 경로 입력 받기
file_path = st.text_input("📂 JSON 파일 경로를 입력하세요:", r"C:\Users\user\Desktop\Github\prjRepo_JJU\project\programs.json")

if file_path:
    program_data = load_program_data(file_path)

# ✅ 채팅 UI (이전 대화 내역 표시)
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state["messages"]:
    role_class = "user" if isinstance(message, HumanMessage) else "assistant"
    st.markdown(f'<div class="chat-message {role_class}">{message.content}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ✅ 질문에서 키워드 및 필터링 조건 추출 (점프업 관련 키워드 확장)
def extract_filters(query):
    query_lower = query.lower()

    # 특정 월 필터 (ex: "2월", "3월")
    month_match = re.findall(r"(\d{4}\.\d{2})", query_lower)

    # 특정 키워드 감지 (점프업 포함 확장)
    keywords = ["점프업", "점프업 포인트", "점프업 자기주도형 포인트", "점프업 프로그램", "비교과 포인트", "ncs", "멘토링", "창업", "자격증", "특강", "취업"]
    matched_keywords = [kw for kw in keywords if kw in query_lower]

    # "점프업"이 포함되면 관련된 모든 항목 검색
    if "점프업" in query_lower:
        matched_keywords.extend(["점프업 포인트", "점프업 자기주도형 포인트", "점프업 프로그램"])

    return month_match, matched_keywords

# ✅ 비교과 프로그램 검색 함수
def find_program(query):
    month_match, matched_keywords = extract_filters(query)
    results = []

    for program in program_data:
        title = program.get("제목", "").lower()
        description = program.get("설명", "").lower()
        period = program.get("기간", "").lower()
        benefits = program.get("혜택", "").lower()

        # 특정 월 포함 필터링
        if month_match and not any(month in period for month in month_match):
            continue  

        # 특정 키워드 필터링 (점프업 관련 키워드 포함)
        if matched_keywords and not any(kw in description or kw in title or kw in benefits for kw in matched_keywords):
            continue

        results.append(program)

    return results

# ✅ RAG 기반 응답 생성 함수 (GPT가 JSON 데이터를 기반으로 답변 생성)
def generate_rag_response(query):
    results = find_program(query)

    if results:
        gpt_prompt = f"""
        사용자 질문: "{query}"
        검색된 비교과 프로그램 목록:
        {json.dumps(results, indent=2, ensure_ascii=False)}
        
        위 정보를 바탕으로 사용자가 이해하기 쉽게 설명해줘.
        """
        response = chat_model.invoke(gpt_prompt)
        return response.content
    else:
        return "⚠️ 관련된 비교과 프로그램을 찾을 수 없습니다. 다시 검색해 주세요!"

# ✅ 사용자 입력 (채팅 인터페이스)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 여기에 메시지를 입력하세요...", key="user_input")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    # ✅ 사용자 메시지를 세션 상태에 추가
    st.session_state["messages"].append(HumanMessage(content=user_input))

    # ✅ OpenAI 모델을 사용하여 응답 생성
    response_content = generate_rag_response(user_input)

    # ✅ AI 응답을 세션 상태에 추가
    st.session_state["messages"].append(AIMessage(content=response_content))

    # ✅ UI 업데이트 (새로운 메시지 즉시 적용)
    st.rerun()
