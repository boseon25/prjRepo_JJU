# RAG기반 Chatbot

import os
import json
import streamlit as st
import re  
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI  
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")

# ✅ OpenAI GPT 모델 초기화
chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# ✅ Streamlit UI 구성
st.set_page_config(page_title="전주대학교 비교과 챗봇", page_icon="🎓", layout="centered")
st.write("✅ Streamlit이 정상적으로 실행되고 있습니다!")

st.title("🎓 전주대학교 비교과 챗봇")
st.write("전주대학교 비교과 프로그램에 대해 질문하세요!")

# ✅ 세션 상태 강제 초기화 (에러 방지)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ JSON 데이터 로드 함수
@st.cache_data
def load_program_data():
    file_path = r"C:\Users\user\Desktop\Github\prjRepo_JJU\project\programs.json"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # JSON 키 자동 감지
        possible_keys = ["프로그램_정보", "비교과_프로그램", "프로그램"]
        for key in possible_keys:
            if key in data:
                return data[key]

        return []
    except Exception as e:
        st.error(f"❌ JSON 로드 오류: {str(e)}")
        return []

program_data = load_program_data()

# ✅ 질문에서 키워드 추출 함수
def extract_filters(query):
    query_lower = query.lower()

    # 특정 월 필터 (ex: "2월", "3월")
    month_match = re.search(r"(\d{1,2})월", query_lower)
    month_filter = month_match.group(1) if month_match else None

    # 특정 키워드 감지 (점프업, NCS 등)
    keywords = ["점프업 포인트", "비교과 포인트", "ncs", "멘토링", "창업", "자격증", "특강"]
    matched_keywords = [kw for kw in keywords if kw in query_lower]

    # 특정 대상 필터 (ex: "3학년", "1학년", "졸업 예정자")
    target_match = re.search(r"(\d학년|졸업 예정자)", query_lower)
    target_filter = target_match.group(1) if target_match else None

    return month_filter, matched_keywords, target_filter

# ✅ 비교과 프로그램 검색 함수
def find_program(query):
    month_filter, matched_keywords, target_filter = extract_filters(query)
    results = []

    for program in program_data:
        title = program.get("제목", "").lower()
        description = program.get("설명", "").lower()
        period = program.get("기간", "").lower()
        benefits = program.get("혜택", "").lower()
        target = program.get("신청대상", "").lower()

        # "기간" 필드에서 특정 월이 포함된지 확인
        if month_filter and not re.search(rf"{month_filter}\.", period):  
            continue  # 특정 월이 포함되지 않으면 필터링

        # 특정 키워드 필터링
        if matched_keywords and not any(kw in description or kw in title or kw in benefits for kw in matched_keywords):
            continue

        # 특정 대상(학년) 필터링
        if target_filter and target_filter not in target:
            continue

        results.append(program)

    return results

# ✅ RAG 기반 응답 생성 함수 (GPT가 JSON 데이터를 기반으로 답변 생성)
def generate_rag_response(query):
    results = find_program(query)  # JSON에서 관련 정보 검색

    if results:
        # ✅ GPT에게 검색된 데이터 기반으로 자연어 응답 생성 요청
        gpt_prompt = f"""
        사용자 질문: "{query}"
        검색된 비교과 프로그램 목록:
        {json.dumps(results, indent=2, ensure_ascii=False)}
        
        위 정보를 바탕으로 사용자가 이해하기 쉽게 설명해줘.
        """
        response = chat_model.invoke(gpt_prompt)
        return response.content  # GPT가 생성한 응답 반환
    else:
        return "⚠️ 관련된 비교과 프로그램을 찾을 수 없습니다. 다시 검색해 주세요!"

# ✅ 채팅 UI
chat_container = st.container()

# ✅ 대화 기록 표시
for msg in st.session_state["messages"]:
    with chat_container:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)

# ✅ 사용자 입력
user_input = st.chat_input("비교과 프로그램에 대해 질문하세요!")

if user_input:
    st.session_state["messages"].append(HumanMessage(content=user_input))

    # ✅ GPT를 활용하여 JSON 데이터 기반 응답 생성
    response_content = generate_rag_response(user_input)

    st.session_state["messages"].append(AIMessage(content=response_content))

    # ✅ 채팅 UI 즉시 갱신
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(response_content)
