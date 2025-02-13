import os
import json
import streamlit as st
import numpy as np
import re
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 환경 변수에서 API 키 가져오기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ✅ OpenAI GPT 모델 초기화
chat_model = ChatOpenAI(model_name="gpt-4o", temperature=0.1)

# ✅ Streamlit UI 설정
st.set_page_config(page_title="전주대학교 비교과 챗봇", page_icon="🎓", layout="centered")
st.title("🎓 전주대학교 비교과 챗봇")
st.write("전주대학교 비교과 프로그램에 대해 질문하세요!")

# ✅ 세션 상태 초기화 (KeyError 방지)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ JSON 데이터 로드 함수
@st.cache_data
def load_program_data():
    file_path = r"C:\Users\user\Desktop\Github\prjRepo_JJU\project\programs.json"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        possible_keys = ["프로그램_정보", "비교과_프로그램", "프로그램"]
        for key in possible_keys:
            if key in data:
                return data[key]

        return []
    except Exception as e:
        st.error(f"❌ JSON 로드 오류: {str(e)}")
        return []

program_data = load_program_data()

# ✅ OpenAI 임베딩 모델 사용
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ✅ JSON 데이터를 벡터화하여 저장 (임베딩으로 저장)
@st.cache_data
def create_embeddings(data):
    if not data:
        return np.array([])
    texts = [f"{item['제목']} {item['설명']} {item.get('신청대상', '')} {item.get('혜택', '')}" for item in data]
    embeddings = embeddings_model.embed_documents(texts)
    return np.array(embeddings)

program_embeddings = create_embeddings(program_data)

# ✅ 키워드 및 필터링 조건 추출
def extract_filters(query):
    query_lower = query.lower()

    # 특정 월 필터 (ex: "2월", "3월")
    month_match = re.findall(r"(\d{4}\.\d{2})", query_lower)

    # 특정 기간 필터 (ex: "2025.02.10 ~ 2025.02.20")
    date_range_match = re.search(r"(\d{4}\.\d{2}\.\d{2})\s*~\s*(\d{4}\.\d{2}\.\d{2})", query_lower)

    # 특정 키워드 감지
    keywords = ["점프업", "점프업 포인트", "점프업 자기주도형 포인트", "점프업 프로그램", "비교과 포인트", "ncs", "멘토링", "창업", "자격증", "특강", "취업"]
    matched_keywords = [kw for kw in keywords if kw in query_lower]

    # "점프업"이 포함되면 관련된 모든 항목을 검색하도록 설정
    if "점프업" in query_lower:
        matched_keywords.extend(["점프업 포인트", "점프업 자기주도형 포인트", "점프업 프로그램"])

    # 특정 대상 필터 (ex: "3학년", "1학년", "졸업 예정자")
    target_match = re.search(r"(\d학년|졸업 예정자)", query_lower)
    target_filter = target_match.group(1) if target_match else None

    return month_match, date_range_match, matched_keywords, target_filter

# ✅ 비교과 프로그램 검색 함수
def find_program(query):
    month_match, date_range_match, matched_keywords, target_filter = extract_filters(query)
    results = []

    for program in program_data:
        title = program.get("제목", "").lower()
        description = program.get("설명", "").lower()
        period = program.get("기간", "").lower()
        benefits = program.get("혜택", "").lower()
        target = program.get("신청대상", "").lower()

        # 특정 월 포함 필터링
        if month_match and not any(month in period for month in month_match):
            continue  

        # 특정 키워드 필터링
        if matched_keywords and not any(kw in description or kw in title or kw in benefits for kw in matched_keywords):
            continue

        # 특정 대상(학년) 필터링
        if target_filter and target_filter not in target:
            continue

        results.append(program)

    return results

# ✅ RAG 기반 응답 생성 함수 (GPT가 JSON 데이터를 기반으로 응답 생성)
def generate_rag_response(query):
    results = find_program(query)

    if results:
        gpt_prompt = f"""
        사용자 질문: "{query}"
        검색된 비교과 프로그램 목록:
        {json.dumps(results, indent=2, ensure_ascii=False)}
        
        위 정보를 바탕으로 사용자에게 적절한 답변을 만들어줘.
        """
    else:
        gpt_prompt = f"""
        사용자 질문: "{query}"
        관련된 비교과 프로그램이 데이터에 명확히 없습니다. 하지만 유사한 정보를 제공할 수 있도록 최선을 다할게요.
        """

    response = chat_model.invoke(gpt_prompt)
    return response.content

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
