import os
import json
import streamlit as st
import re  # 정규 표현식 사용

# ✅ 필요한 라이브러리 추가
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# ✅ 환경 변수 설정
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["LANGCHAIN_API_KEY"] = "lsv2_..."
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "chatbot-test2"

# ✅ Streamlit UI 구성
st.set_page_config(page_title="전주대학교 비교과 챗봇", page_icon="🎓", layout="centered")
st.write("✅ Streamlit이 정상적으로 실행되고 있습니다!")

st.title("🎓 전주대학교 비교과 챗봇")
st.write("전주대학교 비교과 프로그램에 대해 질문하세요!")

# ✅ JSON 데이터 로드 함수
@st.cache_data
def load_program_data():
    file_path = r"C:\Users\user\Desktop\Github\prjRepo_JJU\project\programs.json"
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # 📌 JSON 키 자동 감지
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

    # 📌 특정 월 필터 (ex: "2월", "3월")
    month_match = re.search(r"(\d{1,2})월", query_lower)
    month_filter = month_match.group(1) if month_match else None

    # 📌 특정 키워드 감지 (점프업, NCS 등)
    keywords = ["점프업 포인트", "비교과 포인트", "ncs", "멘토링", "창업", "자격증", "특강"]
    matched_keywords = [kw for kw in keywords if kw in query_lower]

    # 📌 특정 대상 필터 (ex: "3학년", "1학년", "졸업 예정자")
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

        # 📌 "기간" 필드에서 특정 월이 포함된지 확인
        if month_filter and not re.search(rf"{month_filter}\.", period):  # "2." → "2월"을 의미
            continue  # 특정 월이 포함되지 않으면 필터링

        # 📌 특정 키워드 필터링
        if matched_keywords and not any(kw in description or kw in title or kw in benefits for kw in matched_keywords):
            continue

        # 📌 특정 대상(학년) 필터링
        if target_filter and target_filter not in target:
            continue

        results.append(program)

    return results

# ✅ 응답 메시지 동적 생성 함수
def generate_response(query, results):
    month_filter, matched_keywords, target_filter = extract_filters(query)

    # 📌 질문 유형에 따른 맞춤형 제목 설정
    if matched_keywords:
        response_title = f"**📌 {' '.join(matched_keywords)} 관련 프로그램입니다:**"
    elif target_filter:
        response_title = f"**📌 {target_filter} 대상 추천 비교과 프로그램입니다:**"
    elif month_filter:
        response_title = f"**📌 {month_filter}월 진행되는 비교과 프로그램입니다:**"
    else:
        response_title = "**📌 추천 비교과 프로그램입니다:**"

    # 📌 검색 결과 출력
    if results:
        response_content = response_title + "\n\n" + "\n\n".join(
            [f"🔹 **{p['제목']}**\n📌 설명: {p['설명']}\n📅 기간: {p['기간']}\n📍 장소: {p['장소']}\n🎁 혜택: {p['혜택']}\n🎯 신청대상: {p['신청대상']}\n📞 문의처: {p['문의처']}" for p in results]
        )
    else:
        response_content = "⚠️ 해당 조건에 맞는 비교과 프로그램을 찾을 수 없습니다. 다른 키워드로 검색해보세요!"

    return response_content

# ✅ 채팅 UI
chat_container = st.container()

# ✅ 대화 기록 표시
for msg in st.session_state.get("messages", []):
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

    # ✅ 검색 실행
    program_results = find_program(user_input)
    response_content = generate_response(user_input, program_results)

    st.session_state["messages"].append(AIMessage(content=response_content))

    # ✅ 채팅 UI 즉시 갱신
    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(response_content)
