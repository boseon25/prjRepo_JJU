import os
import json
import re  
import chromadb
import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# ✅ 환경 변수 로드
load_dotenv()

# ✅ OpenAI API 키
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ ChromaDB 초기화
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="programs")

# ✅ OpenAI Embeddings 설정
embed_model = OpenAIEmbeddings()

# ✅ Streamlit UI 설정
st.set_page_config(page_title="전주대학교 비교과 챗봇", page_icon="🎓", layout="centered")
st.title("🎓 전주대학교 비교과 챗봇")
st.write("전주대학교 비교과 프로그램에 대해 질문하세요!")

# ✅ 세션 상태 초기화 (에러 방지)
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ✅ JSON 데이터 로드
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

# ✅ ChromaDB에 데이터 추가 함수
def add_data_to_chroma():
    existing_items = collection.get()
    if existing_items and "ids" in existing_items and existing_items["ids"]:
        collection.delete(ids=existing_items["ids"])  # 저장된 데이터가 있을 경우에만 삭제 수행

    for program in program_data:
        doc_id = program.get("제목", "Unknown")
        content = f"{program.get('제목', '')} {program.get('설명', '')} {program.get('혜택', '')}"
        vector = embed_model.embed_query(content)

        # ✅ metadata에서 리스트 값을 문자열로 변환
        processed_metadata = {}
        for key, value in program.items():
            if isinstance(value, list):  
                processed_metadata[key] = ", ".join(value)  # 리스트를 문자열로 변환
            else:
                processed_metadata[key] = value 

        collection.add(ids=[doc_id], embeddings=[vector], metadatas=[processed_metadata])

add_data_to_chroma()  # 데이터 삽입

# ✅ 질문에서 키워드 추출
def extract_filters(query):
    query_lower = query.lower()
    month_match = re.search(r"(\d{1,2})월", query_lower)
    month_filter = month_match.group(1) if month_match else None
    keywords = ["점프업 포인트", "비교과 포인트", "ncs", "멘토링", "창업", "자격증", "특강"]
    matched_keywords = [kw for kw in keywords if kw in query_lower]
    target_match = re.search(r"(\d학년|졸업 예정자)", query_lower)
    target_filter = target_match.group(1) if target_match else None

    return month_filter, matched_keywords, target_filter

# ✅ 벡터 검색 기반 프로그램 추천
def search_similar_programs(query):
    query_vector = embed_model.embed_query(query)
    results = collection.query(query_embeddings=[query_vector], n_results=3)
    return results["metadatas"][0] if "metadatas" in results else []

# ✅ 키워드 기반 검색
def find_program(query):
    month_filter, matched_keywords, target_filter = extract_filters(query)
    results = []

    for program in program_data:
        title = program.get("제목", "").lower()
        description = program.get("설명", "").lower()
        period = program.get("기간", "").lower()
        benefits = program.get("혜택", "").lower()
        target = program.get("신청대상", "").lower()

        if month_filter and not re.search(rf"{month_filter}\.", period):  
            continue
        if matched_keywords and not any(kw in description or kw in title or kw in benefits for kw in matched_keywords):
            continue
        if target_filter and target_filter not in target:
            continue

        results.append(program)

    return results

# ✅ 응답 생성 함수
def generate_response(query, results):
    month_filter, matched_keywords, target_filter = extract_filters(query)

    if matched_keywords:
        response_title = f"**📌 {' '.join(matched_keywords)} 관련 프로그램입니다:**"
    elif target_filter:
        response_title = f"**📌 {target_filter} 대상 추천 비교과 프로그램입니다:**"
    elif month_filter:
        response_title = f"**📌 {month_filter}월 진행되는 비교과 프로그램입니다:**"
    else:
        response_title = "**📌 추천 비교과 프로그램입니다:**"

    if results:
        response_content = response_title + "\n\n" + "\n\n".join(
            [f"🔹 **{p['제목']}**\n📌 설명: {p['설명']}\n📅 기간: {p['기간']}\n📍 장소: {p['장소']}\n🎁 혜택: {p['혜택']}\n🎯 신청대상: {p['신청대상']}\n📞 문의처: {p['문의처']}" for p in results]
        )
    else:
        response_content = "⚠️ 해당 조건에 맞는 비교과 프로그램을 찾을 수 없습니다. 다음과 같은 프로그램이 있습니다:\n\n"
        alternative_results = program_data[:5]
        response_content += "\n\n".join(
            [f"🔹 **{p['제목']}**\n📌 설명: {p['설명']}\n📅 기간: {p['기간']}" for p in alternative_results]
        )

    return response_content

# ✅ 채팅 UI
chat_container = st.container()

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

    similar_results = search_similar_programs(user_input)
    keyword_results = find_program(user_input)
    final_results = similar_results + keyword_results

    response_content = generate_response(user_input, final_results)
    st.session_state["messages"].append(AIMessage(content=response_content))

    with chat_container:
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            st.write(response_content)
