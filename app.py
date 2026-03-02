import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from supabase import create_client, Client
import tempfile
import os

# --- 1. 🔑 核心配置区（从云端保险箱拿所有钥匙） ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# 初始化 AI 大脑和数据库
client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

USER_DATABASE = {
    "admin": "123456",       
    "friend1": "ielts75"
}

IELTS_BANK = {
    "Part 1: 个人生活 (Daily Life)": [
        "Do you work or are you a student?",
        "What do you usually do in your free time?",
        "Tell me about the city you live in."
    ],
    "Part 2: 深入描述 (Long Turn)": [
        "Describe a piece of technology you find useful.",
        "Describe a time when you had to make a difficult decision.",
        "Describe a movie or website you like."
    ]
}

# --- 2. 🛡️ 登录系统初始化 ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = ""

# --- 3. 🚪 登录界面 ---
if not st.session_state.logged_in:
    st.title("🔐 高分雅思模拟室 - 内部邀请版")
    username = st.text_input("👤 账号")
    password = st.text_input("🔑 密码", type="password")
    
    if st.button("登录"):
        if username in USER_DATABASE and USER_DATABASE[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success(f"登录成功！欢迎回来，{username}！")
            st.rerun()
        else:
            st.error("❌ 账号或密码错误！")

# --- 4. 🎙️ 核心主界面 ---
else:
    current_user = st.session_state.current_user
    
    st.sidebar.write(f"👤 当前练习者：**{current_user}**")
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

    st.title(f"专属口语考场 🎙️")
    st.subheader("📝 Step 1: 从题库中抽题")
    
    category = st.selectbox("📂 选择题库分类：", list(IELTS_BANK.keys()))
    question = st.selectbox("🎯 选择具体题目：", IELTS_BANK[category])
    st.info(f"**考官提问：** {question}")

    # --- 🗂️ 核心升级：从 Supabase 数据库实时读取历史记录 ---
    db_response = supabase.table("practice_history").select("record_text").eq("username", current_user).eq("question", question).execute()
    past_records = db_response.data
    
    if len(past_records) > 0:
        with st.expander(f"📖 查看这道题的 {len(past_records)} 次历史点评记录"):
            for i, record in enumerate(past_records):
                st.markdown(f"**▶ 第 {i+1} 次练习：**")
                st.write(record["record_text"])
                st.write("---")
    else:
        st.caption("✨ 这道题你还没练过，现在开始第一次尝试吧！")

    # --- 🎙️ 录音与 AI 点评区 ---
    st.write("---")
    st.subheader("🗣️ Step 2: 你的回答")
    audio_bytes = audio_recorder(text="点击麦克风开始作答", icon_size="2x")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("🧠 考官正在仔细聆听并评估..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
                
            try:
                audio_file = client.files.upload(file=tmp_file_path)
                
                prompt = f"""
                你现在是一名雅思口语考官。考生 {current_user} 正在回答题目：“{question}”。
                我已经上传了考生的回答录音。请你：
                1. 【精准听写】：写下听到的英文原话。
                2. 【切题度与雅思预估分】：评价是否切题，并给出预估分数。
                3. 【纠错与升级】：严厉指出错误，并给出 2 个针对这道题的高阶示范回答。
                4. 【考官建议】：用中文给一段犀利的备考建议。
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[audio_file, prompt]
                )
                
                # 展现本次点评
                st.success("🎉 考官点评完成！")
                st.markdown(response.text)
                st.balloons()
                
                # --- 🗂️ 核心升级：把点评结果永久写入 Supabase 数据库！ ---
                supabase.table("practice_history").insert({
                    "username": current_user,
                    "question": question,
                    "record_text": response.text
                }).execute()
                
            except Exception as e:
                st.error(f"发生了一点小意外：{e}")
                
            os.remove(tmp_file_path)
