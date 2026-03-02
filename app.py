import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from supabase import create_client, Client
import pandas as pd
import tempfile
import os
import json # 新增：用于解析 AI 返回的代码格式数据

# --- 1. 🔑 核心配置区 ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

USER_DATABASE = {
    "admin": "123456",       
    "friend1": "ielts75"
}

# --- 2. 🛡️ 登录系统初始化 ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = ""

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

# --- 3. 🎙️ 核心主界面 ---
else:
    current_user = st.session_state.current_user
    st.sidebar.write(f"👤 当前练习者：**{current_user}**")
    
    # --- 👑 核心升级：管理员专属的“智能视觉上传通道” ---
    if current_user == "admin":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ 管理员后台")
        # 魔法点1：现在允许上传 PDF 了！
        uploaded_file = st.sidebar.file_uploader("📂 智能导入机经 (CSV / PDF)", type=["csv", "pdf"])
        # 👇 新增：极其危险但好用的“一键清空”核按钮
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗑️ 危险操作区")
        # type="primary" 会让这个按钮变成醒目的红色！
        if st.sidebar.button("🚨 一键清空所有题库", type="primary"):
            with st.spinner("正在销毁所有题目..."):
                # 数据库操作：删除 id 不等于 0 的所有数据（也就是全删）
                supabase.table("question_bank").delete().neq("id", 0).execute()
            st.sidebar.success("✅ 题库已彻底清空！请手动刷新网页。")
        if uploaded_file is not None:
            if st.sidebar.button("🚀 启动智能分析与导入"):
                
                # 情况A：如果是老老实实的 CSV 表格
                if uploaded_file.name.endswith('.csv'):
                    with st.spinner("正在写入表格数据..."):
                        df = pd.read_csv(uploaded_file)
                        for index, row in df.iterrows():
                            supabase.table("question_bank").insert({
                                "category": str(row["category"]),
                                "question_text": str(row["question"])
                            }).execute()
                    st.sidebar.success("✅ CSV 题库导入成功！请刷新网页。")
                
                # 情况B：如果是极其硬核的 PDF 机经文件！
                elif uploaded_file.name.endswith('.pdf'):
                    with st.spinner("🤖 正在召唤 Gemini 大脑阅读 PDF 并提取题目，这可能需要几十秒..."):
                        # 1. 先把 PDF 存在一个安全的临时文件里
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                            tmp_pdf.write(uploaded_file.read())
                            pdf_path = tmp_pdf.name
                        
                        try:
                            # 2. 把文件传给 AI
                            pdf_file = client.files.upload(file=pdf_path)
                            
                            # 3. 下达极其严苛的指令，要求它输出程序能看懂的 JSON 格式
                            prompt = """
                            你是一个极其精准的数据提取程序。请阅读这份雅思机经/题库 PDF 文件，提取出所有的口语题目。
                            请严格将结果以 JSON 数组的形式返回。每一个元素是一个字典，包含两个键：
                            "category"（题目所属的分类，比如 Part 1: Hometown, Part 2: Technology）
                            "question"（具体的英文题目）。
                            绝对不要输出任何 markdown 标记、绝对不要包含 ```json 这样的开头，只输出纯文本的 JSON 数组。
                            示例：[{"category": "Part 1: Daily Life", "question": "Do you work or are you a student?"}]
                            """
                            
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[pdf_file, prompt]
                            )
                            
                            # 4. 清理 AI 返回的内容并解析成 Python 字典
                            raw_text = response.text.strip()
                            # 预防 AI 调皮加上 markdown 格式
                            if raw_text.startswith("```json"): raw_text = raw_text[7:]
                            if raw_text.startswith("```"): raw_text = raw_text[3:]
                            if raw_text.endswith("```"): raw_text = raw_text[:-3]
                            
                            extracted_data = json.loads(raw_text.strip())
                            
                            st.sidebar.info(f"✨ 成功从 PDF 提取到 {len(extracted_data)} 道题目！正在高压推入数据库...")
                            
                            # 5. 把提取出来的题目批量打进数据库！
                            for item in extracted_data:
                                supabase.table("question_bank").insert({
                                    "category": str(item.get("category", "未分类")),
                                    "question_text": str(item.get("question", "提取失败"))
                                }).execute()
                                
                            st.sidebar.success("✅ PDF 智能导入成功！快去中间抽题吧！")
                            
                        except Exception as e:
                            st.sidebar.error(f"解析过程发生短路：{e} (可能是 PDF 太复杂导致 AI 格式输出错误，建议截取少量页数重试)")
                        finally:
                            os.remove(pdf_path) # 无论成功与否，销毁临时 PDF
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

    st.title(f"专属口语考场 🎙️")
    
    db_questions = supabase.table("question_bank").select("*").execute()
    
    IELTS_BANK = {}
    for row in db_questions.data:
        cat = row["category"]
        q = row["question_text"]
        if cat not in IELTS_BANK:
            IELTS_BANK[cat] = []
        IELTS_BANK[cat].append(q)

    if not IELTS_BANK:
        IELTS_BANK = {"未分类": ["当前题库为空，请联系管理员在左侧上传题库。"]}

    st.subheader("📝 Step 1: 从题库中抽题")
    category = st.selectbox("📂 选择题库分类：", list(IELTS_BANK.keys()))
    question = st.selectbox("🎯 选择具体题目：", IELTS_BANK[category])
    st.info(f"**考官提问：** {question}")

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
                
                st.success("🎉 考官点评完成！")
                st.markdown(response.text)
                st.balloons()
                
                supabase.table("practice_history").insert({
                    "username": current_user,
                    "question": question,
                    "record_text": response.text
                }).execute()
                
            except Exception as e:
                st.error(f"发生了一点小意外：{e}")
                
            os.remove(tmp_file_path)

