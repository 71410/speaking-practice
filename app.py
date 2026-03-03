import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from supabase import create_client, Client
import pandas as pd
import tempfile
import os
import json

# --- 1. 🔑 核心配置区 ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

client = genai.Client(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

USER_DATABASE = st.secrets["passwords"]

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

else:
    current_user = st.session_state.current_user
    st.sidebar.write(f"👤 当前练习者：**{current_user}**")
    
    # --- 👑 管理员后台（支持三级结构导入） ---
    if current_user == "admin":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ 管理员后台")
        uploaded_file = st.sidebar.file_uploader("📂 智能导入机经 (CSV / PDF)", type=["csv", "pdf"])
        
        if uploaded_file is not None:
            if st.sidebar.button("🚀 启动智能分析与导入"):
                
                if uploaded_file.name.endswith('.csv'):
                    with st.spinner("正在写入表格数据..."):
                        df = pd.read_csv(uploaded_file)
                        for index, row in df.iterrows():
                            supabase.table("question_bank").insert({
                                "part": str(row["part"]),
                                "theme": str(row["theme"]),
                                "question_text": str(row["question"])
                            }).execute()
                    st.sidebar.success("✅ CSV 题库导入成功！请刷新网页。")
                
                elif uploaded_file.name.endswith('.pdf'):
                    with st.spinner("🤖 正在召唤大脑阅读 PDF..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                            tmp_pdf.write(uploaded_file.read())
                            pdf_path = tmp_pdf.name
                        
                        try:
                            pdf_file = client.files.upload(file=pdf_path)
                            
                            # ⚠️ 核心指令升级：要求 AI 提取 part 和 theme
                            prompt = """
                            你是一个极其精准的数据提取程序。请阅读这份雅思机经/题库 PDF 文件，提取出所有的口语题目。
                            请严格将结果以 JSON 数组的形式返回。每一个元素是一个字典，包含三个键：
                            "part"（如 "Part 1", "Part 2", "Part 3"）
                            "theme"（题目的具体主题，如 "Hometown", "Technology"）
                            "question"（具体的英文题目）。
                            绝对不要输出任何 markdown 标记、绝对不要包含 ```json 这样的开头，只输出纯文本。
                            示例：[{"part": "Part 1", "theme": "Daily Life", "question": "Do you work?"}]
                            """
                            
                            response = client.models.generate_content(
                                model='gemini-2.5-flash',
                                contents=[pdf_file, prompt]
                            )
                            
                            raw_text = response.text.strip()
                            if raw_text.startswith("```json"): raw_text = raw_text[7:]
                            if raw_text.startswith("```"): raw_text = raw_text[3:]
                            if raw_text.endswith("```"): raw_text = raw_text[:-3]
                            
                            extracted_data = json.loads(raw_text.strip())
                            
                            st.sidebar.info(f"✨ 成功提取 {len(extracted_data)} 道题目！正在推入数据库...")
                            
                            for item in extracted_data:
                                supabase.table("question_bank").insert({
                                    "part": str(item.get("part", "未分类")),
                                    "theme": str(item.get("theme", "未分类")),
                                    "question_text": str(item.get("question", "提取失败"))
                                }).execute()
                                
                            st.sidebar.success("✅ PDF 智能导入成功！快去中间抽题吧！")
                            
                        except Exception as e:
                            st.sidebar.error(f"解析过程发生短路：{e}")
                        finally:
                            os.remove(pdf_path)
                            
        st.sidebar.markdown("---")
        st.sidebar.subheader("🗑️ 危险操作区")
        if st.sidebar.button("🚨 一键清空所有题库", type="primary"):
            with st.spinner("正在销毁所有题目..."):
                supabase.table("question_bank").delete().neq("id", 0).execute()
            st.sidebar.success("✅ 题库已彻底清空！请手动刷新网页。")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

    st.title(f"专属口语考场 🎙️")
    
    # --- 📚 核心升级：构建三级结构字典 ---
    db_questions = supabase.table("question_bank").select("*").execute()
    
    IELTS_BANK = {}
    for row in db_questions.data:
        p = row.get("part", "未分类")
        t = row.get("theme", "未分类")
        q = row.get("question_text", "提取失败")
        
        if p not in IELTS_BANK:
            IELTS_BANK[p] = {}
        if t not in IELTS_BANK[p]:
            IELTS_BANK[p][t] = []
        IELTS_BANK[p][t].append(q)

    st.subheader("📝 Step 1: 从题库中抽题")
    
    if not IELTS_BANK:
        st.info("当前题库为空，请联系管理员在左侧上传题库。")
    else:
        # 三个下拉菜单闪亮登场！
        selected_part = st.selectbox("📂 选择 Part：", list(IELTS_BANK.keys()))
        selected_theme = st.selectbox("🏷️ 选择主题 (Theme)：", list(IELTS_BANK[selected_part].keys()))
        question = st.selectbox("🎯 选择具体题目：", IELTS_BANK[selected_part][selected_theme])
        
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
