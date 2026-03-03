import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from supabase import create_client, Client
import pandas as pd
import tempfile
import os
import json
from gtts import gTTS

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
    st.title("🔐 高分英语训练舱 - 内部邀请版")
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
    
    # --- 👑 终极版管理员后台：支持文件与纯文本双通道导入 ---
    if current_user == "admin":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ 管理员后台")
        
        upload_target = st.sidebar.radio("🎯 选择导入目标：", ["🗣️ 口语题库", "📖 阅读文章库"])
        
        # ==========================
        # 分支 A：导入【口语题库】(保持文件上传)
        # ==========================
        if upload_target == "🗣️ 口语题库":
            uploaded_file = st.sidebar.file_uploader("📂 智能导入口语题 (CSV / PDF)", type=["csv", "pdf"])
            if uploaded_file is not None:
                if st.sidebar.button("🚀 启动智能分析与导入"):
                    if uploaded_file.name.endswith('.csv'):
                        with st.spinner("正在写入口语表格..."):
                            df = pd.read_csv(uploaded_file)
                            for index, row in df.iterrows():
                                supabase.table("question_bank").insert({
                                    "part": str(row["part"]),
                                    "theme": str(row["theme"]),
                                    "question_text": str(row["question"])
                                }).execute()
                        st.sidebar.success("✅ 口语 CSV 导入成功！")
                    
                    elif uploaded_file.name.endswith('.pdf'):
                        with st.spinner("🤖 正在召唤大脑提取口语题目..."):
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                                tmp_pdf.write(uploaded_file.read())
                                pdf_path = tmp_pdf.name
                            try:
                                pdf_file = client.files.upload(file=pdf_path)
                                prompt = """
                                提取雅思口语题目，返回 JSON 数组。键名："part", "theme", "question"。只输出纯文本 JSON。
                                """
                                response = client.models.generate_content(model='gemini-2.5-flash', contents=[pdf_file, prompt])
                                raw_text = response.text.strip()
                                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                if raw_text.startswith("```"): raw_text = raw_text[3:]
                                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                extracted_data = json.loads(raw_text.strip())
                                
                                for item in extracted_data:
                                    supabase.table("question_bank").insert({
                                        "part": str(item.get("part", "未分类")),
                                        "theme": str(item.get("theme", "未分类")),
                                        "question_text": str(item.get("question", "提取失败"))
                                    }).execute()
                                st.sidebar.success(f"✅ 成功导入 {len(extracted_data)} 道口语题！")
                            except Exception as e:
                                st.sidebar.error(f"解析短路：{e}")
                            finally:
                                os.remove(pdf_path)

        # ==========================
        # 分支 B：导入【阅读文章库】(增加手机端极其友好的纯文本通道！)
        # ==========================
        elif upload_target == "📖 阅读文章库":
            input_method = st.sidebar.radio("📥 录入方式：", ["📁 文件上传", "✍️ 手动粘贴文本"])
            
            # 情况 1：传统的批量传文件
            if input_method == "📁 文件上传":
                uploaded_file = st.sidebar.file_uploader("📂 导入阅读文章 (CSV / PDF)", type=["csv", "pdf"])
                if uploaded_file is not None:
                    if st.sidebar.button("🚀 启动智能分析与导入"):
                        if uploaded_file.name.endswith('.csv'):
                            with st.spinner("正在写入阅读表格..."):
                                df = pd.read_csv(uploaded_file)
                                for index, row in df.iterrows():
                                    supabase.table("reading_bank").insert({
                                        "title": str(row["title"]),
                                        "content": str(row["content"])
                                    }).execute()
                            st.sidebar.success("✅ 阅读 CSV 导入成功！")
                        
                        elif uploaded_file.name.endswith('.pdf'):
                            with st.spinner("🤖 正在召唤大脑拆解阅读文章..."):
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                                    tmp_pdf.write(uploaded_file.read())
                                    pdf_path = tmp_pdf.name
                                try:
                                    pdf_file = client.files.upload(file=pdf_path)
                                    prompt = """
                                    你是一个数据提取程序。请从这份 PDF 中提取出适合英语朗读的段落或文章。
                                    请严格以 JSON 数组返回。每个元素包含两个键：
                                    "title"（文章或段落的标题/概括）
                                    "content"（具体的英文原文正文）。
                                    只输出纯文本 JSON，绝对不要包含 ```json 标记。
                                    """
                                    response = client.models.generate_content(model='gemini-2.5-flash', contents=[pdf_file, prompt])
                                    raw_text = response.text.strip()
                                    if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                    if raw_text.startswith("```"): raw_text = raw_text[3:]
                                    if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                    extracted_data = json.loads(raw_text.strip())
                                    
                                    for item in extracted_data:
                                        supabase.table("reading_bank").insert({
                                            "title": str(item.get("title", "未命名文章")),
                                            "content": str(item.get("content", "内容提取失败"))
                                        }).execute()
                                    st.sidebar.success(f"✅ 成功导入 {len(extracted_data)} 篇阅读文章！")
                                except Exception as e:
                                    st.sidebar.error(f"解析短路：{e}")
                                finally:
                                    os.remove(pdf_path)

            # 情况 2：极其轻量、适合手机端的直接粘贴法
            elif input_method == "✍️ 手动粘贴文本":
                manual_title = st.sidebar.text_input("🏷️ 文章标题 (如: 经济学人每日晨读)")
                # height=250 让输入框在手机上也有足够的高度方便检查文本
                manual_content = st.sidebar.text_area("📝 文章正文 (直接在这里粘贴纯英文段落)", height=250)
                
                if st.sidebar.button("🚀 闪电保存至数据库", type="primary"):
                    if manual_title.strip() and manual_content.strip():
                        with st.spinner("正在安全归档..."):
                            supabase.table("reading_bank").insert({
                                "title": manual_title.strip(),
                                "content": manual_content.strip()
                            }).execute()
                        st.sidebar.success(f"✅ 《{manual_title}》已成功存入你的阅读库！刷新网页即可朗读。")
                    else:
                        st.sidebar.warning("⚠️ 标题和正文都不能为空哦！")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🗑️ 危险操作区")
        if st.sidebar.button("🚨 一键清空口语题库", type="primary"):
            supabase.table("question_bank").delete().neq("id", 0).execute()
            st.sidebar.success("✅ 口语题库已清空！")
        if st.sidebar.button("🚨 一键清空阅读文章", type="primary"):
            supabase.table("reading_bank").delete().neq("id", 0).execute()
            st.sidebar.success("✅ 阅读文章库已清空！")
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

    st.title(f"专属英语训练舱 🚀")
    
    tab_qa, tab_reading = st.tabs(["🗣️ 雅思口语问答", "📖 英式朗读纠音"])
    
    # ==========================================
    # 模块一：口语问答 
    # ==========================================
    with tab_qa:
        db_questions = supabase.table("question_bank").select("*").execute()
        IELTS_BANK = {}
        for row in db_questions.data:
            p = row.get("part", "未分类")
            t = row.get("theme", "未分类")
            q = row.get("question_text", "提取失败")
            if p not in IELTS_BANK: IELTS_BANK[p] = {}
            if t not in IELTS_BANK[p]: IELTS_BANK[p][t] = []
            IELTS_BANK[p][t].append(q)

        st.subheader("📝 Step 1: 从题库中抽题")
        if not IELTS_BANK:
            st.info("当前题库为空，请联系管理员在左侧上传题库。")
        else:
            selected_part = st.selectbox("📂 选择 Part：", list(IELTS_BANK.keys()), key="qa_part")
            selected_theme = st.selectbox("🏷️ 选择主题 (Theme)：", list(IELTS_BANK[selected_part].keys()), key="qa_theme")
            question = st.selectbox("🎯 选择具体题目：", IELTS_BANK[selected_part][selected_theme], key="qa_q")
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
            audio_bytes_qa = audio_recorder(text="点击麦克风开始作答", icon_size="2x", key="recorder_qa")

            if audio_bytes_qa:
                st.audio(audio_bytes_qa, format="audio/wav")
                with st.spinner("🧠 考官正在仔细聆听并评估..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_bytes_qa)
                        tmp_file_path = tmp_file.name
                    try:
                        audio_file = client.files.upload(file=tmp_file_path)
                        prompt = f"""
                        你现在是一名雅思口语考官。考生 {current_user} 正在回答题目：“{question}”。
                        请你：
                        1. 【精准听写】：写下听到的英文原话。
                        2. 【切题度与雅思预估分】：评价是否切题，给出预估分数。
                        3. 【纠错与升级】：给出 2 个针对这道题的高阶示范回答。
                        4. 【考官建议】：用中文给一段备考建议。
                        """
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=[audio_file, prompt])
                        st.success("🎉 考官点评完成！")
                        st.markdown(response.text)
                        
                        supabase.table("practice_history").insert({
                            "username": current_user,
                            "question": question,
                            "record_text": response.text
                        }).execute()
                    except Exception as e:
                        st.error(f"发生小意外：{e}")
                    os.remove(tmp_file_path)

    # ==========================================
    # 模块二：英式朗读纠音
    # ==========================================
    with tab_reading:
        st.subheader("📖 英文原版朗读纠音")
        
        db_readings = supabase.table("reading_bank").select("*").execute()
        READING_MATERIALS = {row["title"]: row["content"] for row in db_readings.data}
        
        if not READING_MATERIALS:
            st.info("当前阅读库为空。请用 admin 账号在左侧侧边栏选择【📖 阅读文章库】上传或粘贴文本。")
        else:
            reading_title = st.selectbox("📂 选择朗读材料：", list(READING_MATERIALS.keys()), key="sel_reading")
            reading_text = READING_MATERIALS[reading_title]
            
            st.markdown(f"**请仔细朗读以下段落：**\n> ### {reading_text}")
            
            if st.button("🎧 听专业英式播音员朗读"):
                with st.spinner("正在呼叫伦敦总部的播音员..."):
                    tts = gTTS(text=reading_text, lang='en', tld='co.uk')
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_tts:
                        tts.save(tmp_tts.name)
                        st.audio(tmp_tts.name, format="audio/mp3")

            reading_db_response = supabase.table("reading_history").select("record_text").eq("username", current_user).eq("reading_title", reading_title).execute()
            past_reading_records = reading_db_response.data
            
            if len(past_reading_records) > 0:
                with st.expander(f"📖 查看这篇短文的 {len(past_reading_records)} 次历史纠音记录"):
                    for i, record in enumerate(past_reading_records):
                        st.markdown(f"**▶ 第 {i+1} 次跟读：**")
                        st.write(record["record_text"])
                        st.write("---")

            st.write("---")
            st.subheader("🎙️ 轮到你了")
            audio_bytes_reading = audio_recorder(text="点击录制你的朗读", icon_size="2x", key="recorder_reading")

            if audio_bytes_reading:
                st.audio(audio_bytes_reading, format="audio/wav")
                with st.spinner("🧠 纠音导师正在逐字核对你的发音..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(audio_bytes_reading)
                        tmp_file_path = tmp_file.name
                    try:
                        audio_file = client.files.upload(file=tmp_file_path)
                        prompt = f"""
                        你现在是一名雅思口语从业多年的考官兼流利度教练。考生正在朗读这段指定的文本：“{reading_text}”
                        我已经上传了考生的录音。
                        请注意：**绝对不要纠结考生的口音是英式还是美式**，只要发音清晰即可。你的重点是按照雅思口语的发音（PR）和流利度（FC）标准来进行严苛评判。
                        请严格按以下格式输出反馈：
                        1. 【流利度与节奏】：评价朗读时的语速、停顿是否合理，有无不自然的卡顿、结巴或频繁的自我纠正。
                        2. 【发音准确度（错词/漏词）】：精准指出他严重读错、漏读或多读的具体单词。
                        3. 【语音语调（重音与连读）】：评价考生的句子意群断句（Chunking）、单词重音（Word Stress）和连读（Linking）是否自然，是否具备英语母语者的语感。
                        4. 【考官提分建议】：给出一段犀利且实用的综合提升建议，帮助考生在雅思实战中听起来更自然、更自信。
                        """
                        response = client.models.generate_content(model='gemini-2.5-flash', contents=[audio_file, prompt])
                        st.success("🎉 发音诊断报告已生成！")
                        st.markdown(response.text)
                        st.balloons()
                        
                        supabase.table("reading_history").insert({
                            "username": current_user,
                            "reading_title": reading_title,
                            "record_text": response.text
                        }).execute()
                    except Exception as e:
                        st.error(f"发生小意外：{e}")
                    os.remove(tmp_file_path)

