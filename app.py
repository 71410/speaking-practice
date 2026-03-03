import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from supabase import create_client, Client
import pandas as pd
import tempfile
import os
import json
import re  
from gtts import gTTS
import base64
import io
from openai import OpenAI
import PyPDF2

# --- 1. 🔑 核心配置区 (中西合璧：DeepSeek + Gemini) ---
GEMINI_API_KEY_VOICE = st.secrets["GEMINI_API_KEY_VOICE"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

# 引擎 A：负责后台苦力（DeepSeek 文本解析）
client_admin = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com" # DeepSeek 官方接口
)

# 引擎 B：负责前台考官（Gemini 语音打分）
client_voice = genai.Client(api_key=GEMINI_API_KEY_VOICE)

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
    
    # --- 👑 管理员后台 (DeepSeek 接管 PDF 解析) ---
    if current_user == "admin":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ 管理员后台")
        upload_target = st.sidebar.radio("🎯 选择导入目标：", ["🗣️ 口语题库", "📖 阅读文章库"])
        
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
                        with st.spinner("🤖 正在召唤 DeepSeek 大脑提取题目..."):
                            try:
                                reader = PyPDF2.PdfReader(uploaded_file)
                                pdf_text = ""
                                for page in reader.pages:
                                    pdf_text += page.extract_text() + "\n"
                                
                                prompt = f"""
                                提取以下文本中的所有雅思口语题目。
                                请严格将结果以 JSON 数组的形式返回。每一个元素包含三个键：
                                "part"（如 "Part 1", "Part 2"）、"theme"（主题）、"question"（具体英文题目）。
                                绝对不要输出任何 markdown 标记、不要废话，只输出纯文本 JSON 数组。
                                \n\n【源文本】:\n{pdf_text[:30000]}
                                """
                                response = client_admin.chat.completions.create(
                                    model="deepseek-chat",
                                    messages=[
                                        {"role": "system", "content": "You are a precise JSON data extraction tool. Output strictly valid JSON arrays without markdown syntax."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.1
                                )
                                
                                raw_text = response.choices[0].message.content.strip()
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
                                st.sidebar.success(f"✅ DeepSeek 成功导入 {len(extracted_data)} 道口语题！")
                            except Exception as e:
                                st.sidebar.error(f"DeepSeek 解析短路：{e}")

        elif upload_target == "📖 阅读文章库":
            input_method = st.sidebar.radio("📥 录入方式：", ["📁 文件上传", "✍️ 手动粘贴文本"])
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
                            with st.spinner("🤖 正在召唤 DeepSeek 大脑拆解文章..."):
                                try:
                                    reader = PyPDF2.PdfReader(uploaded_file)
                                    pdf_text = ""
                                    for page in reader.pages:
                                        pdf_text += page.extract_text() + "\n"
                                    
                                    prompt = f"""
                                    提取以下文本中适合英语朗读的段落或文章。
                                    请严格以 JSON 数组返回。每个元素包含两个键：
                                    "title"（文章或段落的标题/概括）、"content"（具体的英文原文正文）。
                                    绝对不要输出任何 markdown 标记、不要废话，只输出纯文本 JSON 数组。
                                    \n\n【源文本】:\n{pdf_text[:30000]}
                                    """
                                    response = client_admin.chat.completions.create(
                                        model="deepseek-chat",
                                        messages=[
                                            {"role": "system", "content": "You are a precise JSON data extraction tool. Output strictly valid JSON arrays without markdown syntax."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.1
                                    )
                                    
                                    raw_text = response.choices[0].message.content.strip()
                                    if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                    if raw_text.startswith("```"): raw_text = raw_text[3:]
                                    if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                    
                                    extracted_data = json.loads(raw_text.strip())
                                    for item in extracted_data:
                                        supabase.table("reading_bank").insert({
                                            "title": str(item.get("title", "未命名文章")),
                                            "content": str(item.get("content", "内容提取失败"))
                                        }).execute()
                                    st.sidebar.success(f"✅ DeepSeek 成功导入 {len(extracted_data)} 篇阅读文章！")
                                except Exception as e:
                                    st.sidebar.error(f"DeepSeek 解析短路：{e}")
                                    
            elif input_method == "✍️ 手动粘贴文本":
                manual_title = st.sidebar.text_input("🏷️ 文章标题")
                manual_content = st.sidebar.text_area("📝 文章正文", height=250)
                if st.sidebar.button("🚀 闪电保存至数据库", type="primary"):
                    if manual_title.strip() and manual_content.strip():
                        with st.spinner("正在安全归档..."):
                            supabase.table("reading_bank").insert({
                                "title": manual_title.strip(),
                                "content": manual_content.strip()
                            }).execute()
                        st.sidebar.success(f"✅ 《{manual_title}》已成功存入！")
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
    
    tab_qa, tab_reading = st.tabs(["🗣️ 雅思口语问答", "📖 英文原版朗读纠音"])
    
    # ==========================================
    # 模块一：口语问答 (使用 client_voice 当考官)
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

            st.write("---")
            st.subheader("🗣️ Step 2: 你的回答")
            
            qa_key_name = f"counter_{question}"
            if qa_key_name not in st.session_state:
                st.session_state[qa_key_name] = 0
                
            audio_bytes_qa = audio_recorder(
                text="点击麦克风开始作答", 
                icon_size="2x", 
                key=f"recorder_qa_{question}_{st.session_state[qa_key_name]}"
            )

            if audio_bytes_qa:
                st.audio(audio_bytes_qa, format="audio/wav")
                last_audio_tracker_qa = f"last_audio_{question}"
                
                if st.session_state.get(last_audio_tracker_qa) != audio_bytes_qa:
                    with st.spinner("🧠 专属考官 Voice 引擎正在仔细聆听..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes_qa)
                            tmp_file_path = tmp_file.name
                        try:
                            audio_file = client_voice.files.upload(file=tmp_file_path)
                            prompt = f"""
                            你现在是一名雅思口语考官。考生 {current_user} 正在回答题目：“{question}”。
                            请你：
                            1. 【精准听写】：写下听到的英文原话。
                            2. 【切题度与雅思预估分】：评价是否切题，给出预估分数。
                            3. 【纠错与升级】：给出 2 个针对这道题的高阶示范回答。
                            4. 【考官建议】：用中文给一段备考建议。
                            """
                            response = client_voice.models.generate_content(model='gemini-2.5-flash', contents=[audio_file, prompt])
                            st.success("🎉 考官点评完成！")
                            st.markdown(response.text)
                            
                            supabase.table("practice_history").insert({
                                "username": current_user,
                                "question": question,
                                "record_text": response.text
                            }).execute()
                            
                            st.session_state[last_audio_tracker_qa] = audio_bytes_qa
                            
                        except Exception as e:
                            st.error(f"Voice 引擎发生小意外：{e}")
                        os.remove(tmp_file_path)

                st.markdown("---")
                if st.button("🔄 不满意？清除录音，再练一次！", key=f"btn_qa_{question}_{st.session_state[qa_key_name]}"):
                    st.session_state[qa_key_name] += 1
                    st.rerun()

    # ==========================================
    # 模块二：英文原版朗读纠音 (使用 client_voice 当教练)
    # ==========================================
    with tab_reading:
        db_readings = supabase.table("reading_bank").select("*").execute()
        READING_MATERIALS = {row["title"]: row["content"] for row in db_readings.data}
        
        if not READING_MATERIALS:
            st.info("当前阅读库为空。请用 admin 账号在左侧侧边栏上传或粘贴文本。")
        else:
            reading_title = st.selectbox("📂 选择朗读材料：", list(READING_MATERIALS.keys()), key="sel_reading")
            reading_text = READING_MATERIALS[reading_title]
            
            practice_mode = st.radio("🎯 选择训练模式：", ["📖 全文连读", "🔍 逐句精读 (推荐)"], horizontal=True)
            st.write("---")
            
            if practice_mode == "📖 全文连读":
                target_text = reading_text
                db_save_title = reading_title
                st.markdown(f"**请仔细朗读以下完整段落：**\n> ### {target_text}")
            else:
                raw_sentences = re.split(r'(?<=[.!?])\s+', reading_text)
                sentences = [s.strip() for s in raw_sentences if s.strip()]
                if not sentences: sentences = [reading_text]
                    
                sentence_idx = st.selectbox(
                    "📍 选择要攻克的句子：", 
                    range(len(sentences)), 
                    format_func=lambda x: f"第 {x+1} 句: {sentences[x][:40]}..."
                )
                target_text = sentences[sentence_idx]
                db_save_title = f"{reading_title} (第{sentence_idx+1}句)"
                st.markdown(f"**请仔细朗读当前句子（第 {sentence_idx+1}/{len(sentences)} 句）：**\n> ### {target_text}")
            
            if st.button("🎧 听专业播音员示范"):
                with st.spinner("正在呼叫播音员..."):
                    tts = gTTS(text=target_text, lang='en', tld='co.uk')
                    sound_file = io.BytesIO()
                    tts.write_to_fp(sound_file)
                    sound_file.seek(0)
                    
                    b64 = base64.b64encode(sound_file.read()).decode()
                    md = f"""
                        <audio controls autoplay style="width: 100%;">
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                        您的浏览器不支持音频播放。
                        </audio>
                        """
                    st.markdown(md, unsafe_allow_html=True)

            reading_db_response = supabase.table("reading_history").select("record_text").eq("username", current_user).eq("reading_title", db_save_title).execute()
            past_reading_records = reading_db_response.data
            
            if len(past_reading_records) > 0:
                with st.expander(f"📖 查看此项的 {len(past_reading_records)} 次历史纠音记录"):
                    for i, record in enumerate(past_reading_records):
                        st.markdown(f"**▶ 第 {i+1} 次跟读：**")
                        st.write(record["record_text"])
                        st.write("---")

            st.write("---")
            st.subheader("🎙️ 轮到你了")
            
            reading_key_name = f"counter_{db_save_title}"
            if reading_key_name not in st.session_state:
                st.session_state[reading_key_name] = 0
                
            audio_bytes_reading = audio_recorder(
                text="点击录制你的朗读", 
                icon_size="2x", 
                key=f"recorder_reading_{db_save_title}_{st.session_state[reading_key_name]}"
            )

            if audio_bytes_reading:
                st.audio(audio_bytes_reading, format="audio/wav")
                last_audio_tracker_reading = f"last_audio_{db_save_title}"
                
                if st.session_state.get(last_audio_tracker_reading) != audio_bytes_reading:
                    with st.spinner("🧠 专属教练 Voice 引擎正在评估..."):
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(audio_bytes_reading)
                            tmp_file_path = tmp_file.name
                        try:
                            audio_file = client_voice.files.upload(file=tmp_file_path)
                            prompt = f"""
                            你现在是一名雅思口语考官兼流利度教练。考生正在朗读这段指定的文本：“{target_text}”
                            我已经上传了考生的录音。
                            请注意：**绝对不要纠结考生的口音是英式还是美式**，只要发音清晰即可。你的重点是按照雅思口语的发音（PR）和流利度（FC）标准来进行严苛评判。
                            请严格按以下格式输出反馈：
                            1. 【流利度与节奏】：评价朗读时的语速、停顿是否合理，有无不自然的卡顿、结巴或频繁的自我纠正。
                            2. 【发音准确度（错词/漏词）】：精准指出他严重读错、漏读或多读的具体单词。
                            3. 【语音语调（重音与连读）】：评价考生的意群断句（Chunking）、单词重音（Word Stress）和连读（Linking）是否自然。
                            4. 【考官提分建议】：给出一段犀利且实用的综合提升建议。
                            """
                            response = client_voice.models.generate_content(model='gemini-2.5-flash', contents=[audio_file, prompt])
                            st.success("🎉 发音诊断报告已生成！")
                            st.markdown(response.text)
                            st.balloons()
                            
                            supabase.table("reading_history").insert({
                                "username": current_user,
                                "reading_title": db_save_title,
                                "record_text": response.text
                            }).execute()
                            
                            st.session_state[last_audio_tracker_reading] = audio_bytes_reading
                            
                        except Exception as e:
                            st.error(f"Voice 引擎发生小意外：{e}")
                        os.remove(tmp_file_path)

                st.markdown("---")
                if st.button("🔄 感觉没读顺？清除录音，重读本句！", key=f"btn_reading_{db_save_title}_{st.session_state[reading_key_name]}"):
                    st.session_state[reading_key_name] += 1
                    st.rerun()
