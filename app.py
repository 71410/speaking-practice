import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
import tempfile
import os

# --- 1. 🔑 你的后台配置区 ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# 这是一个模拟的小型数据库，你可以在这里随便添加你朋友的账号和密码
USER_DATABASE = {
    "admin": "123456",       # 你的超级管理员账号
    "friend1": "ielts75",    # 朋友1的账号
    "friend2": "hello2026"   # 朋友2的账号
}

# --- 2. 🛡️ 登录状态检查器 ---
# 如果这是用户第一次打开网页，给他们发一个“未登录”的访客牌
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = ""

# --- 3. 🚪 登录界面 (未登录时显示) ---
if not st.session_state.logged_in:
    st.title("🔐 高分雅思模拟室 - 内部邀请版")
    st.write("这是一个私有练习空间，请输入您的专属账号和密码登录：")
    
    # 登录框
    username = st.text_input("👤 账号 (Username)")
    password = st.text_input("🔑 密码 (Password)", type="password") # type="password" 会让输入的字变成小黑点保护隐私
    
    if st.button("登录 (Login)"):
        # 校验账号密码是否和我们上面的小数据库对得上
        if username in USER_DATABASE and USER_DATABASE[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.success(f"登录成功！欢迎回来，{username}！正在为您准备专属考场...")
            st.rerun() # 刷新网页，放行进入主界面
        else:
            st.error("❌ 账号或密码错误，请检查后重试！")

# --- 4. 🎙️ 核心主界面 (只有登录成功后才会执行这里的代码) ---
else:
    # 侧边栏：显示当前用户，并提供退出功能
    st.sidebar.write(f"👤 当前练习者：**{st.session_state.current_user}**")
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

    # 以下是你原本的雅思模拟器代码
    st.title(f"欢迎来到专属考场, {st.session_state.current_user} 🎙️")
    st.write("今天的练习目标：直面考官，自信表达！深呼吸，准备好你的答案。")

    st.subheader("📺 Step 1: 观看情景视频")
    video_url = "https://www.w3schools.com/html/mov_bbb.mp4" 
    st.video(video_url)

    st.write("---")
    st.subheader("🗣️ Step 2: 轮到你了")
    st.write("点击下方麦克风开始作答，再点一次结束：")

    audio_bytes = audio_recorder(text="点击图标录音", icon_size="2x")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        st.success("🎉 录音成功！雅思考官正在审核你的回答...")
        
        st.write("---")
        st.subheader("🤖 Step 3: 考官权威点评")
        
        with st.spinner("🧠 考官正在仔细聆听并评估你的发音、词汇和语法..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
                
            try:
                audio_file = client.files.upload(file=tmp_file_path)
                
                # 让 AI 在点评时称呼当前用户的名字，更有专属感！
                prompt = f"""
                你现在是一名极其专业、严格的雅思口语考官。正在进行口语测试的考生叫 {st.session_state.current_user}。
                我已经上传了考生的口语回答录音。请你：
                1. 【精准听写】：把听到的英文原话一字不差地写下来。
                2. 【雅思预估分】：给出流利度、词汇多样性、语法准确性、发音的综合预估分数。
                3. 【纠错与升级】：严厉指出语法或用词错误，并给出 2 个 7.5 分以上的高阶地道示范。
                4. 【考官建议】：用中文给 {st.session_state.current_user} 一段简短、犀利的备考建议。
                """
                
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[audio_file, prompt]
                )
                
                st.markdown(response.text)
                st.balloons()
                
            except Exception as e:
                st.error(f"发生了一点小意外：{e}")
                
            os.remove(tmp_file_path)
