import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
import tempfile
import os

# --- 🔑 换上你的专属钥匙 ---
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

# 1. 网页标题变身
st.title("高分雅思口语模拟室 🎙️")
st.write("今天的练习目标：直面考官，自信表达！深呼吸，准备好你的答案。")

# 2. 视频区域大升级
st.subheader("📺 Step 1: 观看情景视频")
# 💡 【修改指南】：
# 如果你想放本地的视频，把下面的链接换成类似 "D:/我的视频/test.mp4" 的路径
# Streamlit 也完美支持 YouTube！你可以直接把 B站或 YouTube 的视频链接粘贴在这里
video_url = "https://www.w3schools.com/html/mov_bbb.mp4"
st.video(video_url)

# 3. 录音互动区域
st.write("---")
st.subheader("🗣️ Step 2: 轮到你了")
st.write("点击下方麦克风开始作答，再点一次结束：")

audio_bytes = audio_recorder(text="点击图标录音", icon_size="2x")

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    st.success("🎉 录音成功！雅思考官正在审核你的回答...")
    
    # ------------------ 全新进化的 AI 大脑与耳朵 ------------------
    st.write("---")
    st.subheader("🤖 Step 3: 考官权威点评")
    
    with st.spinner("🧠 考官正在仔细聆听并评估你的发音、词汇和语法..."):
        # 把声音存下来，准备直接发给 Gemini
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
            
        try:
            # 🚀 核心升级：直接把音频文件上传给 Gemini，准确度极高！
            audio_file = client.files.upload(file=tmp_file_path)
            
            # 🎭 核心升级：AI 角色切换为雅思口语考官
            prompt = """
            你现在是一名极其专业、严格的雅思口语考官。我已经上传了我的口语回答录音。
            请你帮我完成以下几件事：
            1. 【精准听写】：首先，把你听到的我的英文原话一字不差地写下来，即使有语法错误也要如实记录。
            2. 【雅思预估分】：根据雅思口语评分标准（流利度、词汇多样性、语法准确性、发音），给出一个综合预估分数。
            3. 【纠错与升级】：严厉指出我的语法、用词或中式英语思维错误，并给出 2 个能拿到雅思 7.5 分以上的高阶地道示范回答。
            4. 【考官建议】：用中文给我一段简短、犀利的备考建议。
            """
            
            # 让 Gemini 同时处理“音频”和“文字指令”
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[audio_file, prompt]
            )
            
            # 显示考官的最终报告
            st.markdown(response.text)
            st.balloons()
            
        except Exception as e:
            st.error(f"发生了一点小意外：{e}")
            
        # 打扫战场：清理临时音频文件
        os.remove(tmp_file_path)