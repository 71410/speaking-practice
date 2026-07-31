import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from google.genai import types as genai_types
from supabase import create_client, Client
import csv
import json
import re
import requests
from datetime import datetime, timedelta, timezone
from gtts import gTTS
import base64
import hashlib
import io
import time
from openai import OpenAI
import pypdf

# --- 1. 🔑 核心配置区 (中西合璧：DeepSeek + Gemini) ---
GEMINI_API_KEY_VOICE = st.secrets["GEMINI_API_KEY_VOICE"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
# 可选：配了才启用「客观测量」层，没配就退回纯 Gemini 评分。
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "")

# 引擎 A：负责后台苦力（DeepSeek 文本解析）
@st.cache_resource
def get_admin_client() -> OpenAI:
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

# 引擎 B：负责前台考官（Gemini 语音打分）
@st.cache_resource
def get_voice_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY_VOICE)

@st.cache_resource
def get_supabase_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

client_admin = get_admin_client()
client_voice = get_voice_client()
supabase: Client = get_supabase_client()
USER_DATABASE = st.secrets["passwords"]
GEMINI_FLASH_MODEL = "gemini-2.5-flash"
NAV_PAGES = [
    "🏠 训练台",
    "🗣️ 模拟考官",
    "🎤 我的素材朗读",
    "📖 英文原版朗读纠音",
    "✍️ 雅思写作练习",
    "👤 个人档案",
]

# release_inactive_audio 靠这些前缀识别「手动存进 session_state 的录音」。
# 新增录音模块时必须把前缀登记到这里，否则录音不会被回收。
AUDIO_STATE_PREFIXES = ("audio_qa_", "audio_reading_", "audio_material_")

# 录音器参数。
# sample_rate：默认 44.1kHz 的未压缩 WAV，2 分钟就有 10 MB，光上传就要等很久；
#   Gemini 处理语音本来就按 16kHz 来，降到 16kHz 只减体积不掉识别精度（约省 64%）。
RECORDER_SAMPLE_RATE = 16000
# pause_threshold：静音多少秒后自动停止录音。之前设成 60，意味着读完之后
#   还要干等整整一分钟录音才结束 —— 这是「反馈慢」的最大来源。
#   朗读时不会停顿 3 秒，所以朗读类用 3 秒；口语作答会边想边说，放宽到 8 秒。
PAUSE_THRESHOLD_READING = 3.0
PAUSE_THRESHOLD_SPEAKING = 8.0

# gTTS 用 tld 切换口音。
TTS_ACCENTS = {
    "🇬🇧 英式": "co.uk",
    "🇺🇸 美式": "com",
    "🇦🇺 澳式": "com.au",
}


SUPABASE_PAGE_SIZE = 1000


def request_page_change(page_name: str) -> None:
    st.session_state.requested_page = page_name
    st.rerun()


def fetch_all_rows(table: str, columns: str, order_column: str = "id") -> list:
    """分页拉取整张表。

    Supabase 项目默认 max-rows = 1000，直接 select() 超过上限会静默截断且不报错。
    这里按 range 逐页取，直到某一页不满为止。
    """
    rows: list = []
    start = 0
    while True:
        response = (
            supabase.table(table)
            .select(columns)
            .order(order_column)
            .range(start, start + SUPABASE_PAGE_SIZE - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < SUPABASE_PAGE_SIZE:
            return rows
        start += SUPABASE_PAGE_SIZE


@st.cache_data(ttl=300, show_spinner=False)
def count_rows(table: str, column: str | None = None, value=None) -> int:
    """只取行数，不拉正文。用于首页训练台的统计展示。"""
    query = supabase.table(table).select("id", count="exact").limit(1)
    if column is not None:
        query = query.eq(column, value)
    response = query.execute()
    return response.count or 0


BEIJING_TZ = timezone(timedelta(hours=8))


def format_record_time(raw) -> str:
    """把 Supabase 的 UTC 时间戳转成北京时间展示。"""
    if not raw:
        return ""
    try:
        moment = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return str(raw)[:16]
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")


def render_history_records(records: list, text_field: str) -> None:
    """统一渲染历史记录，最新一次在最前面。"""
    for index, record in enumerate(records):
        label = "最近一次" if index == 0 else f"往前第 {index} 次"
        st.markdown(f"**▶ {label}**　{format_record_time(record.get('created_at'))}")
        st.write(record[text_field])
        st.write("---")


def release_inactive_audio(active_audio_key: str) -> None:
    """只保留当前题目的录音。

    录音是手动写进 session_state 的（不是 widget state，Streamlit 不会自动回收），
    连续换题练习会让多段未压缩 WAV 常驻内存，最终撑爆 Streamlit Cloud 的内存配额。
    """
    for key in list(st.session_state.keys()):
        if not isinstance(key, str):
            continue
        if key.startswith(AUDIO_STATE_PREFIXES) and key != active_audio_key:
            st.session_state.pop(key, None)
            st.session_state.pop("audio_hash_" + key[len("audio_"):], None)


def split_sentences(text: str) -> list[str]:
    """按句末标点切句，用于逐句精读。切不出来时退回整段。"""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return sentences or [text.strip()]


def is_transient_gemini_error(error: Exception) -> bool:
    message = str(error).lower()
    if is_gemini_quota_error(error):
        return False
    transient_markers = [
        "503",
        "unavailable",
        "high demand",
        "temporar",
        "timeout",
        "deadline",
        "429",
        "resource_exhausted",
        "internal",
    ]
    return any(marker in message for marker in transient_markers)


def is_gemini_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    quota_markers = [
        "resource_exhausted",
        "exceeded your current quota",
        "quota",
        "billing",
    ]
    return any(marker in message for marker in quota_markers)


def gemini_config(thinking_budget: int | None):
    """thinking_budget=0 关闭思考；None 表示交给模型动态决定（默认行为）。

    实测：语音评分开着思考要烧 2800-4400 个思考 token（约 20 秒），
    而这些 token 完全不体现在报告里 —— 关掉之后快 3.9 倍（30.3s → 7.8s），
    输出反而更细（音标数 28 → 82）。所以除写作批改外一律关掉。
    """
    if thinking_budget is None:
        return None
    return genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_budget=thinking_budget)
    )


def generate_gemini_content_with_retry(
    contents: list,
    model: str = GEMINI_FLASH_MODEL,
    retry_delays: tuple[int, ...] = (1, 2, 4),
    thinking_budget: int | None = 0,
):
    config = gemini_config(thinking_budget)
    kwargs = {"config": config} if config else {}
    for attempt in range(len(retry_delays) + 1):
        try:
            return client_voice.models.generate_content(
                model=model, contents=contents, **kwargs
            )
        except Exception as exc:
            if attempt >= len(retry_delays) or not is_transient_gemini_error(exc):
                raise
            time.sleep(retry_delays[attempt])


def stream_gemini_content(
    contents: list,
    model: str = GEMINI_FLASH_MODEL,
    retry_delays: tuple[int, ...] = (1, 2, 4),
    thinking_budget: int | None = 0,
):
    """流式产出评分文本，让用户 1-2 秒就能看到内容，而不是干等一整轮。

    只有「一个字都还没吐出来」时才重试；已经开始输出后再失败就直接抛，
    否则重试会把前半段内容重复打印一遍。
    """
    config = gemini_config(thinking_budget)
    kwargs = {"config": config} if config else {}
    for attempt in range(len(retry_delays) + 1):
        started = False
        try:
            for chunk in client_voice.models.generate_content_stream(
                model=model, contents=contents, **kwargs
            ):
                if chunk.text:
                    started = True
                    yield chunk.text
            return
        except Exception as exc:
            if started or attempt >= len(retry_delays) or not is_transient_gemini_error(exc):
                raise
            time.sleep(retry_delays[attempt])


def show_gemini_busy_error(error: Exception) -> None:
    if is_gemini_quota_error(error):
        st.error("Gemini API 配额或限流已触发。录音已保留，处理额度/账单后可以重新提交评分。")
    elif is_transient_gemini_error(error):
        st.error("Gemini 当前繁忙，自动重试后仍未成功。录音已保留，可以稍后直接重新请求评分。")
    else:
        st.error("Gemini 请求失败。录音已保留，可以稍后直接重新请求评分。")
    st.caption(f"服务返回：{str(error)[:240]}")


def wav_audio_part(audio_bytes: bytes):
    return genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")


def audio_fingerprint(audio_bytes: bytes) -> str:
    return hashlib.md5(audio_bytes).hexdigest()


def show_audio_payload_info(audio_bytes: bytes) -> None:
    size_mb = len(audio_bytes) / 1024 / 1024
    st.caption(f"录音大小：{size_mb:.2f} MB。2 分钟内通常可以提交；如果超过 15 MB，建议分段练习。")
    if size_mb > 15:
        st.warning("这段录音较大，评分可能明显变慢或失败。建议缩短录音或分成多段提交。")


def submit_audio_for_scoring(pending_key: str) -> None:
    st.session_state[pending_key] = True
    st.session_state[f"{pending_key}_started_at"] = time.strftime("%H:%M:%S")
    st.rerun()


def render_recording_practice(
    *,
    scope_id: str,
    audio_prefix: str,
    build_prompt,
    save_history,
    reference_text: str | None = None,
    recorder_text: str = "点击麦克风开始录音",
    submit_label: str = "📤 提交本次录音评分",
    reset_label: str = "🔄 清除录音，再录一次",
    success_message: str = "🎉 评分完成！",
    celebrate: bool = False,
    pause_threshold: float = PAUSE_THRESHOLD_READING,
) -> None:
    """录音 → 提交 → Gemini 评分 → 存档 的完整流程。

    scope_id 决定这一组 session_state 的命名空间（换素材/换句子就换一组）。
    评分结果会写进 session_state，这样后续 rerun 时报告不会消失。
    """
    counter_key = f"counter_{audio_prefix}_{scope_id}"
    if counter_key not in st.session_state:
        st.session_state[counter_key] = 0
    gen = st.session_state[counter_key]

    audio_bytes = audio_recorder(
        text=recorder_text,
        icon_size="2x",
        pause_threshold=pause_threshold,
        sample_rate=RECORDER_SAMPLE_RATE,
        key=f"recorder_{audio_prefix}_{scope_id}_{gen}",
    )
    st.caption(
        f"读完点一下麦克风就能立即结束；不点的话，静音满 {pause_threshold:.0f} 秒会自动停止。"
    )

    audio_key = f"audio_{audio_prefix}_{scope_id}"
    hash_key = f"audio_hash_{audio_prefix}_{scope_id}"
    release_inactive_audio(audio_key)
    if audio_bytes:
        st.session_state[audio_key] = audio_bytes
        st.session_state[hash_key] = audio_fingerprint(audio_bytes)

    saved_audio = st.session_state.get(audio_key)
    saved_hash = st.session_state.get(hash_key, "")
    if not saved_audio:
        return

    st.audio(saved_audio, format="audio/wav")
    show_audio_payload_info(saved_audio)

    tracker_key = f"last_audio_{audio_prefix}_{scope_id}"
    pending_key = f"pending_{audio_prefix}_{scope_id}"
    started_key = f"{pending_key}_started_at"
    report_key = f"report_{audio_prefix}_{scope_id}"
    already_scored = st.session_state.get(tracker_key) == saved_hash

    if already_scored:
        stored_report = st.session_state.get(report_key, "")
        if stored_report:
            st.markdown(stored_report)
        else:
            st.info("本次录音已完成评分。需要重新评分请先清除录音后再录一次。")
    elif st.session_state.get(pending_key):
        status_box = st.status("正在评分，报告会边生成边显示", expanded=True)
        status_box.write(f"已收到提交（{st.session_state.get(started_key, '刚刚')}）")
        try:
            # 第一步：客观测量。有原文 + 配了 NVIDIA key 才做。
            measurements = None
            if reference_text and NVIDIA_API_KEY:
                status_box.write("正在做逐词比对（NVIDIA Parakeet ASR）……")
                try:
                    measurements = measure_reading(
                        reference_text, transcribe_with_nvidia(saved_audio)
                    )
                except Exception as asr_error:
                    # 测量失败不该挡住评分，退回纯 LLM 评分即可。
                    status_box.write("逐词比对失败，本次只给 AI 主观点评。")
                    st.info(f"客观测量这一步没跑通（{asr_error}），下面的报告是纯 AI 判断。")

            if measurements:
                st.markdown("#### 📐 客观测量（程序算出来的，不是 AI 听感）")
                render_measurements(measurements)
                st.markdown("#### 🎧 考官点评")

            status_box.write("正在生成点评……")
            # 流式输出：1-2 秒就开始出字，不用盯着空白等一整轮。
            report = st.write_stream(
                stream_gemini_content(
                    contents=[wav_audio_part(saved_audio), build_prompt(measurements)]
                )
            )
            status_box.update(label="评分完成，正在归档", state="running")
            st.success(success_message)

            # 存档失败不应该让用户丢掉已经拿到的评分，所以单独捕获。
            try:
                save_history(report)
            except Exception as save_error:
                st.warning(f"评分已生成，但历史记录没能存进数据库：{save_error}")

            st.session_state[report_key] = report
            st.session_state[tracker_key] = saved_hash
            st.session_state[pending_key] = False
            st.session_state.pop(started_key, None)
            status_box.update(label="本次评分完成", state="complete")
            if celebrate:
                st.balloons()
        except Exception as e:
            st.session_state[pending_key] = False
            st.session_state.pop(started_key, None)
            status_box.update(label="本次评分未完成", state="error")
            show_gemini_busy_error(e)
            if st.button("重新提交本次录音评分", key=f"retry_{audio_prefix}_{scope_id}_{gen}"):
                submit_audio_for_scoring(pending_key)
    else:
        st.success("录音已保存。确认无误后点击下方按钮提交评分。")
        if st.button(
            submit_label,
            type="primary",
            key=f"submit_{audio_prefix}_{scope_id}_{gen}",
        ):
            submit_audio_for_scoring(pending_key)

    st.markdown("---")
    if st.button(reset_label, key=f"reset_{audio_prefix}_{scope_id}_{gen}"):
        for key in (audio_key, hash_key, tracker_key, pending_key, started_key, report_key):
            st.session_state.pop(key, None)
        st.session_state[counter_key] += 1
        st.rerun()


def render_personalized_answer(question: str, username: str, answer_key: str, button_key: str) -> None:
    if st.session_state.get(answer_key):
        st.markdown("---")
        st.subheader("✨ 你的专属个性化答案")
        st.markdown(st.session_state[answer_key])
        return

    if st.button("✨ 生成专属参考答案", key=button_key):
        profile_info = load_profile_information(username)
        with st.spinner("正在根据你的个人档案定制专属答案..."):
            personalized_answer = generate_personalized_answer(question, profile_info)
        st.session_state[answer_key] = personalized_answer
        st.markdown("---")
        st.subheader("✨ 你的专属个性化答案")
        if not profile_info.strip():
            st.caption(
                "你尚未填写个人档案，答案为通用示范。前往左侧「个人档案」填写后，答案将更贴合你的真实人设。"
            )
        st.markdown(personalized_answer)


@st.cache_data(ttl=60, show_spinner=False)
def load_profile_information(username: str) -> str:
    response = (
        supabase.table("profiles")
        .select("information")
        .eq("username", username)
        .execute()
    )
    if response.data and response.data[0].get("information"):
        return response.data[0]["information"]
    return ""


def save_profile_information(username: str, information: str) -> None:
    existing = (
        supabase.table("profiles")
        .select("username")
        .eq("username", username)
        .execute()
    )
    if existing.data:
        supabase.table("profiles").update({"information": information}).eq(
            "username", username
        ).execute()
    else:
        supabase.table("profiles").insert(
            {"username": username, "information": information}
        ).execute()
    load_profile_information.clear()


def generate_personalized_answer(question: str, profile_info: str) -> str:
    profile_block = profile_info.strip() if profile_info.strip() else "（用户尚未填写个人档案）"
    prompt = f"""
你是一名雅思口语教练。请根据考生的个人档案，为以下题目生成一段「专属参考答案」。

【题目】：{question}

【考生个人档案】：
{profile_block}

要求：
1. 答案必须 100% 贴合该考生的人设、背景、专业与爱好，像他自己会说的真实经历与观点。
2. 使用口语化、自然的英文表达，像真实人类在雅思考场里即兴说话，避免书面腔和模板句。
3. 长度符合该题型的正常作答时长（Part 1 约 3-4 句，Part 2 需有清晰结构，Part 3 可稍展开）。
4. 只输出英文答案正文，不要标题、不要 markdown、不要中文解释。
"""
    response = client_admin.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You write natural, personalized IELTS speaking sample answers in spoken English only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text.strip()))


def writing_fingerprint(task_id: int, essay: str) -> str:
    normalized = " ".join(essay.strip().split())
    return hashlib.md5(f"{task_id}:{normalized}".encode("utf-8")).hexdigest()


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = "快速结论"
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines or not sections:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines).strip()) for title, lines in sections if "\n".join(lines).strip()]


def render_writing_evaluation_result(evaluation: str) -> None:
    """渲染批改结果。

    这里刻意不开 unsafe_allow_html：内容是模型直接产出的，没有必要当 HTML 执行；
    批改报告里唯一需要的富格式是 markdown 表格，st.markdown 原生就支持。
    """
    st.subheader("✅ 批改结果")
    sections = split_markdown_sections(evaluation)

    if len(sections) <= 1:
        st.markdown(evaluation)
        return

    for index, (title, body) in enumerate(sections):
        expanded = index == 0 or "总评" in title or "总分" in title or "Overall" in title
        with st.expander(title, expanded=expanded):
            st.markdown(body)


def render_training_dashboard(current_user: str) -> None:
    st.subheader("今天想练什么？")
    st.caption("从一个模块开始就好。训练过程中录音、作文和批改结果都会尽量保留在当前会话里。")

    # 首页只展示统计数字，用 count 查询即可，不必把整个题库和朗读原文拉下来。
    question_count = count_rows("question_bank")
    reading_count = count_rows("reading_bank")
    task1_count = count_rows("writing_bank", "task_type", "Task 1")
    task2_count = count_rows("writing_bank", "task_type", "Task 2")
    profile_filled = bool(load_profile_information(current_user).strip())
    # 素材库依赖新表，没建好之前不能让首页整块崩掉。
    material_count = (
        count_rows("speaking_materials", "username", current_user)
        if speaking_tables_ready()
        else None
    )

    col_speaking, col_material, col_reading, col_writing = st.columns(4)

    with col_speaking:
        st.markdown("#### 🗣️ 模拟考官")
        st.caption(f"题库：{question_count} 道题")
        if st.button("开始口语训练", type="primary", width="stretch"):
            request_page_change("🗣️ 模拟考官")

    with col_material:
        st.markdown("#### 🎤 我的素材")
        st.caption(
            f"我的素材：{material_count} 条" if material_count is not None else "素材库待初始化"
        )
        if st.button("练自己的素材", type="primary", width="stretch"):
            request_page_change("🎤 我的素材朗读")

    with col_reading:
        st.markdown("#### 📖 朗读纠音")
        st.caption(f"材料：{reading_count} 篇")
        if st.button("开始朗读训练", type="primary", width="stretch"):
            request_page_change("📖 英文原版朗读纠音")

    with col_writing:
        st.markdown("#### ✍️ 写作批改")
        st.caption(f"Task 1：{task1_count} 题｜Task 2：{task2_count} 题")
        if st.button("开始写作练习", type="primary", width="stretch"):
            request_page_change("✍️ 雅思写作练习")

    st.markdown("---")
    profile_status = "已填写" if profile_filled else "未填写"
    st.info(f"个人档案：{profile_status}。档案会用于生成更贴近你自己的口语参考答案。")
    if st.button("查看或修改个人档案"):
        request_page_change("👤 个人档案")


@st.cache_data(ttl=86400, show_spinner=False)
def synthesize_tts_audio(text: str, tld: str = "co.uk", slow: bool = False) -> bytes:
    sound_file = io.BytesIO()
    tts = gTTS(text=text, lang="en", tld=tld, slow=slow)
    tts.write_to_fp(sound_file)
    return sound_file.getvalue()


def render_tts_demo(target_text: str, key_prefix: str) -> None:
    """示范朗读：可选口音与语速。音频按 (文本, 口音, 语速) 缓存 24 小时。"""
    col_accent, col_speed = st.columns([2, 1])
    with col_accent:
        accent_label = st.selectbox(
            "🔊 示范口音：",
            list(TTS_ACCENTS.keys()),
            key=f"{key_prefix}_accent",
        )
    with col_speed:
        slow = st.checkbox("🐢 慢速", key=f"{key_prefix}_slow")

    if st.button("🎧 听示范朗读", key=f"{key_prefix}_play"):
        with st.spinner("正在生成示范朗读..."):
            try:
                audio = synthesize_tts_audio(target_text, TTS_ACCENTS[accent_label], slow)
            except Exception as e:
                st.error(f"示范朗读生成失败（gTTS 需要联网）：{e}")
                return
        st.audio(audio, format="audio/mp3", autoplay=True)


def base64_to_image_bytes(b64_str: str) -> bytes | None:
    if not b64_str or not str(b64_str).strip():
        return None
    raw = str(b64_str).strip()
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw)


def image_mime_type(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


WRITING_IMAGE_EXTRACT_PROMPT = """
这是一张雅思写作考试的题目图片。请分析图片并严格输出一个 JSON 对象（不要 markdown 代码块、不要任何中文解释或寒暄）。

JSON 必须且仅包含以下两个字段：
1. "topic": 极其简短的英文作文主题（例如 "Location of dance classes"），不超过 12 个英文单词。
2. "instructions": 纯粹的题目描述与写作要求全文（例如题干 "The charts below give information..." 以及 "Summarise the information by selecting and reporting the main features..." 等）。

【严禁包含】图表内部的具体数据点、数字、百分比、坐标轴标签、刻度、图例分类名称、表格单元格数值等图表细节。
只保留考生需要阅读的「题目说明」和「写作指令」。

输出示例格式：
{"topic": "Population change in three countries", "instructions": "The chart below shows... Write at least 150 words."}
"""


def parse_writing_extract_json(raw_text: str) -> dict:
    raw = raw_text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    data = json.loads(raw.strip())
    topic = str(data.get("topic", "")).strip()
    instructions = str(data.get("instructions", "")).strip()
    if not topic or not instructions:
        raise ValueError("AI 返回的 JSON 缺少 topic 或 instructions 字段。")
    return {"topic": topic, "instructions": instructions}


def extract_writing_prompt_from_image(image_bytes: bytes) -> dict:
    response = generate_gemini_content_with_retry(
        contents=[
            genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type=image_mime_type(image_bytes),
            ),
            WRITING_IMAGE_EXTRACT_PROMPT,
        ],
    )
    return parse_writing_extract_json(response.text)


def init_admin_writing_session_state() -> None:
    defaults = {
        "admin_writing_fingerprint": "",
        "admin_writing_image_b64": "",
        "admin_writing_ready": False,
        "admin_writing_topic_draft": "",
        "admin_writing_content_draft": "",
        "admin_writing_form_gen": 0,
        "admin_writing_uploader_gen": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_admin_writing_session_state() -> None:
    """只清理数据层状态；通过递增 form_gen 废弃旧 widget，不直接改写 widget 的 session_state。"""
    st.session_state.admin_writing_fingerprint = ""
    st.session_state.admin_writing_image_b64 = ""
    st.session_state.admin_writing_ready = False
    st.session_state.admin_writing_topic_draft = ""
    st.session_state.admin_writing_content_draft = ""
    st.session_state.admin_writing_form_gen += 1


def admin_writing_topic_widget_key() -> str:
    return f"admin_writing_topic_widget_{st.session_state.admin_writing_form_gen}"


def admin_writing_content_widget_key() -> str:
    return f"admin_writing_content_widget_{st.session_state.admin_writing_form_gen}"


def get_admin_writing_topic() -> str:
    return st.session_state.get(
        admin_writing_topic_widget_key(),
        st.session_state.get("admin_writing_topic_draft", ""),
    ).strip()


def get_admin_writing_content() -> str:
    return st.session_state.get(
        admin_writing_content_widget_key(),
        st.session_state.get("admin_writing_content_draft", ""),
    ).strip()


@st.cache_data(ttl=300, show_spinner=False)
def load_writing_tasks(task_type: str) -> list:
    """只取题目列表用的轻字段，question_image 留给 load_writing_task_detail 按需拉。"""
    response = (
        supabase.table("writing_bank")
        .select("id, task_type, title, content")
        .eq("task_type", task_type)
        .order("id")
        .range(0, SUPABASE_PAGE_SIZE - 1)
        .execute()
    )
    return response.data or []


@st.cache_data(ttl=300, show_spinner=False)
def load_writing_task_detail(task_id: int) -> dict:
    response = (
        supabase.table("writing_bank")
        .select("id, task_type, title, content, question_image")
        .eq("id", task_id)
        .limit(1)
        .execute()
    )
    if response.data:
        return response.data[0]
    return {}


def clear_writing_task_caches() -> None:
    load_writing_tasks.clear()
    load_writing_task_detail.clear()
    count_rows.clear()


def clear_question_bank_caches() -> None:
    load_question_bank.clear()
    count_rows.clear()


def clear_reading_bank_caches() -> None:
    load_reading_bank.clear()
    count_rows.clear()


# ---------- 个人素材库（每个用户自己的朗读素材） ----------

SPEAKING_MATERIALS_SQL = """create table if not exists speaking_materials (
  id          bigint generated always as identity primary key,
  username    text        not null,
  title       text        not null,
  content     text        not null,
  created_at  timestamptz not null default now()
);
create index if not exists speaking_materials_user_idx
  on speaking_materials (username, id desc);

create table if not exists material_history (
  id              bigint generated always as identity primary key,
  username        text        not null,
  material_title  text        not null,
  record_text     text        not null,
  created_at      timestamptz not null default now()
);
create index if not exists material_history_lookup_idx
  on material_history (username, material_title, created_at desc);"""


@st.cache_data(ttl=60, show_spinner=False)
def speaking_tables_ready() -> bool:
    """素材库依赖两张新表；没建好时页面要给出可执行的建表 SQL 而不是直接崩。"""
    try:
        supabase.table("speaking_materials").select("id").limit(1).execute()
        supabase.table("material_history").select("id").limit(1).execute()
        return True
    except Exception:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def load_speaking_materials(username: str) -> list:
    """加载某用户的全部素材，最新添加的排在前面。"""
    response = (
        supabase.table("speaking_materials")
        .select("id, title, content, created_at")
        .eq("username", username)
        .order("id", desc=True)
        .range(0, SUPABASE_PAGE_SIZE - 1)
        .execute()
    )
    return response.data or []


def save_speaking_material(username: str, title: str, content: str) -> None:
    supabase.table("speaking_materials").insert(
        {"username": username, "title": title, "content": content}
    ).execute()
    load_speaking_materials.clear()
    count_rows.clear()


def delete_speaking_material(username: str, material_id: int) -> None:
    # 带上 username 条件，避免误删到别人的素材。
    supabase.table("speaking_materials").delete().eq("id", material_id).eq(
        "username", username
    ).execute()
    load_speaking_materials.clear()
    count_rows.clear()


@st.cache_data(ttl=30, show_spinner=False)
def load_material_history(username: str, material_title: str) -> list:
    response = (
        supabase.table("material_history")
        .select("record_text, created_at")
        .eq("username", username)
        .eq("material_title", material_title)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return response.data or []


def save_material_history(username: str, material_title: str, record_text: str) -> None:
    supabase.table("material_history").insert(
        {
            "username": username,
            "material_title": material_title,
            "record_text": record_text,
        }
    ).execute()
    load_material_history.clear()


DEEPSEEK_EXTRACT_SYSTEM_PROMPT = (
    "You are a precise JSON data extraction tool. "
    "Output strictly valid JSON arrays without markdown syntax."
)

# 数据库列 -> (来源字段名, 缺失时的兜底值)，CSV 与 PDF 两条导入路径共用。
QUESTION_BANK_FIELDS = {
    "part": ("part", "未分类"),
    "theme": ("theme", "未分类"),
    "question_text": ("question", "提取失败"),
}
READING_BANK_FIELDS = {
    "title": ("title", "未命名文章"),
    "content": ("content", "内容提取失败"),
}


def strip_json_code_fence(raw_text: str) -> str:
    raw = raw_text.strip()
    if raw.startswith("```json"):
        raw = raw[7:]
    if raw.startswith("```"):
        raw = raw[3:]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


def read_pdf_text(uploaded_file, limit: int = 30000) -> str:
    reader = pypdf.PdfReader(uploaded_file)
    pages = [(page.extract_text() or "") for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("这个 PDF 没有可提取的文字层（可能是扫描件）。")
    return text[:limit]


def map_records(records: list, field_map: dict) -> list:
    return [
        {
            column: str(record.get(source_key) or fallback).strip()
            for column, (source_key, fallback) in field_map.items()
        }
        for record in records
    ]


def read_csv_records(uploaded_file, field_map: dict) -> list:
    text = uploaded_file.getvalue().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    missing = [source_key for source_key, _ in field_map.values() if source_key not in columns]
    if missing:
        raise ValueError(
            f"CSV 缺少必需的列：{'、'.join(missing)}；文件里实际的列是：{'、'.join(columns) or '（空）'}"
        )
    return map_records(list(reader), field_map)


def extract_records_with_deepseek(pdf_text: str, instruction: str, field_map: dict) -> list:
    prompt = f"""{instruction}
绝对不要输出任何 markdown 标记、不要废话，只输出纯文本 JSON 数组。

【源文本】:
{pdf_text}
"""
    response = client_admin.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": DEEPSEEK_EXTRACT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=8192,
    )
    extracted = json.loads(strip_json_code_fence(response.choices[0].message.content))
    if not isinstance(extracted, list):
        raise ValueError("AI 没有返回 JSON 数组。")
    return map_records(extracted, field_map)


def import_bank_file(
    uploaded_file,
    table: str,
    field_map: dict,
    pdf_instruction: str,
    label: str,
    clear_cache,
) -> None:
    """CSV / PDF 两种来源统一走这条导入通道（口语题库与阅读文章库共用）。"""
    filename = uploaded_file.name.lower()
    try:
        if filename.endswith(".csv"):
            with st.spinner(f"正在解析 {label} CSV..."):
                rows = read_csv_records(uploaded_file, field_map)
        elif filename.endswith(".pdf"):
            with st.spinner("🤖 正在召唤 DeepSeek 大脑提取内容..."):
                rows = extract_records_with_deepseek(
                    read_pdf_text(uploaded_file), pdf_instruction, field_map
                )
        else:
            st.sidebar.error("只支持 CSV 或 PDF 文件。")
            return
    except Exception as e:
        st.sidebar.error(f"解析失败：{e}")
        return

    if not rows:
        st.sidebar.warning("没有解析出任何内容，请检查文件格式。")
        return

    try:
        with st.spinner("正在写入数据库..."):
            supabase.table(table).insert(rows).execute()
    except Exception as e:
        st.sidebar.error(f"写入数据库失败：{e}")
        return

    clear_cache()
    st.sidebar.success(f"✅ 成功导入 {len(rows)} 条{label}！")


@st.cache_data(ttl=300, show_spinner=False)
def load_question_bank() -> dict:
    """加载完整口语题库，缓存 5 分钟。管理员上传新题后自动刷新。"""
    rows = fetch_all_rows("question_bank", "part, theme, question_text")
    bank: dict = {}
    for row in rows:
        p = row.get("part", "未分类")
        t = row.get("theme", "未分类")
        q = row.get("question_text", "提取失败")
        if p not in bank:
            bank[p] = {}
        if t not in bank[p]:
            bank[p][t] = []
        bank[p][t].append(q)
    return bank


@st.cache_data(ttl=300, show_spinner=False)
def load_reading_bank() -> dict:
    """加载完整阅读材料库，缓存 5 分钟。"""
    rows = fetch_all_rows("reading_bank", "title, content")
    return {row["title"]: row["content"] for row in rows}


@st.cache_data(ttl=30, show_spinner=False)
def load_practice_history(username: str, question: str) -> list:
    """加载某用户某道题最近 5 次练习记录，最新的排在前面，缓存 30 秒。"""
    response = (
        supabase.table("practice_history")
        .select("record_text, created_at")
        .eq("username", username)
        .eq("question", question)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return response.data or []


@st.cache_data(ttl=30, show_spinner=False)
def load_reading_history(username: str, reading_title: str) -> list:
    """加载某用户某材料最近 5 次朗读记录，最新的排在前面，缓存 30 秒。"""
    response = (
        supabase.table("reading_history")
        .select("record_text, created_at")
        .eq("username", username)
        .eq("reading_title", reading_title)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return response.data or []


@st.cache_data(ttl=30, show_spinner=False)
def load_writing_history(username: str, task_id: int) -> list:
    """加载某用户某写作题的历史批改，缓存 30 秒。"""
    response = (
        supabase.table("writing_history")
        .select("evaluation, created_at")
        .eq("username", username)
        .eq("task_id", task_id)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    return response.data or []


# ---------- 客观测量层（NVIDIA Parakeet ASR）----------
#
# 为什么需要这层：LLM 只是「听个大概然后写出一段听起来专业的文字」。实测把
# 与原文完全一致的音频喂给 Gemini，它照样列出 6 处「错误」，而且「你读成」和
# 「正确音标」两列是同一个字符串 —— 纯属为了填满表格而编。
#
# 我们这里有原文，所以漏读/多读/替换、语速、停顿这几项完全可以「算」出来。
# 算出来的硬数字再交给 LLM 去组织语言，模型就没有空间瞎编了。
#
# 注意：该 REST 端点只返回 word/start/end，没有 confidence，所以
# 「某个词发音含糊」这类判断仍然只能由 LLM 给，属于主观部分。
NVIDIA_ASR_URL = (
    "https://1598d209-5e27-4d3c-8079-4751568b1081.invocation.api.nvcf.nvidia.com"
    "/v1/audio/transcriptions"
)
PAUSE_GAP_MS = 500  # 相邻词间隔超过这个值算一次明显停顿


def transcribe_with_nvidia(wav_bytes: bytes, timeout: int = 90) -> dict:
    """调用 Parakeet CTC 1.1B，返回 {'text': str, 'words': [{word,start,end}]}。"""
    response = requests.post(
        NVIDIA_ASR_URL,
        headers={"Authorization": f"Bearer {NVIDIA_API_KEY}"},
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={"language": "en-US", "word_time_offsets": "True"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def normalize_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w]


def measure_reading(reference: str, asr: dict) -> dict:
    """把 ASR 结果和原文比对，产出可验证的客观指标。"""
    import difflib

    ref_words = normalize_words(reference)
    heard_words = normalize_words(asr.get("text", ""))
    timed = asr.get("words") or []

    substitutions: list[tuple[str, str]] = []
    omissions: list[str] = []
    insertions: list[str] = []
    matched = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        None, ref_words, heard_words
    ).get_opcodes():
        if tag == "equal":
            matched += i2 - i1
        elif tag == "replace":
            substitutions.append((" ".join(ref_words[i1:i2]), " ".join(heard_words[j1:j2])))
        elif tag == "delete":
            omissions.append(" ".join(ref_words[i1:i2]))
        elif tag == "insert":
            insertions.append(" ".join(heard_words[j1:j2]))

    # 时间戳单位是毫秒
    speech_ms = (timed[-1]["end"] - timed[0]["start"]) if len(timed) >= 2 else 0
    wpm = round(len(timed) / (speech_ms / 60000), 1) if speech_ms > 0 else None

    pauses = []
    for prev, nxt in zip(timed, timed[1:]):
        gap = nxt["start"] - prev["end"]
        if gap >= PAUSE_GAP_MS:
            pauses.append({"after": prev["word"], "before": nxt["word"], "ms": gap})

    return {
        "heard_text": asr.get("text", "").strip(),
        "reference_word_count": len(ref_words),
        "coverage": round(100 * matched / len(ref_words), 1) if ref_words else 0.0,
        "substitutions": substitutions,
        "omissions": omissions,
        "insertions": insertions,
        "wpm": wpm,
        "duration_s": round(speech_ms / 1000, 1) if speech_ms else None,
        "pauses": pauses,
    }


def format_measurements_for_prompt(m: dict) -> str:
    """把客观指标写成一段文本，注入 Prompt，作为 LLM 不许违背的事实。"""
    lines = [
        f"- ASR 实际转写：{m['heard_text']}",
        f"- 原文词数 {m['reference_word_count']}，正确读出比例 {m['coverage']}%",
        f"- 语速：{m['wpm']} 词/分钟（有效时长 {m['duration_s']} 秒）"
        if m["wpm"] else "- 语速：无法测量",
    ]
    lines.append(
        "- 读错的词：" + ("；".join(f"原文「{a}」读成了「{b}」" for a, b in m["substitutions"])
                        if m["substitutions"] else "无")
    )
    lines.append("- 漏读：" + ("；".join(m["omissions"]) if m["omissions"] else "无"))
    lines.append("- 多读：" + ("；".join(m["insertions"]) if m["insertions"] else "无"))
    if m["pauses"]:
        lines.append("- 明显停顿：" + "；".join(
            f"「{p['after']}」之后停了 {p['ms']/1000:.1f} 秒" for p in m["pauses"][:8]))
    else:
        lines.append("- 明显停顿：无（没有超过 0.5 秒的间隔）")
    return "\n".join(lines)


def render_measurements(m: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("正确读出", f"{m['coverage']}%")
    cols[1].metric("语速", f"{m['wpm']} 词/分" if m["wpm"] else "—")
    cols[2].metric("明显停顿", f"{len(m['pauses'])} 处")

    if m["substitutions"] or m["omissions"] or m["insertions"]:
        rows = []
        for a, b in m["substitutions"]:
            rows.append(f"| 读错 | `{a}` | `{b}` |")
        for w in m["omissions"]:
            rows.append(f"| 漏读 | `{w}` | — |")
        for w in m["insertions"]:
            rows.append(f"| 多读 | — | `{w}` |")
        st.markdown(
            "| 类型 | 原文 | 你读的 |\n|---|---|---|\n" + "\n".join(rows)
        )
    else:
        st.success("逐词比对：没有读错、漏读或多读。")

    with st.expander("查看 ASR 转写原文"):
        st.write(m["heard_text"])


def build_material_scoring_prompt(target_text: str, measurements: dict | None = None) -> str:
    measured_block = ""
    if measurements:
        measured_block = f"""
【已测量的客观事实 —— 这些是程序算出来的，不是听感，你必须以此为准】
{format_measurements_for_prompt(measurements)}

使用规则：
- 上面的语速、漏读、多读、读错词已经确定，禁止推翻，也禁止再自行「听出」别的漏读或读错。
- 如果上面写了「无」，就不要在报告里编造该类问题。
- 你的价值在于上面测不出来的部分：具体音素发得准不准、重音位置、语调、连读弱读。
- 停顿是否合理，请结合上面给出的真实停顿位置来评价。
"""

    return f"""你是一名雅思口语考官，本次只评估 Pronunciation（发音）这一单项。考生正在朗读下面这段指定文本，我已上传他的录音。

【指定文本】
{target_text}
{measured_block}
请按雅思 Pronunciation 标准打分。所有分数用 0-9 分，允许 0.5 分档。

严格按以下结构输出：

## 🎯 综合发音得分
**X.X / 9**
一句话说明这个分数对应雅思 Pronunciation 单项的什么水平。

## 1️⃣ 发音准确度
**分数：X.X**
用表格逐个列出读错的词（至少列出所有明显错误，没有就写「无」）：

| 原文单词 | 你读成 | 正确音标 | 怎么改 |
|---|---|---|---|

再补充：
- 漏读 / 多读 / 替换掉的词
- 有没有系统性的音素问题（例如 /θ/ 读成 /s/、词尾辅音吞掉、长短元音不分、/l/ 与 /n/ 混淆），有就单独点名

## 2️⃣ 流利度与节奏
**分数：X.X**
- 估算语速（约 X 词/分钟），并说明相对朗读语速是偏快还是偏慢
- 停顿是否落在意群边界上；列出停错位置的具体词
- 指出卡顿、重复、自我纠正分别出现在哪几个词附近

## 3️⃣ 重音与语调
**分数：X.X**
- 单词重音（Word Stress）错误：列出单词并用大写标出正确重音位置，例如 `PHOtograph` → `phoTOgrapher`
- 句子重音（Sentence Stress）：该被强调的实词有没有被弱读掉
- 语调（升调/降调）用得是否恰当
- 连读（Linking）和弱读（Weak Form）有没有做出来；举出原文中本该连读却断开的位置

## 🔁 针对性练习
给 3 条马上能做的练习，每条必须对应上面指出的一个具体问题，不要泛泛而谈。

## 📌 下次朗读只改这一点
一句话，只说最该改的那一个点。

要求：
- 不要纠结英式还是美式口音，只要前后一致且清晰即可。
- 所有评价必须引用录音里的具体单词作为证据，严禁空泛表扬。
- 如果录音听不清或几乎没有人声，直接说明情况并给出重录建议，不要编造评分。"""


def evaluate_writing_task1_gemini(
    topic: str,
    instructions: str,
    question_image_b64: str | None,
    user_essay: str,
) -> str:
    prompt = f"""
你是一名资深雅思写作考官（British Council 标准）。请对以下 Task 1 作文进行严格、专业的多维度批改。

【题目主题】：{topic}
【题目要求】：
{instructions}

【考生作文】：
{user_essay}

请结合题目配图（如有）严格按以下结构输出（中英文对照，条理清晰）：

## 📊 预估总得分 (Overall Band Score)
给出 0.5 为单位的 Band 分数（如 6.5），并简要说明理由。

## 1️⃣ Task Achievement (TA) / 写作任务回应情况
- 英文点评 + 中文解读
- 是否准确描述图表/地图/流程的关键趋势、极值、对比，有无遗漏或臆造数据

## 2️⃣ Coherence and Cohesion (CC) / 连贯与衔接
- 段落结构、逻辑推进、连接词使用是否自然恰当
- 指出具体问题并给出改进建议

## 3️⃣ Lexical Resource (LR) / 词汇资源
- 列出文中 3-5 处明显的低级/重复词汇，给出高级替换建议（原词 → 升级词）
- 评价词汇多样性与搭配准确性

## 4️⃣ Grammatical Range and Accuracy (GRA) / 语法多样性与准确性
- 揪出语法错误（时态、主谓一致、冠词、从句等），给出正确写法
- 评价句式多样性

## 📝 逐句语法修改润色对照表
用表格形式列出（Markdown 表格）：
| 原句 | 修改后 | 修改说明（中文）|
至少覆盖 5 处最值得修改的句子（若错误不足 5 处则全部列出）。

## 💡 全面提分建议
用中文给出 3-5 条可操作的备考建议，针对该考生本次作文的薄弱点。
"""
    contents: list = []
    image_bytes = base64_to_image_bytes(question_image_b64) if question_image_b64 else None
    if image_bytes:
        contents.append(
            genai_types.Part.from_bytes(
                data=image_bytes,
                mime_type=image_mime_type(image_bytes),
            )
        )
    contents.append(prompt)
    # 写作批改是纯推理任务，这里保留动态思考（不像语音评分那样思考纯属浪费）。
    response = generate_gemini_content_with_retry(contents=contents, thinking_budget=None)
    return response.text


DEEPSEEK_TASK2_SYSTEM_PROMPT = """你是一名经过 British Council 严格培训的资深雅思考官，拥有超过 15 年执考经验，曾批改过数万份中国大陆考生的 Task 2 作文。

【核心使命】
严格按照雅思官方 Writing Task 2 四项评分标准（TR, CC, LR, GRA），对考生作文进行极其真实、严苛、证据驱动的评分。你的批改报告必须达到"考生拿到后可以直接作为复议依据"的专业水准。

【中国大陆考生基准画像（极其重要——你必须内化为评分直觉）】
- 根据历年官方统计数据，中国大陆考生 A 类写作平均分约为 5.5-5.8。
- 这意味着：一篇"看起来还行、没有明显跑题、语法大致正确"的作文，大概率落在 5.5-6.0 区间，而非 6.5-7.0。
- 6.5 分已经是前 25% 的水平，7.0 分是前 10% 的水平。不要轻易给出 7.0+。
- 当你觉得"还不错"时，默认起点是 5.5。只有找到明确的加分证据后，再向上调整。

【评分四维铁律——逐项严格扣分】

一、Task Response (TR) —— 任务回应
评分逻辑：先扣分，再找亮点。
扣分触发（以下每条满足即至少扣 0.5，可叠加）：
• 题目有两个及以上问题/要求，只回应了部分 → -0.5 起步
• 立场模糊、摇摆或自相矛盾 → -0.5
• 主体段落的论点没有具体例子/论据支撑，仅为空洞断言 → -0.5
• 例子与论点不匹配或牵强 → -0.5
• 跑题或引入大量无关内容 → -1.0 起步
• 字数严重不足（<200 词）→ -0.5 至 -1.0
加分条件（必须满足扣分项极少时再考虑）：
• 有 nuanced 的立场（承认复杂性而非简单二元）→ +0.5
• 例证具体、贴切、有说服力 → +0.5
TR 起点：5.0。逐条加减后锁定。

二、Coherence and Cohesion (CC) —— 连贯与衔接
评分逻辑：结构骨架决定上限。
扣分触发：
• 没有清晰的开头段-主体段-结尾段结构 → -0.5
• 主体段内部没有明确的中心句（Topic Sentence）→ -0.5
• 段落内句子之间逻辑跳跃，缺乏推进关系 → -0.5
• 过度使用或错误使用机械连接词（Firstly/Secondly/Moreover 堆砌，每段超过 2 个即算过度）→ -0.5
• 代词指代不清，this/these/it 没有明确的先行词 → -0.5
• 结尾段引入新观点而非总结 → -0.5

三、Lexical Resource (LR) —— 词汇丰富程度
评分逻辑：地道 > 花哨，搭配 > 单词。
扣分触发：
• 拼写错误每 3 处 → -0.5（累加）
• 用词不当导致语义扭曲（如将 "economic" 写成 "economical"）→ 每处至少 -0.25
• 全文反复使用 5 个以上的高频基础词（good, bad, important, big, thing 等）且无任何替换 → -0.5
• 滥用所谓"高级词汇"但搭配不地道 → -0.5
• 词性错误（如将名词当动词用）→ 每处 -0.25
加分条件：
• 使用地道 collocation 且准确 → +0.5
• 词汇场（lexical field）丰富，同一概念能用不同词表达 → +0.5

四、Grammatical Range and Accuracy (GRA) —— 语法多样性与准确性
评分逻辑：无错句比例是硬指标。
扣分触发：
• 简单句占比 > 70% → -0.5
• 尝试复杂句但系统性出错（如从句缺谓语、run-on sentence 超过 3 处）→ -0.5
• 时态混乱（过去/现在/完成体混用超过 3 处）→ -0.5
• 主谓一致错误超过 2 处 → -0.5
• 冠词和介词错误系统性地频繁出现（>5 处）→ -0.5
• 标点错误影响可读性（逗号拼接句等）→ -0.5
加分条件：
• 准确使用了 3 种以上不同从句类型（定语从句、状语从句、名词性从句）且错误率 < 20% → +0.5
• 有至少一处高分句式（倒装、强调句、虚拟语气等），且使用正确 → +0.5

【综合打分强行约束——防止"端水"】
1. TR 和 CC 是"天花板"：如果你的 TR 给了 5.5，CC 给 6.5 就极其可疑（逻辑都回应不好，结构怎么可能优秀？）四项分数必须相互印证，相邻两项差距通常不超过 1.0 分。若有例外，必须在报告中单独解释。
2. 总分计算：按 (TR + CC + LR + GRA) / 4 后，向下取整到最近的 0.5。例如 5.875 → 5.5，不是 6.0。
3. 终极校准反思（每次评分前必须自问）：
   - "这篇作文如果放在 10 万份中国大陆考生的答卷中，真的能排进前 10%（7.0+）吗？"
   - "我给出的分数比平均分 5.5 高了多少？每一分的增幅都有充分证据吗？"
   - "如果另一位雅思考官看到我的评分，会不会认为我给高了？"

【语气铁律】
- 彻底放弃 AI 助手的温暖口吻。你是考官，不是陪练。
- 严禁使用任何 Emoji。
- 用专业、克制、一针见血的学术语气。直接指出问题，不要用"建议""可以考虑""或许可以"这类模糊措辞。
- 中文点评部分简洁有力，英文术语部分保持原汁原味。"""


def evaluate_writing_task2_deepseek(
    topic: str,
    instructions: str,
    user_essay: str,
) -> str:
    user_prompt = f"""
请以资深雅思考官身份，对以下 Task 2 作文进行严格批改。

【题目主题】：{topic}

【题目完整要求】：
{instructions}

【考生作文全文】：
{user_essay}

请严格按照以下结构输出批改报告。格式必须精确，不得遗漏任何部分。

---

## 📊 总评分 (Overall Band Score)

格式：**X.X 分**

必须包含：
- 总分计算过程：(TR + CC + LR + GRA) / 4 = Y.Y → 向下取整至 X.X
- 一句话总结该分数对应的真实水平段位（如"该分数位于中国大陆考生前 X% 水平"）

---

## 1️⃣ 任务回应 (Task Response — TR)

**分数：X.X**

【扣分清单】（逐条列出）：
- [具体问题] → 扣分幅度
- （若本条无扣分项，写"无"）

【加分证据】（逐条列出）：
- [具体亮点] → 加分幅度
- （若本条无加分项，写"无"）

【TR 评语】：
英文核心诊断（2-3 句）+ 中文详细解读（指出论证的硬伤或亮点，引用原文中的具体句子作为证据）

---

## 2️⃣ 连贯与衔接 (Coherence and Cohesion — CC)

**分数：X.X**

（结构同上：扣分清单 / 加分证据 / CC 评语）

---

## 3️⃣ 词汇丰富程度 (Lexical Resource — LR)

**分数：X.X**

（结构同上）

【词汇替换建议】（至少 3 组）：
| 原文低级用词 | 考场地道替换 | 替换理由 |
|---|---|---|

---

## 4️⃣ 语法多样性与准确性 (Grammatical Range and Accuracy — GRA)

**分数：X.X**

（结构同上）

【句式统计】：
- 简单句占比：约 X%
- 复合句占比：约 X%
- 从句类型使用：____（列出实际出现的从句类型）
- 语法错误总数：约 X 处
- 无错句比例：约 X%

---

## 📝 逐句精批（至少 5 处）

| 编号 | 原句 | 考官改写 | 错误类型 | 修改说明 |
|---|---|---|---|---|
| 1 | （原文摘录） | （地道改写） | 语法/词汇/逻辑 | （中文） |

---

## 🔬 考官诊断总评

用约 150 字中文精准概括：
1. 这篇作文的核心短板是什么（只有一个，不要列清单）
2. 如果考生想从当前分数提升到 +0.5 的下一档，必须优先攻克的单一问题
3. 预计需要多长时间的系统训练才能实现这一跨越

---

## 💡 针对性提分训练计划

三条建议，每条必须包含：
- 具体问题 → 具体操作 → 预期效果
- 严禁泛泛而谈的"多读多写多练"
    """

    response = client_admin.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": DEEPSEEK_TASK2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def route_writing_evaluation(
    task_type: str,
    topic: str,
    instructions: str,
    question_image_b64: str | None,
    user_essay: str,
) -> str:
    if task_type == "Task 1":
        return evaluate_writing_task1_gemini(
            topic, instructions, question_image_b64, user_essay
        )
    if task_type == "Task 2":
        return evaluate_writing_task2_deepseek(topic, instructions, user_essay)
    raise ValueError(f"未知题型：{task_type}")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = ""

if not st.session_state.logged_in:
    st.title(" 高分英语训练舱 - 内部邀请版")
    # 用 form 包起来，密码框里按回车即可提交（不必去点按钮）。
    with st.form("login_form"):
        username = st.text_input("👤 账号")
        password = st.text_input("🔑 密码", type="password")
        submitted = st.form_submit_button("登录", type="primary")

    if submitted:
        if username in USER_DATABASE and USER_DATABASE[username] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = username
            st.session_state.current_page = "🏠 训练台"
            st.rerun()
        else:
            st.error("❌ 账号或密码错误！")

else:
    current_user = st.session_state.current_user
    if st.session_state.get("requested_page") in NAV_PAGES:
        st.session_state.current_page = st.session_state.pop("requested_page")
    if st.session_state.get("current_page") not in NAV_PAGES:
        st.session_state.current_page = "🏠 训练台"

    st.sidebar.write(f"👤 当前练习者：**{current_user}**")
    
    # --- 👑 管理员后台 (DeepSeek 接管 PDF 解析) ---
    if current_user == "admin":
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ 管理员后台")
        upload_target = st.sidebar.radio(
            "🎯 选择导入目标：",
            ["🗣️ 口语题库", "📖 阅读文章库", "✍️ 管理写作题库"],
        )
        
        if upload_target == "🗣️ 口语题库":
            uploaded_file = st.sidebar.file_uploader("📂 智能导入口语题 (CSV / PDF)", type=["csv", "pdf"])
            st.sidebar.caption("CSV 需要包含 part / theme / question 三列。")
            if uploaded_file is not None and st.sidebar.button("🚀 启动智能分析与导入"):
                import_bank_file(
                    uploaded_file,
                    table="question_bank",
                    field_map=QUESTION_BANK_FIELDS,
                    pdf_instruction=(
                        "提取以下文本中的所有雅思口语题目。\n"
                        "请严格将结果以 JSON 数组的形式返回。每一个元素包含三个键："
                        '"part"（如 "Part 1", "Part 2"）、"theme"（主题）、"question"（具体英文题目）。'
                    ),
                    label="口语题",
                    clear_cache=clear_question_bank_caches,
                )

        elif upload_target == "📖 阅读文章库":
            input_method = st.sidebar.radio("📥 录入方式：", ["📁 文件上传", "✍️ 手动粘贴文本"])
            if input_method == "📁 文件上传":
                uploaded_file = st.sidebar.file_uploader("📂 导入阅读文章 (CSV / PDF)", type=["csv", "pdf"])
                st.sidebar.caption("CSV 需要包含 title / content 两列。")
                if uploaded_file is not None and st.sidebar.button("🚀 启动智能分析与导入"):
                    import_bank_file(
                        uploaded_file,
                        table="reading_bank",
                        field_map=READING_BANK_FIELDS,
                        pdf_instruction=(
                            "提取以下文本中适合英语朗读的段落或文章。\n"
                            "请严格以 JSON 数组返回。每个元素包含两个键："
                            '"title"（文章或段落的标题/概括）、"content"（具体的英文原文正文）。'
                        ),
                        label="阅读文章",
                        clear_cache=clear_reading_bank_caches,
                    )

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
                        clear_reading_bank_caches()
                    else:
                        st.sidebar.warning("⚠️ 标题和正文都不能为空哦！")

        elif upload_target == "✍️ 管理写作题库":
            init_admin_writing_session_state()
            st.sidebar.markdown("**✍️ 写作题库上传**（上传图片 → AI 自动识题）")

            writing_task_type = st.sidebar.selectbox(
                "📋 选择题型：",
                ["Task 1", "Task 2"],
                key="admin_writing_task_type",
            )
            uploader_key = f"admin_writing_image_{st.session_state.admin_writing_uploader_gen}"
            writing_image = st.sidebar.file_uploader(
                "🖼️ 上传题目图片（png / jpg）",
                type=["png", "jpg", "jpeg"],
                key=uploader_key,
            )

            if writing_image is None:
                if st.session_state.admin_writing_fingerprint:
                    reset_admin_writing_session_state()
                st.sidebar.caption("请先上传题目图片，AI 将自动提取英文题目文本。")
            else:
                img_bytes = writing_image.getvalue()
                fingerprint = hashlib.md5(img_bytes).hexdigest()

                if st.session_state.admin_writing_fingerprint != fingerprint:
                    with st.spinner("🤖 AI 正在努力看图提取题目文字中..."):
                        try:
                            parsed = extract_writing_prompt_from_image(img_bytes)
                            st.session_state.admin_writing_fingerprint = fingerprint
                            st.session_state.admin_writing_image_b64 = base64.b64encode(
                                img_bytes
                            ).decode("utf-8")
                            st.session_state.admin_writing_topic_draft = parsed["topic"]
                            st.session_state.admin_writing_content_draft = parsed["instructions"]
                            st.session_state.admin_writing_ready = True
                            st.session_state.admin_writing_form_gen += 1
                        except Exception as e:
                            st.sidebar.error(f"AI 识图失败：{e}")

                st.sidebar.image(img_bytes, caption="题目预览", width="stretch")

                if st.session_state.admin_writing_ready:
                    st.sidebar.text_input(
                        "📌 题目主题 (topic)",
                        value=st.session_state.admin_writing_topic_draft,
                        key=admin_writing_topic_widget_key(),
                    )
                    st.sidebar.text_area(
                        "📝 题目要求 (instructions)",
                        value=st.session_state.admin_writing_content_draft,
                        height=280,
                        key=admin_writing_content_widget_key(),
                    )
                    st.sidebar.caption("请核对 topic 与 instructions，确认无误后再保存。")

            if st.sidebar.button("🚀 保存至写作题库", type="primary", key="btn_save_writing"):
                topic = get_admin_writing_topic()
                content = get_admin_writing_content()
                image_b64 = st.session_state.get("admin_writing_image_b64", "")
                if not image_b64:
                    st.sidebar.warning("⚠️ 请先上传题目图片！")
                elif not topic:
                    st.sidebar.warning("⚠️ 题目主题 (topic) 不能为空！")
                elif not content:
                    st.sidebar.warning("⚠️ 题目要求 (instructions) 不能为空！")
                else:
                    with st.spinner("正在写入写作题库..."):
                        supabase.table("writing_bank").insert({
                            "task_type": writing_task_type,
                            "title": topic,
                            "content": content,
                            "question_image": image_b64,
                        }).execute()
                    st.sidebar.success(f"✅ 已保存 {writing_task_type} 写作题！")
                    clear_writing_task_caches()
                    reset_admin_writing_session_state()
                    st.session_state.admin_writing_uploader_gen += 1
                    st.rerun()

        st.sidebar.markdown("---")
        st.sidebar.subheader(" 危险操作区")
        danger_confirm = st.sidebar.text_input(
            "输入 DELETE 才能启用清空按钮",
            key="admin_danger_confirm",
        )
        danger_confirmed = danger_confirm.strip() == "DELETE"
        if not danger_confirmed:
            st.sidebar.caption("清空操作不可恢复，请确认已备份题库。")

        if st.sidebar.button(
            "🚨 一键清空口语题库",
            type="primary",
            disabled=not danger_confirmed,
        ):
            supabase.table("question_bank").delete().neq("id", 0).execute()
            clear_question_bank_caches()
            st.sidebar.success("✅ 口语题库已清空！")
        if st.sidebar.button(
            "🚨 一键清空阅读文章",
            type="primary",
            disabled=not danger_confirmed,
        ):
            supabase.table("reading_bank").delete().neq("id", 0).execute()
            clear_reading_bank_caches()
            st.sidebar.success("✅ 阅读文章库已清空！")
        if st.sidebar.button(
            "🚨 一键清空写作题库",
            type="primary",
            disabled=not danger_confirmed,
        ):
            supabase.table("writing_bank").delete().neq("id", 0).execute()
            clear_writing_task_caches()
            st.sidebar.success("✅ 写作题库已清空！")
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📍 功能导航",
        NAV_PAGES,
        key="current_page",
    )
    if st.sidebar.button("🚪 退出登录"):
        # 必须整体清空：录音、作文草稿、批改结果都挂在 session_state 上，
        # 只清 logged_in 的话，同一浏览器换账号登录会看到上一个人的数据。
        st.session_state.clear()
        st.rerun()

    st.title("专属英语训练舱 🚀")

    # ==========================================
    # 个人档案
    # ==========================================
    if page == "🏠 训练台":
        render_training_dashboard(current_user)

    elif page == "👤 个人档案":
        st.subheader("👤 个人档案")
        saved_info = load_profile_information(current_user)
        profile_text = st.text_area(
            "请输入你的个人背景、专业、爱好等",
            value=saved_info,
            height=300,
            placeholder="例如：我是计算机专业大三学生，喜欢摄影和徒步，曾在杭州实习……",
        )
        if st.button("保存档案", type="primary"):
            save_profile_information(current_user, profile_text.strip())
            st.success("✅ 档案已保存！模拟考官答题时会自动用于生成专属答案。")
            st.rerun()

    # ==========================================
    # 模块一：模拟考官 (使用 client_voice 当考官)
    # ==========================================
    elif page == "🗣️ 模拟考官":
        IELTS_BANK = load_question_bank()

        st.subheader("📝 Step 1: 从题库中抽题")
        if not IELTS_BANK:
            st.info("当前题库为空，请联系管理员在左侧上传题库。")
        else:
            selected_part = st.selectbox("📂 选择 Part：", list(IELTS_BANK.keys()), key="qa_part")
            selected_theme = st.selectbox("🏷️ 选择主题 (Theme)：", list(IELTS_BANK[selected_part].keys()), key="qa_theme")
            question = st.selectbox("🎯 选择具体题目：", IELTS_BANK[selected_part][selected_theme], key="qa_q")
            st.info(f"**考官提问：** {question}")

            past_records = load_practice_history(current_user, question)
            if past_records:
                with st.expander(f"📖 查看这道题最近 {len(past_records)} 次点评记录"):
                    render_history_records(past_records, "record_text")

            st.write("---")
            st.subheader("🗣️ Step 2: 你的回答")
            
            qa_key_name = f"counter_{question}"
            if qa_key_name not in st.session_state:
                st.session_state[qa_key_name] = 0
                
            audio_bytes_qa = audio_recorder(
                text="点击麦克风开始作答",
                icon_size="2x",
                pause_threshold=PAUSE_THRESHOLD_SPEAKING,
                sample_rate=RECORDER_SAMPLE_RATE,
                key=f"recorder_qa_{question}_{st.session_state[qa_key_name]}"
            )
            st.caption(
                f"答完点一下麦克风就能立即结束；不点的话，静音满 {PAUSE_THRESHOLD_SPEAKING:.0f} 秒会自动停止。"
            )

            qa_audio_key = f"audio_qa_{question}"
            qa_audio_hash_key = f"audio_hash_qa_{question}"
            release_inactive_audio(qa_audio_key)
            if audio_bytes_qa:
                st.session_state[qa_audio_key] = audio_bytes_qa
                st.session_state[qa_audio_hash_key] = audio_fingerprint(audio_bytes_qa)

            saved_audio_qa = st.session_state.get(qa_audio_key)
            saved_audio_hash_qa = st.session_state.get(qa_audio_hash_key, "")

            if saved_audio_qa:
                st.audio(saved_audio_qa, format="audio/wav")
                show_audio_payload_info(saved_audio_qa)
                st.success("录音已保存。确认无误后点击下方按钮提交评分。")
                last_audio_tracker_qa = f"last_audio_{question}"
                qa_pending_key = f"pending_qa_{question}"
                qa_answer_key = f"personalized_answer_{question}_{saved_audio_hash_qa}"
                qa_already_scored = st.session_state.get(last_audio_tracker_qa) == saved_audio_hash_qa

                if qa_already_scored:
                    st.info("本次录音已完成评分。需要重新评分请先清除录音后再录一次。")
                    render_personalized_answer(
                        question,
                        current_user,
                        qa_answer_key,
                        f"answer_qa_{question}_{st.session_state[qa_key_name]}",
                    )
                elif st.session_state.get(qa_pending_key):
                    status_box = st.status(
                        "正在处理本次录音评分，请勿刷新页面",
                        expanded=True,
                    )
                    qa_started_at = st.session_state.get(f"{qa_pending_key}_started_at", "刚刚")
                    status_box.write(f"已收到提交（{qa_started_at}），正在发送录音给 Gemini……")
                    try:
                        prompt = f"""
                        你现在是一名雅思口语考官。考生 {current_user} 正在回答题目：“{question}”。
                        请你：
                        1. 【精准听写】：写下听到的英文原话。
                        2. 【切题度与雅思预估分】：评价是否切题，给出预估分数。
                        3. 【纠错与升级】：指出语法、词汇、逻辑上的具体问题，并给出可操作的改进方向（不要写完整示范答案，示范答案会单独生成）。
                        4. 【考官建议】：用中文给一段备考建议。
                        """
                        qa_report = st.write_stream(
                            stream_gemini_content(
                                contents=[wav_audio_part(saved_audio_qa), prompt]
                            )
                        )
                        status_box.update(label="点评完成，正在归档", state="running")
                        st.success("🎉 考官点评完成！")

                        supabase.table("practice_history").insert({
                            "username": current_user,
                            "question": question,
                            "record_text": qa_report
                        }).execute()
                        load_practice_history.clear()

                        st.session_state[last_audio_tracker_qa] = saved_audio_hash_qa
                        st.session_state[qa_pending_key] = False
                        st.session_state.pop(f"{qa_pending_key}_started_at", None)
                        status_box.update(label="本次录音评分完成", state="complete")
                        st.info("如需专属参考答案，请点击下方按钮生成。")
                        # key 必须和「已评分」分支保持一致：点击后会 rerun 并切到那个分支，
                        # key 不同的话按钮返回值会丢，用户得点两次才生效。
                        render_personalized_answer(
                            question,
                            current_user,
                            qa_answer_key,
                            f"answer_qa_{question}_{st.session_state[qa_key_name]}",
                        )
                        
                    except Exception as e:
                        st.session_state[qa_pending_key] = False
                        st.session_state.pop(f"{qa_pending_key}_started_at", None)
                        status_box.update(label="本次评分未完成", state="error")
                        show_gemini_busy_error(e)
                        if st.button(
                            "重新提交本次录音评分",
                            key=f"retry_submit_qa_{question}_{st.session_state[qa_key_name]}",
                        ):
                            submit_audio_for_scoring(qa_pending_key)
                elif st.button(
                    "📤 提交本次录音评分",
                    type="primary",
                    key=f"submit_qa_{question}_{st.session_state[qa_key_name]}",
                ):
                    submit_audio_for_scoring(qa_pending_key)

                st.markdown("---")
                if st.button("🔄 不满意？清除录音，再练一次！", key=f"btn_qa_{question}_{st.session_state[qa_key_name]}"):
                    st.session_state.pop(qa_audio_key, None)
                    st.session_state.pop(qa_audio_hash_key, None)
                    st.session_state.pop(last_audio_tracker_qa, None)
                    st.session_state.pop(qa_answer_key, None)
                    st.session_state.pop(qa_pending_key, None)
                    st.session_state.pop(f"{qa_pending_key}_started_at", None)
                    st.session_state[qa_key_name] += 1
                    st.rerun()

    # ==========================================
    # 模块 1.5：我的素材朗读（用户自备素材 + 发音打分）
    # ==========================================
    elif page == "🎤 我的素材朗读":
        st.subheader("🎤 我的素材朗读")
        st.caption(
            "把你自己准备的素材贴进来直接练。评分覆盖发音准确度、流利度、重音与语调，"
            "并给出可切换口音和语速的示范朗读。"
        )

        if not speaking_tables_ready():
            st.warning("素材库还没初始化 —— 需要先在数据库里建两张表（只需做一次）。")
            st.markdown("打开 Supabase 控制台 → **SQL Editor** → 粘贴并运行下面的语句，然后刷新本页：")
            st.code(SPEAKING_MATERIALS_SQL, language="sql")
        else:
            saved_msg = st.session_state.pop("material_saved_msg", "")
            if saved_msg:
                st.success(f"✅ 已存入素材库：{saved_msg}")

            materials = load_speaking_materials(current_user)
            temp_material = st.session_state.get("temp_material")

            source_options = ["📚 我的素材库", "✍️ 贴一段新素材"]
            # 和 requested_page 一样的套路：widget 的 key 一旦实例化就不能再改，
            # 所以切换请求先存到另一个 key，在创建 radio 之前套用。
            requested_source = st.session_state.pop("material_source_request", None)
            if requested_source in source_options:
                st.session_state.material_source = requested_source

            default_source = 0 if materials and not temp_material else 1
            source = st.radio(
                "📥 素材来源：",
                source_options,
                index=default_source,
                horizontal=True,
                key="material_source",
            )

            active_title = ""
            active_content = ""
            active_scope = ""

            if source == "✍️ 贴一段新素材":
                # 必须用 form：st.text_area 的输入要等失焦或 ⌘+Enter 才会回传，
                # 用普通按钮 + disabled 判断的话，按钮会一直是灰的，而灰按钮又接不到
                # 那次「点击顺带让输入框失焦」的交互，用户就卡死了。
                # form 的 submit 会把表单内所有控件的当前值一起提交。
                with st.form("material_new_form"):
                    new_content = st.text_area(
                        "📝 粘贴英文素材正文",
                        height=220,
                        placeholder="把你素材库里的段落、范文或者想练的句子贴进来……",
                    )
                    new_title = st.text_input("🏷️ 素材标题（留空自动用正文开头命名）")
                    col_try, col_save = st.columns(2)
                    practice_only = col_try.form_submit_button(
                        "▶️ 直接练（不保存）", width="stretch"
                    )
                    save_and_practice = col_save.form_submit_button(
                        "💾 存入素材库并开始练", type="primary", width="stretch"
                    )

                trimmed = new_content.strip()
                resolved_title = new_title.strip() or " ".join(trimmed.split()[:6])

                if (practice_only or save_and_practice) and not trimmed:
                    st.warning("⚠️ 请先粘贴素材正文再提交。")
                elif practice_only:
                    st.session_state.temp_material = {
                        "title": resolved_title or "临时素材",
                        "content": trimmed,
                    }
                    st.rerun()
                elif save_and_practice:
                    try:
                        save_speaking_material(
                            current_user, resolved_title or "未命名素材", trimmed
                        )
                    except Exception as e:
                        st.error(f"保存失败：{e}")
                    else:
                        st.session_state.pop("temp_material", None)
                        # 存完自动切回素材库，新素材排在最前面会被自动选中。
                        st.session_state.material_source_request = "📚 我的素材库"
                        st.session_state.material_saved_msg = resolved_title or "未命名素材"
                        st.rerun()

                if temp_material:
                    st.info(f"当前正在练习临时素材：**{temp_material['title']}**（未保存）")
                    active_title = temp_material["title"]
                    active_content = temp_material["content"]
                    active_scope = f"t{hashlib.md5(active_content.encode()).hexdigest()[:8]}"
                    if st.button("🗑️ 结束这段临时素材"):
                        st.session_state.pop("temp_material", None)
                        st.rerun()

            else:
                if not materials:
                    st.info("素材库还是空的。切到「✍️ 贴一段新素材」加一条吧。")
                else:
                    options = list(range(len(materials)))
                    picked = st.selectbox(
                        "📂 选择素材：",
                        options,
                        format_func=lambda i: (
                            f"{materials[i]['title']}"
                            f"（{count_words(materials[i]['content'])} 词）"
                        ),
                        key="material_pick",
                    )
                    chosen = materials[picked]
                    active_title = chosen["title"]
                    active_content = chosen["content"]
                    active_scope = f"m{chosen['id']}"

                    with st.expander("🗑️ 删除这条素材"):
                        st.caption("删除后不可恢复，历史评分记录会保留。")
                        if st.button("确认删除", key=f"del_material_{chosen['id']}"):
                            delete_speaking_material(current_user, chosen["id"])
                            st.success("已删除。")
                            st.rerun()

            if active_content:
                st.markdown("---")
                practice_mode = st.radio(
                    "🎯 练习范围：",
                    ["📖 整段连读", "🔍 逐句精读"],
                    horizontal=True,
                    key="material_mode",
                )

                if practice_mode == "📖 整段连读":
                    target_text = active_content
                    history_title = active_title
                    scope_id = active_scope
                    st.markdown(f"**请朗读以下内容：**\n> ### {target_text}")
                else:
                    sentences = split_sentences(active_content)
                    sentence_idx = st.selectbox(
                        "📍 选择要攻克的句子：",
                        range(len(sentences)),
                        format_func=lambda i: f"第 {i+1} 句: {sentences[i][:40]}...",
                        key="material_sentence",
                    )
                    target_text = sentences[sentence_idx]
                    history_title = f"{active_title} (第{sentence_idx+1}句)"
                    scope_id = f"{active_scope}_s{sentence_idx}"
                    st.markdown(
                        f"**请朗读当前句子（第 {sentence_idx+1}/{len(sentences)} 句）：**"
                        f"\n> ### {target_text}"
                    )

                st.markdown("---")
                st.markdown("#### 🔊 先听示范")
                render_tts_demo(target_text, key_prefix=f"material_tts_{scope_id}")

                past_records = load_material_history(current_user, history_title)
                if past_records:
                    with st.expander(f"📖 查看这条素材最近 {len(past_records)} 次评分记录"):
                        render_history_records(past_records, "record_text")

                st.markdown("---")
                st.markdown("#### 🎙️ 轮到你了")
                render_recording_practice(
                    scope_id=scope_id,
                    audio_prefix="material",
                    reference_text=target_text,
                    build_prompt=lambda m=None: build_material_scoring_prompt(target_text, m),
                    save_history=lambda report: save_material_history(
                        current_user, history_title, report
                    ),
                    recorder_text="点击麦克风开始朗读",
                    submit_label="📤 提交录音，开始发音打分",
                    reset_label="🔄 读得不满意？清除录音重来",
                    success_message="🎉 发音评分报告已生成！",
                    celebrate=True,
                )

    # ==========================================
    # 模块二：英文原版朗读纠音 (使用 client_voice 当教练)
    # ==========================================
    elif page == "📖 英文原版朗读纠音":
        READING_MATERIALS = load_reading_bank()
        
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
                    st.audio(synthesize_tts_audio(target_text), format="audio/mp3", autoplay=True)

            past_reading_records = load_reading_history(current_user, db_save_title)
            
            if past_reading_records:
                with st.expander(f"📖 查看此项最近 {len(past_reading_records)} 次纠音记录"):
                    render_history_records(past_reading_records, "record_text")

            st.write("---")
            st.subheader("🎙️ 轮到你了")
            
            reading_key_name = f"counter_{db_save_title}"
            if reading_key_name not in st.session_state:
                st.session_state[reading_key_name] = 0
                
            audio_bytes_reading = audio_recorder(
                text="点击录制你的朗读",
                icon_size="2x",
                pause_threshold=PAUSE_THRESHOLD_READING,
                sample_rate=RECORDER_SAMPLE_RATE,
                key=f"recorder_reading_{db_save_title}_{st.session_state[reading_key_name]}"
            )
            st.caption(
                f"读完点一下麦克风就能立即结束；不点的话，静音满 {PAUSE_THRESHOLD_READING:.0f} 秒会自动停止。"
            )

            reading_audio_key = f"audio_reading_{db_save_title}"
            reading_audio_hash_key = f"audio_hash_reading_{db_save_title}"
            release_inactive_audio(reading_audio_key)
            if audio_bytes_reading:
                st.session_state[reading_audio_key] = audio_bytes_reading
                st.session_state[reading_audio_hash_key] = audio_fingerprint(audio_bytes_reading)

            saved_audio_reading = st.session_state.get(reading_audio_key)
            saved_audio_hash_reading = st.session_state.get(reading_audio_hash_key, "")

            if saved_audio_reading:
                st.audio(saved_audio_reading, format="audio/wav")
                show_audio_payload_info(saved_audio_reading)
                st.success("录音已保存。确认无误后点击下方按钮提交评分。")
                last_audio_tracker_reading = f"last_audio_{db_save_title}"
                reading_pending_key = f"pending_reading_{db_save_title}"
                reading_already_scored = (
                    st.session_state.get(last_audio_tracker_reading) == saved_audio_hash_reading
                )

                if reading_already_scored:
                    st.info("本次录音已完成评分。需要重新评分请先清除录音后再录一次。")
                elif st.session_state.get(reading_pending_key):
                    status_box = st.status(
                        "正在处理本次朗读评分，请勿刷新页面",
                        expanded=True,
                    )
                    reading_started_at = st.session_state.get(
                        f"{reading_pending_key}_started_at",
                        "刚刚",
                    )
                    status_box.write(f"已收到提交（{reading_started_at}），正在发送录音给 Gemini……")
                    try:
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
                        reading_report = st.write_stream(
                            stream_gemini_content(
                                contents=[wav_audio_part(saved_audio_reading), prompt]
                            )
                        )
                        status_box.update(label="评分完成，正在归档", state="running")
                        st.success("🎉 发音诊断报告已生成！")
                        st.balloons()

                        supabase.table("reading_history").insert({
                            "username": current_user,
                            "reading_title": db_save_title,
                            "record_text": reading_report
                        }).execute()
                        load_reading_history.clear()

                        st.session_state[last_audio_tracker_reading] = saved_audio_hash_reading
                        st.session_state[reading_pending_key] = False
                        st.session_state.pop(f"{reading_pending_key}_started_at", None)
                        status_box.update(label="本次朗读评分完成", state="complete")
                        
                    except Exception as e:
                        st.session_state[reading_pending_key] = False
                        st.session_state.pop(f"{reading_pending_key}_started_at", None)
                        status_box.update(label="本次评分未完成", state="error")
                        show_gemini_busy_error(e)
                        if st.button(
                            "重新提交本次朗读评分",
                            key=f"retry_submit_reading_{db_save_title}_{st.session_state[reading_key_name]}",
                        ):
                            submit_audio_for_scoring(reading_pending_key)
                elif st.button(
                    "📤 提交本次朗读评分",
                    type="primary",
                    key=f"submit_reading_{db_save_title}_{st.session_state[reading_key_name]}",
                ):
                    submit_audio_for_scoring(reading_pending_key)

                st.markdown("---")
                if st.button("🔄 感觉没读顺？清除录音，重读本句！", key=f"btn_reading_{db_save_title}_{st.session_state[reading_key_name]}"):
                    st.session_state.pop(reading_audio_key, None)
                    st.session_state.pop(reading_audio_hash_key, None)
                    st.session_state.pop(last_audio_tracker_reading, None)
                    st.session_state.pop(reading_pending_key, None)
                    st.session_state.pop(f"{reading_pending_key}_started_at", None)
                    st.session_state[reading_key_name] += 1
                    st.rerun()

    # ==========================================
    # 模块三：雅思写作练习 (Gemini 多模态批改)
    # ==========================================
    elif page == "✍️ 雅思写作练习":
        st.subheader("📝 Step 1: 选择题目")
        writing_task_type = st.selectbox(
            "📋 选择题型：",
            ["Task 1", "Task 2"],
            key="writing_task_type",
        )
        word_target = 150 if writing_task_type == "Task 1" else 250
        tasks = load_writing_tasks(writing_task_type)

        if not tasks:
            st.info(
                f"当前 {writing_task_type} 题库为空，请联系管理员在左侧「✍️ 管理写作题库」上传题目。"
            )
        else:
            def _task_label(idx: int) -> str:
                t = tasks[idx]
                return f"[{t['task_type']}] {t['title']}"

            selected_idx = st.selectbox(
                "🎯 选择具体题目：",
                range(len(tasks)),
                format_func=_task_label,
                key="writing_task_select",
            )
            selected_task_summary = tasks[selected_idx]
            task_id = selected_task_summary["id"]
            selected_task = {
                **selected_task_summary,
                **load_writing_task_detail(task_id),
            }
            task_instructions = (
                selected_task.get("content") or selected_task.get("title") or ""
            )

            st.markdown("**题目要求：**")
            st.markdown(task_instructions)

            img_bytes = base64_to_image_bytes(selected_task.get("question_image"))
            if img_bytes:
                st.image(img_bytes, caption="题目图表", width="stretch")

            past_writing = load_writing_history(current_user, task_id)
            if past_writing:
                with st.expander(f"📖 查看本题最近 {len(past_writing)} 次批改"):
                    render_history_records(past_writing, "evaluation")

            st.write("---")
            st.subheader("✍️ Step 2: 开始写作")
            st.caption(f"建议字数：{word_target} 词（{writing_task_type}）。草稿会保存在当前浏览器会话中，切换回来不会立刻丢失。")

            draft_key = f"writing_textarea_{task_id}"
            user_essay = st.text_area(
                "在此输入你的作文（英文）",
                height=350,
                key=draft_key,
            )
            word_count = count_words(user_essay)
            st.progress(min(word_count / word_target, 1.0))
            if word_count < word_target:
                st.warning(f"📏 当前字数：**{word_count}** / 建议 {word_target} 词（尚未达标）")
            else:
                st.success(f"📏 当前字数：**{word_count}** / 建议 {word_target} 词（已达标 ✅）")

            is_task1 = writing_task_type == "Task 1"
            current_essay_hash = (
                writing_fingerprint(task_id, user_essay)
                if user_essay.strip()
                else ""
            )
            writing_pending_key = f"writing_pending_{task_id}"
            writing_started_key = f"{writing_pending_key}_started_at"
            writing_result_key = f"writing_result_{task_id}"
            writing_result_hash_key = f"writing_result_hash_{task_id}"
            writing_error_key = f"writing_error_{task_id}"
            writing_submitted_essay_key = f"writing_submitted_essay_{task_id}"
            writing_submitted_hash_key = f"writing_submitted_hash_{task_id}"

            if st.session_state.get(writing_pending_key):
                submitted_essay = st.session_state.get(
                    writing_submitted_essay_key,
                    user_essay.strip(),
                )
                submitted_hash = st.session_state.get(
                    writing_submitted_hash_key,
                    writing_fingerprint(task_id, submitted_essay),
                )
                status_box = st.status(
                    "正在批改本次作文，请勿刷新页面",
                    expanded=True,
                )
                started_at = st.session_state.get(writing_started_key, "刚刚")
                status_box.write(f"1/4 已收到提交。开始时间：{started_at}")
                status_box.write("2/4 正在发送题目和作文给 AI。")
                try:
                    evaluation = route_writing_evaluation(
                        writing_task_type,
                        selected_task["title"],
                        task_instructions,
                        selected_task.get("question_image"),
                        submitted_essay,
                    )
                    status_box.write("3/4 AI 批改完成，正在保存历史记录。")
                    supabase.table("writing_history").insert({
                        "username": current_user,
                        "task_id": task_id,
                        "user_essay": submitted_essay,
                        "evaluation": evaluation,
                    }).execute()
                    load_writing_history.clear()

                    st.session_state[writing_result_key] = evaluation
                    st.session_state[writing_result_hash_key] = submitted_hash
                    st.session_state[writing_error_key] = ""
                    st.session_state[writing_pending_key] = False
                    st.session_state.pop(writing_started_key, None)
                    status_box.write("4/4 批改结果已归档。")
                    status_box.update(label="本次作文批改完成", state="complete")
                    st.success("🎉 批改完成！")
                    render_writing_evaluation_result(evaluation)
                    st.balloons()
                except Exception as e:
                    engine = "Gemini" if is_task1 else "DeepSeek"
                    st.session_state[writing_error_key] = f"{engine} 批改失败：{e}"
                    st.session_state[writing_pending_key] = False
                    st.session_state.pop(writing_started_key, None)
                    status_box.update(label="本次批改未完成", state="error")
                    st.error(st.session_state[writing_error_key])
                    st.caption("作文草稿已保留，可以直接重新提交。")
                    if st.button("重新提交本次作文", key=f"retry_writing_{task_id}"):
                        st.session_state[writing_pending_key] = True
                        st.session_state[writing_started_key] = time.strftime("%H:%M:%S")
                        st.rerun()
            else:
                previous_error = st.session_state.get(writing_error_key, "")
                if previous_error:
                    st.error(previous_error)
                    st.caption("作文草稿已保留，修改后可以重新提交。")

                stored_result = st.session_state.get(writing_result_key, "")
                stored_hash = st.session_state.get(writing_result_hash_key, "")
                if stored_result:
                    if stored_hash == current_essay_hash:
                        st.info("当前这版作文已经完成批改。修改草稿后可以再次提交。")
                    else:
                        st.warning("草稿已经修改。下方仍是上一次提交版本的批改结果。")
                    render_writing_evaluation_result(stored_result)

                submit_disabled = (
                    not user_essay.strip()
                    or bool(stored_result and stored_hash == current_essay_hash)
                )
                if st.button(
                    "📤 提交批改",
                    type="primary",
                    key=f"btn_submit_writing_{task_id}",
                    disabled=submit_disabled,
                ):
                    st.session_state[writing_submitted_essay_key] = user_essay.strip()
                    st.session_state[writing_submitted_hash_key] = current_essay_hash
                    st.session_state[writing_pending_key] = True
                    st.session_state[writing_started_key] = time.strftime("%H:%M:%S")
                    st.session_state[writing_error_key] = ""
                    st.rerun()


