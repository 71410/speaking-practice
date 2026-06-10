import streamlit as st
from audio_recorder_streamlit import audio_recorder
from google import genai
from google.genai import types as genai_types
from supabase import create_client, Client
import pandas as pd
import json
import re  
from gtts import gTTS
import base64
import hashlib
import io
import time
from openai import OpenAI
import PyPDF2

# --- 1. 🔑 核心配置区 (中西合璧：DeepSeek + Gemini) ---
GEMINI_API_KEY_VOICE = st.secrets["GEMINI_API_KEY_VOICE"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

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
GEMINI_FLASH_MODEL = "gemini-3.5-flash"


def is_transient_gemini_error(error: Exception) -> bool:
    message = str(error).lower()
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


def generate_gemini_content_with_retry(
    contents: list,
    model: str = GEMINI_FLASH_MODEL,
    retry_delays: tuple[int, ...] = (1, 2, 4),
):
    last_error = None
    for attempt in range(len(retry_delays) + 1):
        try:
            return client_voice.models.generate_content(model=model, contents=contents)
        except Exception as exc:
            last_error = exc
            if attempt >= len(retry_delays) or not is_transient_gemini_error(exc):
                raise
            time.sleep(retry_delays[attempt])
    raise last_error


def show_gemini_busy_error(error: Exception) -> None:
    if is_transient_gemini_error(error):
        st.error("Gemini 当前繁忙，自动重试后仍未成功。录音已保留，可以稍后直接重新请求评分。")
    else:
        st.error("Gemini 请求失败。录音已保留，可以稍后直接重新请求评分。")
    st.caption(f"服务返回：{str(error)[:240]}")


def wav_audio_part(audio_bytes: bytes):
    return genai_types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")


def show_audio_payload_info(audio_bytes: bytes) -> None:
    size_mb = len(audio_bytes) / 1024 / 1024
    st.caption(f"录音大小：{size_mb:.2f} MB。2 分钟内通常可以提交；如果超过 15 MB，建议分段练习。")
    if size_mb > 15:
        st.warning("这段录音较大，评分可能明显变慢或失败。建议缩短录音或分成多段提交。")


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


def file_to_base64(uploaded_file) -> str:
    return base64.b64encode(uploaded_file.getvalue()).decode("utf-8")


@st.cache_data(ttl=86400, show_spinner=False)
def synthesize_tts_b64(text: str, tld: str = "co.uk") -> str:
    sound_file = io.BytesIO()
    tts = gTTS(text=text, lang="en", tld=tld)
    tts.write_to_fp(sound_file)
    sound_file.seek(0)
    return base64.b64encode(sound_file.read()).decode()


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
    response = (
        supabase.table("writing_bank")
        .select("id, task_type, title, content")
        .eq("task_type", task_type)
        .order("id")
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


@st.cache_data(ttl=300, show_spinner=False)
def load_question_bank() -> dict:
    """加载完整口语题库，缓存 5 分钟。管理员上传新题后自动刷新。"""
    response = (
        supabase.table("question_bank")
        .select("part, theme, question_text")
        .execute()
    )
    bank: dict = {}
    for row in (response.data or []):
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
    response = supabase.table("reading_bank").select("title, content").execute()
    return {row["title"]: row["content"] for row in (response.data or [])}


@st.cache_data(ttl=30, show_spinner=False)
def load_practice_history(username: str, question: str) -> list:
    """加载某用户某道题的练习历史，缓存 30 秒。"""
    response = (
        supabase.table("practice_history")
        .select("record_text")
        .eq("username", username)
        .eq("question", question)
        .limit(5)
        .execute()
    )
    return response.data or []


@st.cache_data(ttl=30, show_spinner=False)
def load_reading_history(username: str, reading_title: str) -> list:
    """加载某用户某材料的朗读历史，缓存 30 秒。"""
    response = (
        supabase.table("reading_history")
        .select("record_text")
        .eq("username", username)
        .eq("reading_title", reading_title)
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
    response = generate_gemini_content_with_retry(contents=contents)
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
        upload_target = st.sidebar.radio(
            "🎯 选择导入目标：",
            ["🗣️ 口语题库", "📖 阅读文章库", "✍️ 管理写作题库"],
        )
        
        if upload_target == "🗣️ 口语题库":
            uploaded_file = st.sidebar.file_uploader("📂 智能导入口语题 (CSV / PDF)", type=["csv", "pdf"])
            if uploaded_file is not None:
                if st.sidebar.button("🚀 启动智能分析与导入"):
                    if uploaded_file.name.endswith('.csv'):
                        with st.spinner("正在写入口语表格..."):
                            df = pd.read_csv(uploaded_file)
                            rows = [
                                {
                                    "part": str(row["part"]),
                                    "theme": str(row["theme"]),
                                    "question_text": str(row["question"]),
                                }
                                for _, row in df.iterrows()
                            ]
                            if rows:
                                supabase.table("question_bank").insert(rows).execute()
                        st.sidebar.success("✅ 口语 CSV 导入成功！")
                        load_question_bank.clear()
                    
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
                                    temperature=0.1,
                                    max_tokens=8192  # 👈 新增这行！给它最大的肺活量！
                                )
                                
                                raw_text = response.choices[0].message.content.strip()
                                if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                if raw_text.startswith("```"): raw_text = raw_text[3:]
                                if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                
                                extracted_data = json.loads(raw_text.strip())
                                rows = [
                                    {
                                        "part": str(item.get("part", "未分类")),
                                        "theme": str(item.get("theme", "未分类")),
                                        "question_text": str(item.get("question", "提取失败")),
                                    }
                                    for item in extracted_data
                                ]
                                if rows:
                                    supabase.table("question_bank").insert(rows).execute()
                                st.sidebar.success(f"✅ DeepSeek 成功导入 {len(extracted_data)} 道口语题！")
                                load_question_bank.clear()
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
                                rows = [
                                    {
                                        "title": str(row["title"]),
                                        "content": str(row["content"]),
                                    }
                                    for _, row in df.iterrows()
                                ]
                                if rows:
                                    supabase.table("reading_bank").insert(rows).execute()
                            st.sidebar.success("✅ 阅读 CSV 导入成功！")
                            load_reading_bank.clear()
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
                                        temperature=0.1,
                                        max_tokens=8192  # 👈 新增这行！给它最大的肺活量！
                                    )
                                    
                                    raw_text = response.choices[0].message.content.strip()
                                    if raw_text.startswith("```json"): raw_text = raw_text[7:]
                                    if raw_text.startswith("```"): raw_text = raw_text[3:]
                                    if raw_text.endswith("```"): raw_text = raw_text[:-3]
                                    
                                    extracted_data = json.loads(raw_text.strip())
                                    rows = [
                                        {
                                            "title": str(item.get("title", "未命名文章")),
                                            "content": str(item.get("content", "内容提取失败")),
                                        }
                                        for item in extracted_data
                                    ]
                                    if rows:
                                        supabase.table("reading_bank").insert(rows).execute()
                                    st.sidebar.success(f"✅ DeepSeek 成功导入 {len(extracted_data)} 篇阅读文章！")
                                    load_reading_bank.clear()
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
                        load_reading_bank.clear()
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

                st.sidebar.image(img_bytes, caption="题目预览", use_container_width=True)

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
        if st.sidebar.button("🚨 一键清空口语题库", type="primary"):
            supabase.table("question_bank").delete().neq("id", 0).execute()
            load_question_bank.clear()
            st.sidebar.success("✅ 口语题库已清空！")
        if st.sidebar.button("🚨 一键清空阅读文章", type="primary"):
            supabase.table("reading_bank").delete().neq("id", 0).execute()
            load_reading_bank.clear()
            st.sidebar.success("✅ 阅读文章库已清空！")
        if st.sidebar.button("🚨 一键清空写作题库", type="primary"):
            supabase.table("writing_bank").delete().neq("id", 0).execute()
            clear_writing_task_caches()
            st.sidebar.success("✅ 写作题库已清空！")
    
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "📍 功能导航",
        [
            "🗣️ 模拟考官",
            "📖 英文原版朗读纠音",
            "✍️ 雅思写作练习",
            "👤 个人档案",
        ],
    )
    if st.sidebar.button("🚪 退出登录"):
        st.session_state.logged_in = False
        st.session_state.current_user = ""
        st.rerun()

    st.title("专属英语训练舱 🚀")

    # ==========================================
    # 个人档案
    # ==========================================
    if page == "👤 个人档案":
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
                pause_threshold=60.0,
                key=f"recorder_qa_{question}_{st.session_state[qa_key_name]}"
            )

            if audio_bytes_qa:
                st.audio(audio_bytes_qa, format="audio/wav")
                show_audio_payload_info(audio_bytes_qa)
                last_audio_tracker_qa = f"last_audio_{question}"
                qa_already_scored = st.session_state.get(last_audio_tracker_qa) == audio_bytes_qa

                if qa_already_scored:
                    st.info("本次录音已完成评分。需要重新评分请先清除录音后再录一次。")
                elif st.button(
                    "📤 提交本次录音评分",
                    type="primary",
                    key=f"submit_qa_{question}_{st.session_state[qa_key_name]}",
                ):
                    with st.spinner("🧠 专属考官 Voice 引擎正在仔细聆听..."):
                        try:
                            prompt = f"""
                            你现在是一名雅思口语考官。考生 {current_user} 正在回答题目：“{question}”。
                            请你：
                            1. 【精准听写】：写下听到的英文原话。
                            2. 【切题度与雅思预估分】：评价是否切题，给出预估分数。
                            3. 【纠错与升级】：指出语法、词汇、逻辑上的具体问题，并给出可操作的改进方向（不要写完整示范答案，示范答案会单独生成）。
                            4. 【考官建议】：用中文给一段备考建议。
                            """
                            response = generate_gemini_content_with_retry(
                                contents=[wav_audio_part(audio_bytes_qa), prompt]
                            )
                            st.success("🎉 考官点评完成！")
                            st.markdown(response.text)

                            profile_info = load_profile_information(current_user)
                            with st.spinner("✨ 正在根据你的个人档案定制专属答案..."):
                                personalized_answer = generate_personalized_answer(
                                    question, profile_info
                                )
                            st.markdown("---")
                            st.subheader("✨ 你的专属个性化答案")
                            if not profile_info.strip():
                                st.caption(
                                    "💡 你尚未填写个人档案，答案为通用示范。前往左侧「👤 个人档案」填写后，答案将更贴合你的真实人设。"
                                )
                            st.markdown(personalized_answer)
                            
                            supabase.table("practice_history").insert({
                                "username": current_user,
                                "question": question,
                                "record_text": response.text
                            }).execute()
                            load_practice_history.clear()

                            st.session_state[last_audio_tracker_qa] = audio_bytes_qa
                            
                        except Exception as e:
                            show_gemini_busy_error(e)

                st.markdown("---")
                if st.button("🔄 不满意？清除录音，再练一次！", key=f"btn_qa_{question}_{st.session_state[qa_key_name]}"):
                    st.session_state[qa_key_name] += 1
                    st.rerun()

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
                    b64 = synthesize_tts_b64(target_text)
                    md = f"""
                        <audio controls autoplay style="width: 100%;">
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                        您的浏览器不支持音频播放。
                        </audio>
                        """
                    st.markdown(md, unsafe_allow_html=True)

            past_reading_records = load_reading_history(current_user, db_save_title)
            
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
                pause_threshold=60.0,
                key=f"recorder_reading_{db_save_title}_{st.session_state[reading_key_name]}"
            )

            if audio_bytes_reading:
                st.audio(audio_bytes_reading, format="audio/wav")
                show_audio_payload_info(audio_bytes_reading)
                last_audio_tracker_reading = f"last_audio_{db_save_title}"
                reading_already_scored = (
                    st.session_state.get(last_audio_tracker_reading) == audio_bytes_reading
                )

                if reading_already_scored:
                    st.info("本次录音已完成评分。需要重新评分请先清除录音后再录一次。")
                elif st.button(
                    "📤 提交本次朗读评分",
                    type="primary",
                    key=f"submit_reading_{db_save_title}_{st.session_state[reading_key_name]}",
                ):
                    with st.spinner("🧠 专属教练 Voice 引擎正在评估..."):
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
                            response = generate_gemini_content_with_retry(
                                contents=[wav_audio_part(audio_bytes_reading), prompt]
                            )
                            st.success("🎉 发音诊断报告已生成！")
                            st.markdown(response.text)
                            st.balloons()
                            
                            supabase.table("reading_history").insert({
                                "username": current_user,
                                "reading_title": db_save_title,
                                "record_text": response.text
                            }).execute()
                            load_reading_history.clear()

                            st.session_state[last_audio_tracker_reading] = audio_bytes_reading
                            
                        except Exception as e:
                            show_gemini_busy_error(e)

                st.markdown("---")
                if st.button("🔄 感觉没读顺？清除录音，重读本句！", key=f"btn_reading_{db_save_title}_{st.session_state[reading_key_name]}"):
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
                st.image(img_bytes, caption="题目图表", use_container_width=True)

            past_writing = load_writing_history(current_user, task_id)
            if past_writing:
                with st.expander(f"📖 查看本题的 {len(past_writing)} 次历史批改"):
                    for i, record in enumerate(past_writing):
                        created = record.get("created_at", "")
                        st.markdown(f"**▶ 第 {i + 1} 次** {created}")
                        st.markdown(record["evaluation"])
                        st.write("---")

            st.write("---")
            st.subheader("✍️ Step 2: 开始写作")
            st.caption(f"建议字数：{word_target} 词（{writing_task_type}）")

            user_essay = st.text_area(
                "在此输入你的作文（英文）",
                height=350,
                key=f"writing_textarea_{task_id}",
            )
            word_count = count_words(user_essay)
            if word_count < word_target:
                st.warning(f"📏 当前字数：**{word_count}** / 建议 {word_target} 词（尚未达标）")
            else:
                st.success(f"📏 当前字数：**{word_count}** / 建议 {word_target} 词（已达标 ✅）")

            if st.button("📤 提交批改", type="primary", key=f"btn_submit_writing_{task_id}"):
                if not user_essay.strip():
                    st.error("请先输入作文内容再提交。")
                else:
                    is_task1 = writing_task_type == "Task 1"
                    spinner_msg = (
                        "👁️ Gemini 视觉引擎正在看图并批改..."
                        if is_task1
                        else "🧠 DeepSeek 推理引擎正在深度剖析大作文..."
                    )
                    with st.spinner(spinner_msg):
                        try:
                            evaluation = route_writing_evaluation(
                                writing_task_type,
                                selected_task["title"],
                                task_instructions,
                                selected_task.get("question_image"),
                                user_essay.strip(),
                            )
                            st.success("🎉 批改完成！")
                            st.markdown(
                                evaluation,
                                unsafe_allow_html=not is_task1,
                            )

                            supabase.table("writing_history").insert({
                                "username": current_user,
                                "task_id": task_id,
                                "user_essay": user_essay.strip(),
                                "evaluation": evaluation,
                            }).execute()
                            load_writing_history.clear()
                            st.balloons()
                        except Exception as e:
                            engine = "Gemini" if is_task1 else "DeepSeek"
                            st.error(f"{engine} 批改失败：{e}")


