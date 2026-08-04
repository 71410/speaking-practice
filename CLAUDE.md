# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

雅思英语训练舱 — 基于 Streamlit 的内部邀请制 Web 应用。**只做网页版**（Android 外壳与相关 CI 已于 2026-07 删除，不要再加回来）。

四个训练模块：模拟口语考官、我的素材朗读（用户自备素材 + 发音打分）、英文原版朗读纠音、雅思写作批改。

## 运行与部署

```bash
# 本地运行（需要 secrets.toml 配置 API Key）
streamlit run app.py

# 部署：推送 main 分支 → Streamlit Cloud 自动部署
git push
```

`.streamlit/secrets.toml` 包含所有密钥（API keys、Supabase 连接信息、用户密码），已在 `.gitignore` 中排除。Streamlit Cloud 上通过 Web UI 配置 secrets。

## 技术架构

**单体 Streamlit 应用** (`app.py`，约 2170 行)。前端 UI 和后端逻辑都在同一个文件中，Streamlit 的 session state 管理所有交互状态。

**三个 AI 引擎**：
- **DeepSeek**（OpenAI 兼容接口）：PDF 题目/文章提取、Task 2 写作批改、个性化参考答案
- **Gemini**（Google genai SDK，模型常量 `GEMINI_FLASH_MODEL = "gemini-2.5-flash"`）：语音点评、Task 1 写作图片识题与批改
- **NVIDIA Parakeet ASR**（REST，可选）：朗读的**客观测量**层，见下

### 为什么要有客观测量层（重要）

LLM 做发音评测是「听个大概然后写一段听起来专业的文字」。实测把**与原文完全一致**的
音频喂给 Gemini，它照样列出 6 处「错误」，而且「你读成」和「正确音标」两列是同一个
字符串 —— 纯粹为了填满表格而编。

我们有原文，所以这些可以**算**出来：
`transcribe_with_nvidia()` 取回转写 + 词级时间戳 → `measure_reading()` 做 diff，
得到漏读/多读/读错词、覆盖率、语速（词/分钟）、停顿位置与时长。
再由 `format_measurements_for_prompt()` 把这些硬数字注入 Prompt，明确禁止模型推翻。

接地之后，同一段读对的音频，编造出来的「错误」从 6 条降到 **0** 条。

边界：
- 该 REST 端点只返回 `word/start/end`，**没有 confidence**（canary 与 parakeet-ctc-riva
  也试过，拿不到）。所以「某个词发音含糊」仍然只能由 LLM 主观判断。
- **音素级**（「你把 /θ/ 读成了 /s/」）拿不到，需要 NeMo Forced Aligner 自部署
  或 Azure/Speechace 这类专门评测服务。
- ASR 对口音鲁棒，会把读偏的词「纠正」回正确拼写，所以词级判断偏**宽松**，只能抓明显错误。
- `NVIDIA_API_KEY` 没配时整层自动跳过，退回纯 Gemini 点评，不会报错。

**数据库**：Supabase（云端 PostgreSQL）。注意 `SUPABASE_KEY` 是 **service_role**，RLS 被完全绕过，**每个查询都必须自己带 `.eq("username", ...)` 做隔离**。

| 表 | 用途 |
|----|------|
| `question_bank` | 口语题库（part, theme, question_text）。**已超过 1000 行**，务必走 `fetch_all_rows()` |
| `reading_bank` | 朗读材料库（title, content），admin 上传，全站共用 |
| `writing_bank` | 写作题库（task_type, title, content, question_image 存 base64） |
| `speaking_materials` | **按用户隔离**的个人素材库（username, title, content） |
| `practice_history` / `reading_history` / `writing_history` / `material_history` | 各模块练习记录 |
| `profiles` | 用户个人档案 |

**用户认证**：简单的账号密码登录（`secrets.toml` 的 `[passwords]` 段），admin 账号有题库管理权限。

### Day1-7 配套音频（朗读纠音模块）

21 个 Magpie 合成的对话音频（约 100 MB）存在 Supabase Storage 的 **`day-audio`** public bucket，
URL 形如 `{SUPABASE_URL}/storage/v1/object/public/day-audio/day1-part1.mp3`。

句子级时间轴在 **`data/day_timeline.json`**（165 KB，随仓库走，运行时不查库也不调 ASR）。
点一句就播那一句靠的是它，`render_audio_reader()` 直接 seek 到 `start` 并在 `end` 停。

这份时间轴是**离线重建**出来的：音频当初逐段合成后拼接，中间过程没留 manifest，
所以用 Parakeet ASR 取词级时间戳，再和 docx 原文做 difflib 对齐反推每句的起止
（词匹配率 96.6%-99.5%，闭环抽查切片重新转写吻合度 100%）。
重建脚本不在本仓库，源文档改了需要重跑对齐。

坑：**每一天的主题标题都不一样**（Day1 是 `Work or studies`，Day2 是 `Hobby`……），
引导语分散在源目录各个 `generate_*.py` 里，不要写死白名单。

## 性能：缓存层（重要）

目的是减少每次 Streamlit 重跑时对 Supabase 的网络往返：

| 装饰器 | 用于 | TTL |
|--------|------|-----|
| `@st.cache_resource` | Supabase/DeepSeek/Gemini 客户端 | 永久 |
| `@st.cache_data` | `load_question_bank()` / `load_reading_bank()` / `load_writing_tasks()` / `count_rows()` | 5 分钟 |
| `@st.cache_data` | `load_profile_information()` / `load_speaking_materials()` / `speaking_tables_ready()` | 1 分钟 |
| `@st.cache_data` | `load_*_history()` | 30 秒 |
| `@st.cache_data` | `synthesize_tts_audio()` | 24 小时 |

所有数据写入操作（insert/update/delete）后必须调用对应缓存函数的 `.clear()`，否则用户看不到最新数据。已封装好 `clear_question_bank_caches()` / `clear_reading_bank_caches()` / `clear_writing_task_caches()`，新增写入时沿用同一套。

## 代码结构

`app.py` 大致按以下顺序组织（行号会漂移，用函数名搜索更可靠）：

1. **导入 + 配置**：secrets 读取、`@st.cache_resource` 客户端工厂、`NAV_PAGES`、`AUDIO_STATE_PREFIXES`、`TTS_ACCENTS`
2. **通用工具**：`fetch_all_rows()`、`count_rows()`、`release_inactive_audio()`、`split_sentences()`、`format_record_time()`
3. **录音评分通用流程**：`render_recording_practice()` —— 录音 → 提交 → Gemini 评分 → 存档，新的录音模块都应该复用它
4. **数据访问层**：各表的 `load_* / save_* / delete_*`
5. **AI Prompt**：`build_material_scoring_prompt()`、`DEEPSEEK_TASK2_SYSTEM_PROMPT`、`evaluate_writing_task1_gemini()`、`evaluate_writing_task2_deepseek()`
6. **认证**：`st.form("login_form")` 登录表单
7. **主界面路由**：`st.sidebar.radio` + admin 后台

## 踩过的坑（改代码前必读）

**Streamlit 相关**

- **widget 的 key 一旦实例化就不能再写**。要在代码里切换某个 radio/selectbox 的选中项，必须把请求先存到*另一个* key，在创建该 widget **之前**套用。项目里两处范例：`requested_page → current_page`、`material_source_request → material_source`。直接写会抛 `StreamlitAPIException`。
- **`st.text_area` 的内容要等失焦或 ⌘+Enter 才回传**。所以「文本框 + 根据内容 disable 按钮」是死局：按钮一直灰着，而灰按钮又接不到那次能让文本框失焦的点击。**凡是「填文本框然后点按钮」的交互一律用 `st.form` + `st.form_submit_button`**，提交时表单内所有控件一起回传。
- **录音必须手动回收**。录音是直接写进 `st.session_state` 的（不是 widget state，Streamlit 不会自动清），换题不清会让多段未压缩 WAV 常驻内存撑爆 Streamlit Cloud。新增录音模块时把 key 前缀登记进 `AUDIO_STATE_PREFIXES`，并调用 `release_inactive_audio()`。
- **退出登录用 `st.session_state.clear()`**。只清 `logged_in` 的话，同一浏览器换账号会看到上一个人的作文和录音。

**Supabase 相关**

- **`select()` 不分页会被静默截断**。Supabase 项目默认 `max-rows = 1000`，超了不报错、直接少给数据。拉全表一律走 `fetch_all_rows()`。
- **列表查询必须显式 `.order()`**。Postgres 无 ORDER BY 时顺序未定义，配 `.limit(5)` 实际拿到的是最旧的 5 条 —— 历史记录曾因此永远停在前 5 次。

**Prompt 相关**

- `DEEPSEEK_TASK2_SYSTEM_PROMPT` 是核心 Prompt（约 75 行），设定了严格的考官评分标准（中国大陆考生基准画像、四项评分铁律、反端水约束）。修改时需保持格式严谨，**务必确保 `"""` 正确闭合**。
- `evaluate_writing_task2_deepseek()` 里的 `user_prompt` 是 f-string（`f"""..."""`），末尾的 `"""` 容易在编辑时误删，历史上曾因此导致整个应用崩溃。

**其他**

- 录音以 bytes 直接经 `Part.from_bytes()` 送给 Gemini，不落磁盘。
- admin 的「一键清空题库」需要先输入 `DELETE` 才会启用按钮。
- 依赖已全部钉死版本；`pandas` 已移除（CSV 走标准库 `csv`），`PyPDF2` 已换成维护中的 `pypdf`。
- `.gitattributes` 强制 `eol=lf`。此前 `app.py` 是 CRLF/LF 混用，导致 diff 里整篇文件都显示为已修改。
