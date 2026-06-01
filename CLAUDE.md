# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

雅思英语训练舱 — 基于 Streamlit 的内部邀请制 Web 应用。提供三大训练模块：模拟口语考官、英文朗读纠音、雅思写作批改。

## 运行与部署

```bash
# 本地运行（需要 secrets.toml 配置 API Key）
streamlit run app.py

# 部署：推送 main 分支 → Streamlit Cloud 自动部署
git push
```

`.streamlit/secrets.toml` 包含所有密钥（API keys、Supabase 连接信息、用户密码），已在 `.gitignore` 中排除。Streamlit Cloud 上通过 Web UI 配置 secrets。

## 技术架构

**单体 Streamlit 应用** (`app.py`，约 1200 行)。前端 UI 和后端逻辑都在同一个文件中，Streamlit 的 session state 管理所有交互状态。

**双 AI 引擎**：
- **DeepSeek**（OpenAI 兼容接口）：文本解析、Task 2 写作批改（严格考官 Prompt）、PDF 题目提取
- **Gemini**（Google genai SDK）：语音评分（`gemini-3.5-flash`）、Task 1 写作图片识题、写作批改

**数据库**：Supabase（云端 PostgreSQL），四张表：
- `question_bank` — 口语题库（part, theme, question_text）
- `reading_bank` — 朗读材料库（title, content）
- `writing_bank` — 写作题库（task_type, title, content, question_image）
- `practice_history` / `reading_history` / `writing_history` — 练习记录
- `profiles` — 用户个人档案

**用户认证**：简单的账号密码登录（`secrets.toml` 的 `[passwords]` 段），admin 账号有题库管理权限。

## 性能：缓存层（重要）

2026-06 添加，目的是减少每次 Streamlit 重跑时对 Supabase 的网络往返：

| 装饰器 | 用于 | TTL |
|--------|------|-----|
| `@st.cache_resource` | Supabase/DeepSeek/Gemini 客户端 | 永久 |
| `@st.cache_data` | `load_question_bank()` / `load_reading_bank()` / `load_writing_tasks()` | 5 分钟 |
| `@st.cache_data` | `load_profile_information()` | 1 分钟 |
| `@st.cache_data` | `load_*_history()` | 30 秒 |

所有数据写入操作（insert/update/delete）后必须调用对应缓存函数的 `.clear()`，否则用户看不到最新数据。修改代码时若新增数据库查询，应遵循相同缓存模式。

## 代码结构

`app.py` 按以下顺序组织：

1. **导入 + 配置**（1-40 行）：secrets 读取、客户端创建、缓存工厂函数
2. **数据访问层**（40-300 行）：Supabase 查询函数（均有 `@st.cache_data`）、用户档案、写作题库
3. **AI Prompt 与调用**（300-550 行）：`DEEPSEEK_TASK2_SYSTEM_PROMPT`、Gemini 评分 Prompt、`generate_personalized_answer()`、`evaluate_writing_task1_gemini()`、`evaluate_writing_task2_deepseek()`
4. **认证**（550-580 行）：登录表单
5. **主界面路由**（580+ 行）：`st.sidebar.radio` 四个页面 + admin 后台

## 注意事项

- `DEEPSEEK_TASK2_SYSTEM_PROMPT` 是核心 Prompt（约 75 行），设定了严格的考官评分标准（中国大陆考生基准画像、四项评分铁律、反端水约束）。修改时需保持格式严谨，**务必确保 `"""` 正确闭合**。
- `evaluate_writing_task2_deepseek()` 里的 `user_prompt` 是 f-string（`f"""..."""`），末尾的 `"""` 容易在编辑时误删，历史上曾因此导致整个应用崩溃。
- Gemini 语音文件用 `tempfile` 临时存储，送完后立即删除。如需持久化录音应考虑对象存储。
- admin 账号有"一键清空题库"按钮，无二次确认，慎用。
