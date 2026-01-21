import os
import json
import mimetypes
import tempfile
import hashlib
from datetime import datetime
from typing import List, Dict, Any

import requests
import streamlit as st
import whisper
from gtts import gTTS


st.set_page_config(page_title="外语学习对话伙伴", page_icon="🗣️", layout="wide")


API_URL_DEFAULT = "https://zmgpt.cc"
MODEL_DEFAULT = "gpt-4o-mini"
API_KEY_DEFAULT = "sk-6qq5hf6VXQeYTDi65e9f4193DfAb404b8c1e2659234c1f18"

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, ".data")
HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")


SCENARIOS = {
	"餐厅点餐": "你和服务员在餐厅点餐，目标是自然礼貌地完成点餐与需求沟通。",
	"机场问路": "你在机场向工作人员问路并确认登机信息。",
	"商务会议": "你在商务会议中进行自我介绍、阐述观点并回应提问。",
	"酒店入住": "你在酒店前台办理入住并提出需求。",
	"购物退换": "你在商店沟通退换货并说明原因。",
}


LANG_OPTIONS = {
	"英语": "en",
	"日语": "ja",
	"韩语": "ko",
	"法语": "fr",
	"西班牙语": "es",
	"德语": "de",
	"意大利语": "it",
	"俄语": "ru",
	"中文": "zh",
}


LT_LANGUAGE = {
	"en": "en-US",
	"ja": "ja-JP",
	"ko": "ko-KR",
	"fr": "fr-FR",
	"es": "es",
	"de": "de-DE",
	"it": "it-IT",
	"ru": "ru-RU",
	"zh": "zh",
}


def _api_headers(api_key: str):
	return {
		"Authorization": f"Bearer {api_key}",
		"Content-Type": "application/json",
	}


def load_saved_sessions() -> List[Dict[str, Any]]:
	if not os.path.exists(HISTORY_FILE):
		return []
	try:
		with open(HISTORY_FILE, "r", encoding="utf-8") as f:
			data = json.load(f)
			return data if isinstance(data, list) else []
	except (OSError, json.JSONDecodeError):
		return []


def save_sessions(sessions: List[Dict[str, Any]]):
	os.makedirs(DATA_DIR, exist_ok=True)
	with open(HISTORY_FILE, "w", encoding="utf-8") as f:
		json.dump(sessions, f, ensure_ascii=False, indent=2)


def build_session_label(session: Dict[str, Any]) -> str:
	when = session.get("time", "")
	scenario = session.get("scenario", "未知场景")
	lang = session.get("target_lang_label", "")
	return f"{when} | {scenario} | {lang}"


def call_llm(api_base: str, api_key: str, model: str, messages: list, temperature: float = 0.4):
	url = f"{api_base.rstrip('/')}/v1/chat/completions"
	payload = {
		"model": model,
		"messages": messages,
		"temperature": temperature,
	}
	resp = requests.post(url, headers=_api_headers(api_key), data=json.dumps(payload), timeout=60)
	resp.raise_for_status()
	data = resp.json()
	return data["choices"][0]["message"]["content"]


def load_whisper_model(model_size: str = "base"):
	if "whisper_model" not in st.session_state or st.session_state.get("whisper_model_size") != model_size:
		st.session_state.whisper_model = whisper.load_model(model_size)
		st.session_state.whisper_model_size = model_size
	return st.session_state.whisper_model


def call_transcribe_local(file_bytes: bytes, filename: str, language: str, model_size: str = "base"):
	model = load_whisper_model(model_size)
	mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
	with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as tmp:
		tmp.write(file_bytes)
		tmp_path = tmp.name
	try:
		result = model.transcribe(tmp_path, language=language)
		return result.get("text", "")
	finally:
		try:
			os.remove(tmp_path)
		except OSError:
			pass


def call_tts_gtts(text: str, lang: str) -> bytes:
	with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
		tmp_path = tmp.name
	try:
		tts = gTTS(text=text, lang=lang)
		tts.save(tmp_path)
		with open(tmp_path, "rb") as f:
			return f.read()
	finally:
		try:
			os.remove(tmp_path)
		except OSError:
			pass


def language_tool_check(text: str, lang_code: str):
	url = "https://api.languagetool.org/v2/check"
	payload = {"text": text, "language": lang_code}
	resp = requests.post(url, data=payload, timeout=30)
	resp.raise_for_status()
	data = resp.json()
	matches = data.get("matches", [])
	issues = []
	for m in matches:
		issue = {
			"message": m.get("message"),
			"context": m.get("context", {}).get("text"),
			"offset": m.get("offset"),
			"length": m.get("length"),
			"replacements": [r.get("value") for r in m.get("replacements", [])][:5],
			"rule": m.get("rule", {}).get("description"),
		}
		issues.append(issue)
	return issues


def build_system_prompt(scenario: str, role: str, target_lang: str, difficulty: str):
	return (
		"你是专业外语对话伙伴与纠错老师。"
		f"场景：{scenario}。角色：{role}。"
		f"目标语言：{target_lang}。难度：{difficulty}。"
		"对话要求：以自然真实的语气推进情境，避免一次性输出过长。"
		"每轮回复应包含：继续对话的回复；"
		"不要给出评分。"
	)


def build_writing_prompt(target_lang: str, text: str):
	return (
		"你是写作批改老师。"
		f"目标语言：{target_lang}。"
		"任务：对输入文本进行纠错与表达优化，输出："
		"1) 纠错清单（原句 -> 修改后）；"
		"2) 优化建议（更自然的表达）；"
		"3) 文化背景提示（若相关）。"
		f"待批改文本：\n{text}"
	)


def build_feedback_prompt(target_lang: str, text: str):
	return (
		"你是语法与表达优化助手。"
		f"目标语言：{target_lang}。"
		"请针对用户输入给出："
		"1) 语法纠错（原句 -> 修改后）；"
		"2) 更自然的表达建议；"
		"3) 若涉及文化差异，给出简短提示。"
		f"用户输入：\n{text}"
	)


def render_issues(issues: list):
	if not issues:
		st.success("未发现明显语法问题。")
		return
	for i, issue in enumerate(issues, 1):
		st.markdown(f"**{i}. {issue['message']}**")
		if issue.get("context"):
			st.write(issue["context"])
		if issue.get("replacements"):
			st.write("替换建议：", ", ".join(issue["replacements"]))
		if issue.get("rule"):
			st.caption(issue["rule"])


if "messages" not in st.session_state:
	st.session_state.messages = []

if "history" not in st.session_state:
	st.session_state.history = []

if "last_feedback_input" not in st.session_state:
	st.session_state.last_feedback_input = None

if "last_feedback_text" not in st.session_state:
	st.session_state.last_feedback_text = None

if "last_feedback_issues" not in st.session_state:
	st.session_state.last_feedback_issues = None

if "saved_sessions" not in st.session_state:
	st.session_state.saved_sessions = load_saved_sessions()


st.title("🗣️ 外语学习对话伙伴")
st.caption("模拟真实情境对话，提供语法纠错、表达优化与文化提示。")


with st.sidebar:
	st.header("设置")
	api_base = st.text_input("API URL", value=os.getenv("API_URL", API_URL_DEFAULT))
	api_key = st.text_input("API Key", value=os.getenv("API_KEY", API_KEY_DEFAULT), type="password")
	model = st.text_input("模型", value=os.getenv("MODEL", MODEL_DEFAULT))
	mode = st.radio("练习模式", ["对话练习", "写作练习"], horizontal=True)
	scenario = st.selectbox("对话场景", list(SCENARIOS.keys()))
	role = st.text_input("你的角色", value="学习者")
	difficulty = st.selectbox("难度", ["初级", "中级", "高级"])
	target_lang_label = st.selectbox("目标语言", list(LANG_OPTIONS.keys()))
	whisper_model_size = st.selectbox("语音识别模型（本地 Whisper）", ["tiny", "base", "small", "medium"], index=1)
	st.caption("本地语音识别依赖 ffmpeg；若报错请先安装 ffmpeg 并加入 PATH。")
	enable_tts = st.checkbox("启用语音输出（TTS）", value=False)
	st.subheader("对话管理")
	if st.button("保存当前对话", use_container_width=True):
		if st.session_state.get("messages"):
			saved = st.session_state.get("saved_sessions", [])
			session_id = f"s_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(saved) + 1}"
			saved.append(
				{
					"id": session_id,
					"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
					"scenario": scenario,
					"role": role,
					"difficulty": difficulty,
					"target_lang_label": target_lang_label,
					"messages": st.session_state.get("messages", []),
					"history": st.session_state.get("history", []),
				}
			)
			st.session_state.saved_sessions = saved
			save_sessions(saved)
			st.success("已保存当前对话。")
		else:
			st.warning("当前没有对话可保存。")

	if st.button("清除当前对话", use_container_width=True):
		st.session_state.messages = []
		st.session_state.history = []
		st.session_state.pop("last_transcript", None)
		st.success("已清除当前对话。")

	saved_sessions = st.session_state.get("saved_sessions", [])
	if saved_sessions:
		session_map = {s.get("id"): s for s in saved_sessions}
		selected_id = st.selectbox(
			"历史对话",
			options=list(session_map.keys()),
			format_func=lambda sid: build_session_label(session_map.get(sid, {})),
		)
		if st.button("加载选中对话", use_container_width=True):
			selected = session_map.get(selected_id)
			if selected:
				st.session_state.messages = selected.get("messages", [])
				st.session_state.history = selected.get("history", [])
				st.session_state["loaded_session_meta"] = selected
				st.success("已加载历史对话。")
	else:
		st.info("暂无历史对话。")


if not api_key:
	st.warning("请在左侧填写 API Key。")
	st.stop()


target_lang = LANG_OPTIONS[target_lang_label]
lt_lang = LT_LANGUAGE.get(target_lang, target_lang)




col1, col2 = st.columns([2, 1])


with col1:
	st.subheader("对话 / 输入")

	if mode == "对话练习":
		for msg in st.session_state.messages:
			with st.chat_message(msg["role"]):
				st.markdown(msg["content"])

		with st.chat_message("assistant"):
			st.markdown(SCENARIOS[scenario])

		audio_record = st.audio_input("浏览器录音（直接录制）")
		audio_file = st.file_uploader("语音输入（可选，支持 wav/mp3/m4a）", type=["wav", "mp3", "m4a"])

		if audio_record:
			try:
				record_bytes = audio_record.getvalue() if hasattr(audio_record, "getvalue") else audio_record.read()
				audio_key = hashlib.md5(record_bytes).hexdigest()
				if st.session_state.get("last_audio_key") != audio_key:
					transcribed = call_transcribe_local(record_bytes, "recording.wav", target_lang, whisper_model_size)
					st.session_state["last_transcript"] = transcribed
					st.session_state["last_audio_key"] = audio_key
					st.success("已完成语音识别")
			except Exception as exc:
				st.error(f"语音识别失败：{exc}")

		if audio_file:
			try:
				file_bytes = audio_file.read()
				audio_key = hashlib.md5(file_bytes).hexdigest()
				if st.session_state.get("last_audio_key") != audio_key:
					transcribed = call_transcribe_local(file_bytes, audio_file.name, target_lang, whisper_model_size)
					st.session_state["last_transcript"] = transcribed
					st.session_state["last_audio_key"] = audio_key
					st.success("已完成语音识别")
			except Exception as exc:
				st.error(f"语音识别失败：{exc}")

		if "last_transcript" in st.session_state:
			st.info(st.session_state["last_transcript"])
			send_transcript = st.button("发送识别结果", use_container_width=True)
		else:
			send_transcript = False

		user_text = st.chat_input("请输入你的回复或上传语音")
		if send_transcript and st.session_state.get("last_transcript"):
			user_text = st.session_state.get("last_transcript")

		if user_text:
			st.session_state.messages.append({"role": "user", "content": user_text})

			system_prompt = build_system_prompt(SCENARIOS[scenario], role, target_lang_label, difficulty)
			messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

			with st.chat_message("assistant"):
				with st.spinner("生成回复中..."):
					try:
						reply = call_llm(api_base, api_key, model, messages)
					except Exception as exc:
						st.error(f"调用模型失败：{exc}")
						reply = None
				if reply:
					st.markdown(reply)
					st.session_state.messages.append({"role": "assistant", "content": reply})
					tts_text = str(reply)
					st.session_state["last_tts_text"] = tts_text

					if enable_tts:
						try:
							audio_bytes = call_tts_gtts(tts_text, target_lang)
							st.session_state["last_tts_audio"] = audio_bytes
							st.audio(audio_bytes, format="audio/mp3")
						except Exception as exc:
							st.error(f"语音合成失败：{exc}")

			st.session_state.history.append({"time": datetime.now().isoformat(), "input": user_text})

	else:
		writing_text = st.text_area("请输入待批改文本", height=220)
		if st.button("开始批改") and writing_text.strip():
			st.session_state["writing_text"] = writing_text


with col2:
	st.subheader("纠错与建议")

	if mode == "对话练习":
		if st.session_state.messages:
			last_user = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
			if last_user:
				current_input = last_user["content"]
				if st.session_state.last_feedback_input != current_input:
					st.session_state.last_feedback_input = current_input
					st.session_state.last_feedback_text = None
					st.session_state.last_feedback_issues = None

				st.markdown("**语法检查（LanguageTool）**")
				if st.session_state.last_feedback_issues is None:
					try:
						issues = language_tool_check(current_input, lt_lang)
						st.session_state.last_feedback_issues = issues
					except Exception as exc:
						st.error(f"语法检查失败：{exc}")
				if st.session_state.last_feedback_issues is not None:
					render_issues(st.session_state.last_feedback_issues)

				st.markdown("**表达优化与文化提示（大模型）**")
				if st.session_state.last_feedback_text is None:
					with st.spinner("生成建议中..."):
						try:
							feedback = call_llm(
								api_base,
								api_key,
								model,
								[{"role": "system", "content": build_feedback_prompt(target_lang_label, current_input)}],
								temperature=0.2,
							)
							st.session_state.last_feedback_text = feedback
						except Exception as exc:
							st.error(f"生成建议失败：{exc}")
				if st.session_state.last_feedback_text is not None:
					st.markdown(st.session_state.last_feedback_text)
		else:
			st.info("开始对话后会显示纠错与建议。")

	else:
		writing_text = st.session_state.get("writing_text", "")
		if writing_text:
			st.markdown("**语法检查（LanguageTool）**")
			try:
				issues = language_tool_check(writing_text, lt_lang)
				render_issues(issues)
			except Exception as exc:
				st.error(f"语法检查失败：{exc}")

			st.markdown("**写作批改（大模型）**")
			with st.spinner("批改中..."):
				try:
					review = call_llm(
						api_base,
						api_key,
						model,
						[{"role": "system", "content": build_writing_prompt(target_lang_label, writing_text)}],
						temperature=0.2,
					)
					st.markdown(review)
				except Exception as exc:
					st.error(f"批改失败：{exc}")
		else:
			st.info("输入文本后点击开始批改。")
