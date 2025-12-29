import os
import json
import requests
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date, datetime
import numpy as np

# =========================================
# 0) ENV + AI (boleh sebelum parse_date)
# =========================================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "google/gemini-2.0-flash-lite-preview-02-05:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
]


def ask_llm_api(messages: List[Dict[str, str]], temperature: float = 0.6, timeout: int = 25) -> str:
    """
    OpenRouter fallback chain (FREE models).
    """
    if not OPENROUTER_API_KEY:
        return "ERROR: OPENROUTER_API_KEY not set. Put it in backend/.env"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # untuk dev local: boleh localhost
        "HTTP-Referer": "http://localhost",
        "X-Title": "LunaCycle Biomedical Assistant",
    }

    last_err = None
    for model_id in FREE_MODELS:
        try:
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
            }
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                return data["choices"][0]["message"]["content"]
            last_err = f"{model_id} -> {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = f"{model_id} -> exception: {str(e)}"

    return f"Maaf, semua server AI sedang sibuk. Detail: {last_err}"


def extract_json(text: str):
    """Ambil objek JSON pertama dari output LLM."""
    if not text:
        raise ValueError("Empty LLM output")

    s = text.find("{")
    e = text.rfind("}") + 1
    if s == -1 or e <= 0:
        raise ValueError("No JSON object found in LLM output")

    return json.loads(text[s:e])

def ask_openrouter_json(prompt: str):
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set (check .env or environment variable).")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://127.0.0.1:5500",
        "X-Title": "LunaCycle",
    }

    sys = (
        "You are a biomedical expert. "
        "Return ONLY valid JSON. No markdown, no explanations outside JSON."
    )

    for model_id in FREE_MODELS:
        try:
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.6
            }

            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=25)
            if r.status_code != 200:
                # coba model berikutnya
                continue

            data = r.json()
            text = data["choices"][0]["message"]["content"]
            return extract_json(text)

        except Exception:
            continue

    raise RuntimeError("All OpenRouter models failed or returned invalid JSON.")

# =========================================
# 1) Utilities (baru parse_date dst)
# =========================================
def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def gaussian(x, mu, sigma, amp=1.0):
    x = np.asarray(x)
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def compute_phase_ranges(cycle_length: int, period_length: int):
    luteal_len = 14
    if cycle_length < (period_length + luteal_len + 3):
        luteal_len = max(10, cycle_length - (period_length + 3))

    ovulation_day = cycle_length - luteal_len
    ovu_start = max(period_length + 1, ovulation_day - 1)
    ovu_end = min(cycle_length, ovulation_day + 1)

    menstrual = [1, period_length]
    follicular = [period_length + 1, max(period_length + 1, ovu_start - 1)]
    ovulation = [ovu_start, ovu_end]
    luteal = [min(cycle_length, ovu_end + 1), cycle_length]

    return {
        "menstrual": menstrual,
        "follicular": follicular,
        "ovulation": ovulation,
        "luteal": luteal,
        "ovulation_day": ovulation_day
    }

def current_phase(day_in_cycle: int, phase_ranges: dict) -> str:
    d = day_in_cycle
    if phase_ranges["menstrual"][0] <= d <= phase_ranges["menstrual"][1]:
        return "Menstrual"
    if phase_ranges["follicular"][0] <= d <= phase_ranges["follicular"][1]:
        return "Follicular"
    if phase_ranges["ovulation"][0] <= d <= phase_ranges["ovulation"][1]:
        return "Ovulation"
    if phase_ranges["luteal"][0] <= d <= phase_ranges["luteal"][1]:
        return "Luteal"
    return "Unknown"

def generate_curves(days, cycle_length: int, ovulation_day: int):
    d = np.asarray(days)

    ov_prob = gaussian(d, ovulation_day, sigma=2.2, amp=1.0)
    ov_prob = np.clip(ov_prob, 0, 1)

    e2 = (
        40
        + gaussian(d, ovulation_day - 1, sigma=2.0, amp=250)
        + gaussian(d, ovulation_day + 7, sigma=3.0, amp=130)
    )

    p4 = (0.3 + gaussian(d, ovulation_day + 7, sigma=3.2, amp=18))

    lh = 5 + gaussian(d, ovulation_day, sigma=0.9, amp=75)

    fsh = 6 + gaussian(d, 2, sigma=1.3, amp=6) + gaussian(d, ovulation_day, sigma=1.2, amp=14)

    return {
        "ovulation_prob": ov_prob.tolist(),
        "hormones": {
            "e2": e2.tolist(),
            "p4": p4.tolist(),
            "lh": lh.tolist(),
            "fsh": fsh.tolist(),
        }
    }


# =========================================
# 2) API Schemas
# =========================================
class PredictRequest(BaseModel):
    last_period_prev: str = Field(..., description="YYYY-MM-DD")
    last_period_curr: str = Field(..., description="YYYY-MM-DD")
    today_date: Optional[str] = Field(None, description="YYYY-MM-DD optional")
    age: int = 24
    weight: float = 55.0
    height: float = 160.0
    period_length: Optional[int] = Field(None, description="optional; default 5")

class AIInsightRequest(BaseModel):
    cycle_length: int
    today_day: int
    current_phase: str
    age: int
    bmi: float
    is_irregular: bool

class ChatRequest(BaseModel):
    # histori chat dari frontend: [{"role":"user","content":"..."}, ...]
    messages: List[Dict[str, str]]
    # konteks biomedis biar chatbot nyambung
    context: Dict[str, Any]


# =========================================
# 3) FastAPI app
# =========================================
app = FastAPI(title="LunaCycle API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        prev = parse_date(req.last_period_prev)
        curr = parse_date(req.last_period_curr)
        today = parse_date(req.today_date) if req.today_date else date.today()

        cycle_length = clamp_int((curr - prev).days, 20, 60)

        period_length = req.period_length if req.period_length else 5
        period_length = clamp_int(int(period_length), 3, 10)

        day_in_cycle = clamp_int((today - curr).days + 1, 1, cycle_length)

        h_m = req.height / 100.0
        bmi = (req.weight / (h_m * h_m)) if (req.weight > 0 and h_m > 0) else 0.0
        bmi = float(np.round(bmi, 2))

        pr = compute_phase_ranges(cycle_length, period_length)
        ovulation_day = pr["ovulation_day"]

        days = list(range(1, cycle_length + 1))
        curves = generate_curves(days, cycle_length, ovulation_day)
        phase_now = current_phase(day_in_cycle, pr)

        return {
            "cycle_length": cycle_length,
            "period_length": period_length,
            "age": req.age,
            "bmi": bmi,
            "today_day": day_in_cycle,
            "ovulation_day": ovulation_day,
            "current_phase": phase_now,
            "days": days,
            "phase_ranges": {
                "menstrual": pr["menstrual"],
                "follicular": pr["follicular"],
                "ovulation": pr["ovulation"],
                "luteal": pr["luteal"],
            },
            "ovulation_prob": curves["ovulation_prob"],
            "hormones": curves["hormones"],
        }

    except Exception as e:
        return {"error": str(e)}

@app.post("/ai/insight")
def ai_insight(req: AIInsightRequest):
    try:
        # prompt mirip Streamlit kamu
        prompt = f"""
Berdasarkan data biomedis berikut (JSON):
{json.dumps(req.dict(), ensure_ascii=False)}

Kembalikan HANYA dalam format JSON:
{{
  "health_score": (int 0-100),
  "fertility_score": (int 0-100),
  "rationale": "penjelasan singkat dalam Bahasa Indonesia"
}}
        """.strip()

        out = ask_openrouter_json(prompt)

        # validasi minimal
        hs = int(out.get("health_score", 0))
        fs = int(out.get("fertility_score", 0))
        rationale = str(out.get("rationale", "")).strip() or "—"

        # clamp
        hs = max(0, min(100, hs))
        fs = max(0, min(100, fs))

        return {
            "health_score": hs,
            "fertility_score": fs,
            "rationale": rationale,
        }

    except Exception as e:
        # ini yg bikin kamu lihat 0 semua
        return {
            "health_score": 0,
            "fertility_score": 0,
            "rationale": f"Gagal memproses AI. {str(e)}"
        }

@app.post("/ai/chat")
def ai_chat(req: ChatRequest):
    """
    Chatbot: pakai context biomedis + history chat.
    """
    ctx = req.context or {}
    cycle_len = ctx.get("cycle_length", "N/A")
    today_day = ctx.get("today_day", "N/A")
    phase = ctx.get("current_phase", "N/A")

    sys_msg = (
        "Kamu adalah LunaCycle Assistant. "
        "Jawab dengan ramah, empatik, edukatif, berbasis sains. "
        "Hindari diagnosis, gunakan disclaimer ringan bila perlu. "
        f"Data user saat ini: siklus {cycle_len} hari, hari ke-{today_day}, fase {phase}."
    )

    history = req.messages or []
    messages = [{"role": "system", "content": sys_msg}] + history

    text = ask_llm_api(messages, temperature=0.6)
    return {"reply": text}
