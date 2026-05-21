"""
AI Crop Diseases Detection & Treatment Recommendation System - Backend
"""
import os
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Literal

import bcrypt
import jwt
import httpx
import time
from fastapi import FastAPI, APIRouter, HTTPException, Depends, Header
from fastapi.security import HTTPBearer
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, EmailStr

from emergentintegrations.llm.chat import LlmChat, UserMessage, ImageContent, llm_configured
import database as db

# ---------------- Setup ----------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

# MySQL (replaces MongoDB)
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.environ.get("MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYSQL_DATABASE", "cropdoctor")

EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")
JWT_SECRET = os.environ.get("JWT_SECRET", "fallback-secret-change-me")
JWT_ALG = "HS256"
JWT_DAYS = 30

api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cropdoctor")


# ---------------- Models ----------------
Role = Literal["farmer", "expert", "admin"]


class User(BaseModel):
    user_id: str
    email: str
    name: str
    phone: Optional[str] = None
    role: Role = "farmer"
    language: str = "en"
    picture: Optional[str] = None
    location: Optional[str] = None
    created_at: datetime


class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    name: str
    phone: Optional[str] = None
    role: Role = "farmer"
    language: str = "en"
    location: Optional[str] = None


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class AuthOut(BaseModel):
    token: str
    user: User


class GoogleSessionIn(BaseModel):
    session_id: str


class ScanIn(BaseModel):
    image_base64: str
    crop_name: Optional[str] = None
    notes: Optional[str] = None
    model: str = "gemini"  # "gemini" or "openai"


class TreatmentItem(BaseModel):
    type: str  # pesticide / organic / traditional
    name: str
    dosage: str
    application: str
    safety: str
    cost_estimate: str


class DiseaseScan(BaseModel):
    scan_id: str
    user_id: str
    crop_name: Optional[str] = None
    image_base64: str
    disease_name: str
    is_healthy: bool
    severity: str  # healthy / mild / moderate / severe
    confidence: float
    symptoms: str
    causes: str
    affected_part: str
    spread_risk: str
    next_actions: List[str]
    treatments: List[TreatmentItem]
    preventive_measures: List[str]
    notes: Optional[str] = None
    created_at: datetime


class ChatMsgIn(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatMsg(BaseModel):
    role: str
    content: str
    timestamp: datetime


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    language: Optional[str] = None
    location: Optional[str] = None


# ---------------- Auth helpers ----------------
def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_pw(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


async def _seed_users() -> None:
    seeds = [
        {
            "user_id": "user_admin_seed",
            "email": "admin@cropdoctor.ai",
            "name": "Admin User",
            "phone": "+250788000001",
            "role": "admin",
            "language": "en",
            "location": "Kigali",
            "picture": None,
            "password": hash_pw("Admin@123"),
            "created_at": datetime.now(timezone.utc),
        },
        {
            "user_id": "user_expert_seed",
            "email": "expert@cropdoctor.ai",
            "name": "Dr. Expert",
            "phone": "+250788000002",
            "role": "expert",
            "language": "en",
            "location": "Kigali",
            "picture": None,
            "password": hash_pw("Expert@123"),
            "created_at": datetime.now(timezone.utc),
        },
        {
            "user_id": "user_farmer_seed",
            "email": "farmer@cropdoctor.ai",
            "name": "Test Farmer",
            "phone": "+250788000003",
            "role": "farmer",
            "language": "en",
            "location": "Musanze",
            "picture": None,
            "password": hash_pw("Farmer@123"),
            "created_at": datetime.now(timezone.utc),
        },
    ]
    for doc in seeds:
        if not await db.get_user_by_email(doc["email"]):
            await db.create_user(doc)
    logger.info("Seed users ensured.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db.init_pool()
    except Exception as e:
        logger.error(
            "MySQL connection failed (%s@%s:%s/%s): %s. "
            "Start MySQL and check MYSQL_* vars in backend/.env",
            MYSQL_USER,
            MYSQL_HOST,
            MYSQL_PORT,
            MYSQL_DATABASE,
            e,
        )
        raise
    try:
        await _seed_users()
    except Exception:
        logger.exception("Seed failed")
    yield
    await db.close_pool()


app = FastAPI(title="CropDoctor AI", lifespan=lifespan)


def make_jwt(user_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "exp": now + (JWT_DAYS * 24 * 60 * 60),
        "iat": now,
    }
    key = jwt.jwk.OctetJWK(JWT_SECRET.encode() if isinstance(JWT_SECRET, str) else JWT_SECRET)
    jwt_instance = jwt.JWT()
    return jwt_instance.encode(payload, key, alg=JWT_ALG)


async def get_current_user(authorization: Optional[str] = Header(None)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Not authenticated")
    token = authorization.split(" ", 1)[1].strip()

    user_id: Optional[str] = None
    # 1) Try JWT
    try:
        key = jwt.jwk.OctetJWK(JWT_SECRET.encode() if isinstance(JWT_SECRET, str) else JWT_SECRET)
        jwt_instance = jwt.JWT()
        payload = jwt_instance.decode(token, key, algorithms=[JWT_ALG])
        user_id = payload.get("sub")
    except Exception:
        user_id = None

    # 2) Fallback: Emergent session_token
    if not user_id:
        sess = await db.get_session(token)
        if not sess:
            raise HTTPException(401, "Invalid or expired token")
        exp = sess.get("expires_at")
        if exp and exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if exp and exp < datetime.now(timezone.utc):
            raise HTTPException(401, "Session expired")
        user_id = sess["user_id"]

    user_doc = await db.get_user_by_id(user_id)
    if not user_doc:
        raise HTTPException(401, "User not found")
    return User(**user_doc)


# ---------------- Auth endpoints ----------------
@api_router.post("/auth/register", response_model=AuthOut)
async def register(payload: RegisterIn):
    existing = await db.get_user_by_email(payload.email.lower())
    if existing:
        raise HTTPException(400, "Email already registered")
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    doc = {
        "user_id": user_id,
        "email": payload.email.lower(),
        "name": payload.name,
        "phone": payload.phone,
        "role": payload.role,
        "language": payload.language,
        "location": payload.location,
        "picture": None,
        "password": hash_pw(payload.password),
        "created_at": datetime.now(timezone.utc),
    }
    await db.create_user(doc)
    user = User(**{k: v for k, v in doc.items() if k != "password"})
    return AuthOut(token=make_jwt(user_id), user=user)


@api_router.post("/auth/login", response_model=AuthOut)
async def login(payload: LoginIn):
    doc = await db.get_user_by_email(payload.email.lower(), include_password=True)
    if not doc or "password" not in doc or not verify_pw(payload.password, doc["password"]):
        raise HTTPException(401, "Invalid email or password")
    user = User(**{k: v for k, v in doc.items() if k != "password"})
    return AuthOut(token=make_jwt(doc["user_id"]), user=user)


@api_router.post("/auth/google/session", response_model=AuthOut)
async def google_session(payload: GoogleSessionIn):
    """Exchange Emergent Google session_id for our app token."""
    async with httpx.AsyncClient(timeout=20) as cli:
        r = await cli.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": payload.session_id},
        )
    if r.status_code != 200:
        raise HTTPException(401, f"Invalid session: {r.text}")
    data = r.json()
    email = (data.get("email") or "").lower()
    name = data.get("name") or "Farmer"
    picture = data.get("picture")
    session_token = data.get("session_token")

    user_doc = await db.get_user_by_email(email)
    if not user_doc:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user_doc = {
            "user_id": user_id,
            "email": email,
            "name": name,
            "phone": None,
            "role": "farmer",
            "language": "en",
            "location": None,
            "picture": picture,
            "created_at": datetime.now(timezone.utc),
        }
        await db.create_user(user_doc)

    await db.upsert_session(
        session_token,
        user_doc["user_id"],
        datetime.now(timezone.utc) + timedelta(days=7),
        datetime.now(timezone.utc),
    )
    user = User(**{k: v for k, v in user_doc.items() if k != "password"})
    return AuthOut(token=session_token, user=user)


@api_router.get("/auth/me", response_model=User)
async def me(current: User = Depends(get_current_user)):
    return current


@api_router.patch("/auth/me", response_model=User)
async def update_me(payload: ProfileUpdate, current: User = Depends(get_current_user)):
    upd = {k: v for k, v in payload.model_dump().items() if v is not None}
    if upd:
        await db.update_user(current.user_id, upd)
    doc = await db.get_user_by_id(current.user_id)
    return User(**doc)


@api_router.post("/auth/logout")
async def logout(current: User = Depends(get_current_user)):
    return {"ok": True}


# ---------------- AI Disease Detection ----------------
DISEASE_PROMPT = """You are an expert agricultural plant pathologist analyzing a crop image.

Analyze the image and respond ONLY with a JSON object (no markdown, no explanation) with this exact schema:

{
  "is_healthy": boolean,
  "disease_name": "string (e.g., 'Late Blight', 'Healthy Crop')",
  "severity": "healthy | mild | moderate | severe",
  "confidence": number 0-100,
  "symptoms": "string - visible symptoms 1-2 sentences",
  "causes": "string - main causes 1-2 sentences",
  "affected_part": "string - leaves/stem/fruit/root etc",
  "spread_risk": "low | medium | high | very_high",
  "next_actions": ["3-5 short action items"],
  "treatments": [
    {"type": "pesticide|organic|traditional", "name": "string",
     "dosage": "string", "application": "string",
     "safety": "string", "cost_estimate": "Low|Medium|High"}
  ],
  "preventive_measures": ["3-4 short tips"]
}

If the image is not a crop or unclear, set is_healthy=false, disease_name="Unable to analyze", severity="healthy", confidence=0.
Always provide AT LEAST 2 treatments (1 organic + 1 chemical) when a disease is detected. Use Rwandan/East African context where possible.
"""


def _parse_json_loose(text: str) -> dict:
    import json, re

    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Find first JSON object
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    return json.loads(text)


async def analyze_image_with_ai(image_base64: str, model_choice: str) -> dict:
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gemini_key:
        logger.error("GEMINI_API_KEY not configured in backend/.env")
        raise HTTPException(
            500,
            "AI service not configured. Please add a valid GEMINI_API_KEY to backend/.env",
        )

    chat = LlmChat(
        api_key=gemini_key,
        session_id=f"scan_{uuid.uuid4().hex[:8]}",
        system_message=DISEASE_PROMPT,
    ).with_model("gemini", "gemini-flash-latest")

    img = ImageContent(image_base64=image_base64)
    msg = UserMessage(text="Analyze this crop image and return strict JSON only.", file_contents=[img])

    try:
        resp = await chat.send_message(msg)
        logger.info("AI analysis completed successfully")
        # Parse the JSON response
        try:
            data = _parse_json_loose(resp)
        except Exception as parse_err:
            logger.warning(f"JSON parse failed: {parse_err}, raw response: {resp[:300]}")
            raise ValueError(f"Invalid AI response format: {str(parse_err)[:100]}")
        return data
    except ValueError as e:
        error_msg = str(e)
        logger.error(f"Gemini API error: {error_msg}")
        
        # Parse error details for better user messages
        if "503" in error_msg or "Service Unavailable" in error_msg or "high demand" in error_msg:
            raise HTTPException(
                503,
                "AI service is busy. Please try again in a few seconds.",
            )
        elif "429" in error_msg or "Rate limit" in error_msg:
            raise HTTPException(
                429,
                "Too many requests. Please try again in a moment.",
            )
        elif "401" in error_msg or "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
            raise HTTPException(
                500,
                "API authentication failed. Check GEMINI_API_KEY in backend/.env",
            )
        else:
            raise HTTPException(502, f"AI service error: {error_msg[:150]}")
    except Exception as e:
        logger.exception("Unexpected error during AI analysis")
        raise HTTPException(502, f"Analysis failed: {str(e)[:150]}")


@api_router.post("/scans", response_model=DiseaseScan)
async def create_scan(payload: ScanIn, current: User = Depends(get_current_user)):
    ai_data = await analyze_image_with_ai(payload.image_base64, payload.model)
    scan = DiseaseScan(
        scan_id=f"scan_{uuid.uuid4().hex[:12]}",
        user_id=current.user_id,
        crop_name=payload.crop_name,
        image_base64=payload.image_base64,
        disease_name=ai_data.get("disease_name", "Unknown"),
        is_healthy=bool(ai_data.get("is_healthy", False)),
        severity=ai_data.get("severity", "healthy"),
        confidence=float(ai_data.get("confidence", 0)),
        symptoms=ai_data.get("symptoms", ""),
        causes=ai_data.get("causes", ""),
        affected_part=ai_data.get("affected_part", ""),
        spread_risk=ai_data.get("spread_risk", "low"),
        next_actions=ai_data.get("next_actions", []) or [],
        treatments=[TreatmentItem(**t) for t in ai_data.get("treatments", []) or [] if isinstance(t, dict)],
        preventive_measures=ai_data.get("preventive_measures", []) or [],
        notes=payload.notes,
        created_at=datetime.now(timezone.utc),
    )
    await db.insert_scan(scan.model_dump())
    return scan


@api_router.get("/scans", response_model=List[DiseaseScan])
async def list_scans(current: User = Depends(get_current_user), limit: int = 50):
    docs = await db.list_scans(current.user_id, limit)
    return [DiseaseScan(**d) for d in docs]


@api_router.get("/scans/{scan_id}", response_model=DiseaseScan)
async def get_scan(scan_id: str, current: User = Depends(get_current_user)):
    doc = await db.get_scan(scan_id, current.user_id)
    if not doc:
        raise HTTPException(404, "Scan not found")
    return DiseaseScan(**doc)


@api_router.delete("/scans/{scan_id}")
async def delete_scan(scan_id: str, current: User = Depends(get_current_user)):
    deleted = await db.delete_scan(scan_id, current.user_id)
    return {"deleted": deleted}


# ---------------- Dashboard / Stats ----------------
@api_router.get("/dashboard")
async def dashboard(current: User = Depends(get_current_user)):
    scans = await db.list_scans_for_user(current.user_id, 200)
    total = len(scans)
    healthy = sum(1 for s in scans if s.get("is_healthy"))
    diseased = total - healthy
    # severity counts
    sev_counts = {"healthy": 0, "mild": 0, "moderate": 0, "severe": 0}
    for s in scans:
        sev_counts[s.get("severity", "healthy")] = sev_counts.get(s.get("severity", "healthy"), 0) + 1
    # recent
    recent = [
        {
            "scan_id": s["scan_id"],
            "disease_name": s["disease_name"],
            "severity": s["severity"],
            "confidence": s["confidence"],
            "is_healthy": s["is_healthy"],
            "crop_name": s.get("crop_name"),
            "created_at": s["created_at"],
        }
        for s in scans[:5]
    ]
    # most common diseases
    counts: dict = {}
    for s in scans:
        if not s.get("is_healthy"):
            counts[s["disease_name"]] = counts.get(s["disease_name"], 0) + 1
    top_diseases = sorted(counts.items(), key=lambda x: -x[1])[:5]

    return {
        "total_scans": total,
        "healthy_scans": healthy,
        "diseased_scans": diseased,
        "health_score": int((healthy / total * 100) if total else 100),
        "severity_breakdown": sev_counts,
        "recent_scans": recent,
        "top_diseases": [{"name": n, "count": c} for n, c in top_diseases],
    }


# ---------------- Weather (mock) ----------------
@api_router.get("/weather")
async def weather(current: User = Depends(get_current_user)):
    import random

    today = datetime.now(timezone.utc)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    forecast = []
    for i in range(7):
        d = today + timedelta(days=i)
        forecast.append(
            {
                "day": days[d.weekday()],
                "date": d.strftime("%b %d"),
                "temp_high": random.randint(24, 30),
                "temp_low": random.randint(15, 20),
                "condition": random.choice(["sunny", "cloudy", "rainy", "partly_cloudy"]),
                "humidity": random.randint(55, 85),
                "rainfall_mm": random.randint(0, 25),
            }
        )
    alerts = [
        {
            "id": "alert_1",
            "type": "disease_outbreak",
            "severity": "moderate",
            "title": "High humidity disease risk",
            "message": "Humidity levels favor fungal diseases. Inspect crops daily.",
            "created_at": today.isoformat(),
        },
        {
            "id": "alert_2",
            "type": "rainfall",
            "severity": "info",
            "title": "Heavy rain expected",
            "message": "20-30mm rainfall expected in 2 days. Delay pesticide application.",
            "created_at": today.isoformat(),
        },
    ]
    return {
        "location": current.location or "Kigali, Rwanda",
        "current": {
            "temp": 25,
            "humidity": 72,
            "condition": "partly_cloudy",
            "wind_kmh": 12,
            "rainfall_mm": 0,
        },
        "forecast": forecast,
        "alerts": alerts,
    }


# ---------------- Knowledge Base ----------------
KNOWLEDGE_SEED = [
    {
        "id": "kb_late_blight",
        "category": "disease",
        "title": "Late Blight (Phytophthora infestans)",
        "summary": "Devastating fungal disease affecting potatoes and tomatoes.",
        "crops": ["Potato", "Tomato"],
        "symptoms": "Dark water-soaked spots on leaves, white fuzzy growth on undersides, rapid wilting.",
        "prevention": "Plant resistant varieties; avoid overhead watering; ensure airflow; apply copper fungicides preventively.",
        "treatment": "Remove infected plants; apply chlorothalonil or mancozeb fungicides; rotate crops.",
        "image": "https://images.unsplash.com/photo-1758414083942-bc9f82e026b0",
    },
    {
        "id": "kb_maize_streak",
        "category": "disease",
        "title": "Maize Streak Virus",
        "summary": "Common viral disease in East African maize, spread by leafhoppers.",
        "crops": ["Maize"],
        "symptoms": "Pale yellow streaks on leaves running parallel to veins; stunted growth.",
        "prevention": "Plant resistant hybrids; control leafhopper vectors; remove infected plants.",
        "treatment": "No cure; remove infected plants. Spray insecticides for leafhopper control.",
        "image": "https://images.unsplash.com/photo-1738826588286-edda056cfb8f",
    },
    {
        "id": "kb_coffee_rust",
        "category": "disease",
        "title": "Coffee Leaf Rust (Hemileia vastatrix)",
        "summary": "Major fungal disease threatening coffee yields in Rwanda.",
        "crops": ["Coffee"],
        "symptoms": "Yellow-orange powdery spots on leaf undersides; premature leaf drop.",
        "prevention": "Plant resistant varieties (Ruiru 11); proper spacing; shade management.",
        "treatment": "Copper-based fungicides; triadimefon; remove fallen leaves.",
        "image": "https://images.unsplash.com/photo-1738826588286-edda056cfb8f",
    },
    {
        "id": "kb_banana_xanthomonas",
        "category": "disease",
        "title": "Banana Xanthomonas Wilt (BXW)",
        "summary": "Bacterial disease devastating banana farms across East Africa.",
        "crops": ["Banana"],
        "symptoms": "Yellowing and wilting of leaves; premature ripening; yellowish bacterial ooze.",
        "prevention": "Disinfect tools; remove male buds; use clean planting materials.",
        "treatment": "Single Diseased Stem Removal (SDSR); cut & bury infected plants.",
        "image": "https://images.unsplash.com/photo-1758414083942-bc9f82e026b0",
    },
    {
        "id": "kb_ipm_basics",
        "category": "tip",
        "title": "Integrated Pest Management Basics",
        "summary": "A holistic approach combining biological, cultural and chemical controls.",
        "crops": ["All"],
        "symptoms": "—",
        "prevention": "Crop rotation, resistant varieties, beneficial insects, regular scouting.",
        "treatment": "Use chemicals only when pest threshold is exceeded; choose selective products.",
        "image": "https://images.unsplash.com/photo-1738826588286-edda056cfb8f",
    },
    {
        "id": "kb_org_neem",
        "category": "tip",
        "title": "Neem Oil — Organic Pesticide",
        "summary": "Natural pesticide effective against many crop pests.",
        "crops": ["All"],
        "symptoms": "—",
        "prevention": "Mix 5ml neem oil + 2ml soap per 1L water. Spray weekly in early morning/evening.",
        "treatment": "Apply directly on affected leaves; safe for beneficial insects.",
        "image": "https://images.unsplash.com/photo-1738826588286-edda056cfb8f",
    },
]


@api_router.get("/knowledge")
async def list_knowledge(category: Optional[str] = None):
    items = KNOWLEDGE_SEED
    if category:
        items = [k for k in items if k["category"] == category]
    return items


@api_router.get("/knowledge/{item_id}")
async def get_knowledge(item_id: str):
    for k in KNOWLEDGE_SEED:
        if k["id"] == item_id:
            return k
    raise HTTPException(404, "Not found")


# ---------------- AI Chatbot ----------------
@api_router.post("/chatbot")
async def chatbot(payload: ChatMsgIn, current: User = Depends(get_current_user)):
    if not llm_configured():
        raise HTTPException(
            500,
            "LLM not configured. Add GEMINI_API_KEY to backend/.env",
        )
    session_id = payload.session_id or f"chat_{current.user_id}_{uuid.uuid4().hex[:8]}"

    history_docs = await db.list_chatbot_history(session_id, 40)

    sys = (
        "You are CropDoctor AI, a friendly farming assistant for African farmers (Rwanda focus). "
        "Give short, practical, simple answers (3-6 sentences). "
        "Cover crop diseases, treatments, weather impact, organic remedies, and farming best practices. "
        "If user asks in Kinyarwanda or French, reply in their language."
    )

    chat = LlmChat(
        api_key=os.environ.get("GEMINI_API_KEY", EMERGENT_LLM_KEY),
        session_id=session_id,
        system_message=sys,
    ).with_model("gemini", "gemini-flash-latest")

    try:
        resp = await chat.send_message(UserMessage(text=payload.message))
    except Exception as e:
        logger.exception("Chatbot LLM error")
        raise HTTPException(502, f"AI chat error: {str(e)[:200]}")

    reply = resp if isinstance(resp, str) else str(resp)

    now = datetime.now(timezone.utc)
    await db.insert_chatbot_messages(
        [
            {
                "session_id": session_id,
                "user_id": current.user_id,
                "role": "user",
                "content": payload.message,
                "timestamp": now,
            },
            {
                "session_id": session_id,
                "user_id": current.user_id,
                "role": "assistant",
                "content": reply,
                "timestamp": datetime.now(timezone.utc),
            },
        ]
    )

    return {"session_id": session_id, "reply": reply, "timestamp": now.isoformat()}


@api_router.get("/chatbot/history")
async def chatbot_history(current: User = Depends(get_current_user), session_id: Optional[str] = None):
    return await db.list_chatbot_by_user(current.user_id, session_id, 200)


# ---------------- Expert Chat (mock + real DB) ----------------
@api_router.get("/experts")
async def list_experts():
    # Seed experts (mock list)
    return [
        {
            "user_id": "expert_1",
            "name": "Dr. Jean Mukamana",
            "role": "expert",
            "specialty": "Maize & Cereals",
            "rating": 4.9,
            "available": True,
            "picture": None,
        },
        {
            "user_id": "expert_2",
            "name": "Eric Habimana",
            "role": "expert",
            "specialty": "Coffee & Tea",
            "rating": 4.8,
            "available": True,
            "picture": None,
        },
        {
            "user_id": "expert_3",
            "name": "Alice Uwase",
            "role": "expert",
            "specialty": "Bananas & Fruits",
            "rating": 4.7,
            "available": False,
            "picture": None,
        },
    ]


class ExpertMsgIn(BaseModel):
    expert_id: str
    message: str


@api_router.post("/expert/messages")
async def send_expert_message(payload: ExpertMsgIn, current: User = Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    thread_id = f"thread_{current.user_id}_{payload.expert_id}"
    msg = {
        "thread_id": thread_id,
        "from_user": current.user_id,
        "to_user": payload.expert_id,
        "content": payload.message,
        "timestamp": now,
    }
    await db.insert_expert_message(msg)

    auto = {
        "thread_id": thread_id,
        "from_user": payload.expert_id,
        "to_user": current.user_id,
        "content": "Thanks for reaching out. I'll review your crop concern and get back to you shortly. Could you share a recent photo if you haven't already?",
        "timestamp": datetime.now(timezone.utc) + timedelta(seconds=1),
    }
    await db.insert_expert_message(auto)
    return {"ok": True, "thread_id": thread_id}


@api_router.get("/expert/messages/{expert_id}")
async def get_expert_messages(expert_id: str, current: User = Depends(get_current_user)):
    thread_id = f"thread_{current.user_id}_{expert_id}"
    docs = await db.list_expert_messages(thread_id, 200)
    for d in docs:
        d["timestamp"] = d["timestamp"].isoformat() if isinstance(d.get("timestamp"), datetime) else d["timestamp"]
    return docs


# ---------------- Notifications (mock + DB) ----------------
@api_router.get("/notifications")
async def notifications(current: User = Depends(get_current_user)):
    now = datetime.now(timezone.utc)
    seed = [
        {
            "id": "n1",
            "type": "alert",
            "icon": "AlertTriangle",
            "title": "High disease risk in your region",
            "body": "Late blight reported 5km from your farm. Inspect potatoes today.",
            "created_at": (now - timedelta(hours=2)).isoformat(),
            "read": False,
        },
        {
            "id": "n2",
            "type": "weather",
            "icon": "CloudRain",
            "title": "Heavy rainfall in 24h",
            "body": "Postpone pesticide application until after the rain.",
            "created_at": (now - timedelta(hours=5)).isoformat(),
            "read": False,
        },
        {
            "id": "n3",
            "type": "tip",
            "icon": "Leaf",
            "title": "Spray schedule reminder",
            "body": "Time to do your weekly preventive neem spray.",
            "created_at": (now - timedelta(days=1)).isoformat(),
            "read": True,
        },
        {
            "id": "n4",
            "type": "expert",
            "icon": "MessageSquare",
            "title": "Expert reply available",
            "body": "Dr. Jean Mukamana has responded to your question.",
            "created_at": (now - timedelta(days=2)).isoformat(),
            "read": True,
        },
    ]
    return seed


# ---------------- Admin ----------------
async def require_admin(current: User = Depends(get_current_user)) -> User:
    if current.role != "admin":
        raise HTTPException(403, "Admin only")
    return current


@api_router.get("/admin/stats")
async def admin_stats(current: User = Depends(require_admin)):
    total_users = await db.count_users()
    farmers = await db.count_users("farmer")
    experts = await db.count_users("expert")
    total_scans = await db.count_scans()
    diseased = await db.count_scans(healthy_only=False)
    top = await db.top_diseases(5)

    return {
        "total_users": total_users,
        "farmers": farmers,
        "experts": experts,
        "total_scans": total_scans,
        "diseased_scans": diseased,
        "ai_accuracy": 92.5,
        "top_diseases": [{"name": t["name"], "count": int(t["count"])} for t in top],
    }


# ---------------- Health ----------------
@api_router.get("/")
async def root():
    return {"app": "CropDoctor AI", "status": "ok"}


# Include router
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=[str(ROOT_DIR)],
    )
