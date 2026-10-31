from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import datetime

app = FastAPI(title="Pilot Fatigue Evaluator API", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# ─── Root ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# ─── Phase 2 Placeholder Models ──────────────────────────────────────────────
class PilotLoginRequest(BaseModel):
    employee_id: str
    password: str

class TelemetryEvent(BaseModel):
    timestamp: float
    event_type: str          # "blink", "long_blink", "yawn", "head_drop"
    ear: Optional[float]
    mar: Optional[float]
    score: float

class SessionPayload(BaseModel):
    pilot_id: str
    session_start: str
    session_end: str
    final_score: float
    events: List[TelemetryEvent]

# ─── Phase 2 Placeholder Endpoints ───────────────────────────────────────────
@app.post("/api/v1/auth/login")
async def pilot_login(req: PilotLoginRequest):
    """Phase 2: JWT auth will live here."""
    return {
        "status": "success",
        "message": f"Pilot {req.employee_id} authenticated (stub)",
        "token": "demo-jwt-token-placeholder"
    }

@app.post("/api/v1/telemetry/sync")
async def sync_session(payload: SessionPayload):
    """Phase 2: Persist session data to PostgreSQL here."""
    return {
        "status": "success",
        "pilot_id": payload.pilot_id,
        "final_score": payload.final_score,
        "events_recorded": len(payload.events),
        "server_time": datetime.datetime.utcnow().isoformat()
    }

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

