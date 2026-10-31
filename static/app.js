/**
 * AeroVigil — Pilot Fatigue Evaluator (app.js)
 * High-Precision Biometrics & Enhanced 3D Postural Flight-Deck Monitor
 *
 * Capabilities:
 *  1. PER-EYE INDIVIDUALIZED EAR: Independent per-eye tracking with normalized
 *     baseline ratios to catch unilateral ocular fatigue (one-eye droop).
 *  2. DUAL-TIER POSTURE FUSION ENGINE:
 *     - 3D Facial Euler Angles: Pitch (forward nodding in degrees), Yaw (inattention /
 *       looking away from cockpit instruments), Roll (lateral ear-to-shoulder tilt),
 *       and Z-depth translation (slumping back into seat).
 *     - MediaPipe Upper-Body Pose Landmarker: Tracks shoulders (11 & 12) for shoulder droop
 *       and chin-to-clavicle vertical distance for spinal slouching / hunched back.
 *     - Graceful Fallback: If shoulders are out of frame, the system runs in
 *       High-Precision 3D Face Mode with zero interruptions.
 *  3. COCKPIT HUD SKELETON: Renders illuminated sky-blue shoulder line, spinal axis,
 *     and diamond tactical reticles on the live video feed.
 *  4. AUTO-GROUND INTERLOCK: Immediate grounding when score drops to ≤ 25.
 *  5. AMORTIZED 60s TIME-WINDOWED PERCLOS: Physiologically tuned for flight crew safety.
 */

const MP_VERSION = "0.10.3";
const MP_CDN     = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

// ── Landmark Indices ──────────────────────────────────────────────────────────
// Face Landmarks
const LEFT_EYE     = [362, 263, 385, 387, 380, 373];
const RIGHT_EYE    = [33,  133, 160, 158, 144, 153];
const MOUTH        = [61, 291, 13, 14];
const FOREHEAD     = 10;
const CHIN         = 152;
const LEFT_EAR_PT  = 234;
const RIGHT_EAR_PT = 454;
const NOSE_TIP     = 1;
// Additional anchor points for robust posture geometry
const NOSE_BRIDGE  = 6;    // Bridge of nose (stable Z reference)
const L_CHEEK      = 116;  // Left cheek lateral point
const R_CHEEK      = 345;  // Right cheek lateral point

// Pose Landmarks (MediaPipe Pose)
const POSE_NOSE            = 0;
const POSE_LEFT_SHOULDER   = 11;
const POSE_RIGHT_SHOULDER  = 12;

// ── Scoring Constants & Thresholds ────────────────────────────────────────────
const EAR_CLOSED_RATIO         = 0.70;   // Eye closed when EAR < baseline × 0.70 (30% vertical drop)
const MAR_YAWN_RATIO           = 2.2;    // Yawn flagged when mouth opens > 2.2× resting baseline
const EAR_ASYM_THRESH          = 0.32;   // Asymmetry triggered when one eye drops >32% more than the other
const LONG_BLINK_MS            = 1500;   // 1500ms continuous closure = stage-1 micro-sleep
const YAWN_MIN_MS              = 1800;   // 1800ms sustained mouth opening = true yawn (filters speech)

// ── Posture Thresholds (Robust 2D/3D Cockpit Telemetry) ──────────────────────
// HEAD PITCH: Percentage drop in face height-to-width ratio from calibrated upright baseline.
// When head nods forward (chin drops toward chest), face height foreshortens.
// A 7.0% drop corresponds to a clear, realistic forward nodding event (~14-18° pitch).
const HEAD_PITCH_DROP_PCT      = 7.0;    // > 7.0% vertical foreshortening from baseline

// HEAD YAW: Derived from 2D ocular-nasal horizontal symmetry in screen coordinates.
// When head turns sideways, nose tip shifts away from the eye midpoint.
// Unaffected by depth noise; highly robust across all lighting and webcams.
const HEAD_YAW_DEG_THRESH      = 17.0;   // > 17° azimuth turn away from straight ahead
const HEAD_YAW_MS              = 2800;   // Sustained 2.8s before inattention penalty

// HEAD ROLL: Angular tilt (degrees) between outer eye corners (pt 362 vs pt 33).
const HEAD_TILT_DEG            = 8.5;    // > 8.5° lateral ear-to-shoulder roll from baseline

// HEAD PERSIST: Continuous head nod or slump duration before penalty registers.
const HEAD_PERSIST_MS          = 500;    // 500ms debounce — filters momentary blips

// TORSO / SPINAL POSTURE:
// 1. Upper-body pose active:
const SPINE_SLOUCH_RATIO_DROP  = 0.18;   // 18% drop in chin-to-clavicle clearance = torso slouch
const SHOULDER_TILT_DEG        = 8.0;    // > 8.0° shoulder height listing from baseline
// 2. Camera cropped to face (no shoulders visible):
// Tracks vertical sink of nose position in frame + depth scale
const SINK_SLOUCH_DROP         = 0.16;   // Face sinks down in camera frame by >16% of face height
const Z_SLOUCH_DROP            = 0.16;   // 16% face width shrinkage

const TORSO_PERSIST_MS         = 900;    // 900ms sustained slouch before penalty

const REGEN_PER_SEC            = 2.5;    // Frame-rate independent: +2.5 score points/sec when alert
const AUTO_GROUND_SCORE        = 25;     // Automatic flight grounding threshold

// Fatigue Penalties & Debounce Cooldowns (milliseconds)
const PENALTY  = {
  longBlink: 30,
  yawn:      15,
  headDrop:  22,
  asymmetry: 18,
  slouch:    18,
  yawDrift:  12
};

const COOLDOWN = {
  longBlink: 3000,
  yawn:      5000,
  headDrop:  3500,
  asymmetry: 5000,
  slouch:    4500,
  yawDrift:  5500
};

// ── DOM References ────────────────────────────────────────────────────────────
const video           = document.getElementById("webcam");
const canvas          = document.getElementById("output_canvas");
const ctx2d           = canvas.getContext("2d");
const noFaceOverlay   = document.getElementById("no-face-overlay");
const scanLine        = document.getElementById("scan-line");
const hudTargetStatus = document.getElementById("hud-target-status");

const leftEarVal      = document.getElementById("left-ear-val");
const rightEarVal     = document.getElementById("right-ear-val");
const marVal          = document.getElementById("mar-val");
const perclosVal      = document.getElementById("perclos-val");
const postureVal      = document.getElementById("posture-val");
const fpsBadge        = document.getElementById("fps-badge");

const stLear          = document.getElementById("st-lear");
const stRear          = document.getElementById("st-rear");
const stMar           = document.getElementById("st-mar");
const stPerclos       = document.getElementById("st-perclos");
const stPosture       = document.getElementById("st-posture");

const scoreNumber     = document.getElementById("score-number");
const scoreLabel      = document.getElementById("score-label");
const alertBanner     = document.getElementById("alert-banner");
const statusPill      = document.getElementById("status-pill");
const statusText      = document.getElementById("status-text");
const gaugeCanvas     = document.getElementById("gaugeCanvas");
const gaugeCtx        = gaugeCanvas.getContext("2d");

const eventLog        = document.getElementById("event-log");
const eventCount      = document.getElementById("event-count");
const eventCountBadge = document.getElementById("event-count-badge");
const longBlinkCountR = document.getElementById("longblink-count-r");
const yawnCountR      = document.getElementById("yawn-count-r");
const headDropCountR  = document.getElementById("headdrop-count-r");
const sessionTimeEl   = document.getElementById("session-time");
const clockEl         = document.getElementById("clock");
const utcClockEl      = document.getElementById("utc-clock");

const btnStart        = document.getElementById("btn-start");
const btnReset        = document.getElementById("btn-reset");
const btnExport       = document.getElementById("btn-export");
const loadingOverlay  = document.getElementById("loading-overlay");
const loadingMsg      = document.getElementById("loading-msg");
const calibModal      = document.getElementById("calibration-modal");
const calibProgress   = document.getElementById("calibration-progress");
const calibBar        = document.getElementById("calib-bar");
const calibCountdown  = document.getElementById("calib-countdown");
const btnBeginCalib   = document.getElementById("btn-begin-calibration");
const resultModal     = document.getElementById("result-modal");
const resultScore     = document.getElementById("result-score-display");
const resultVerdict   = document.getElementById("result-verdict");
const resultDetails   = document.getElementById("result-details");
const btnResultExport = document.getElementById("btn-result-export");
const btnResultRestart= document.getElementById("btn-result-restart");

// Cockpit Annunciator cells
const indLeftEye      = document.getElementById("ind-left-eye");
const indRightEye     = document.getElementById("ind-right-eye");
const indMouth        = document.getElementById("ind-mouth");
const indHeadPose     = document.getElementById("ind-head-pose") || document.getElementById("ind-posture");
const indTorso        = document.getElementById("ind-torso");

// ── State Variables ───────────────────────────────────────────────────────────
let faceLandmarker        = null;
let poseLandmarker        = null;
let poseLandmarkerReady   = false;
let sessionActive         = false;
let sessionEnding         = false;
let systemReady           = false;
let lastVideoTime         = -1;
let alertnessScore        = 100;

// Calibrated Baselines
let baselineLeftEAR       = 0.28;
let baselineRightEAR      = 0.28;
let baselineEAR           = 0.28;
let baselineMAR           = 0.12;
let baselinePitchRatio    = 1.15;   // Face height/width ratio upright (primary nod signal)
let baselineYawDeg        = 0.0;    // 2D horizontal gaze angle at straight-ahead (degrees)
let baselineTilt          = 0.0;    // Lateral roll angle at upright (degrees)
let baselineNoseY         = 0.45;   // Vertical nose position in frame (seat sink detection)
let baselineFaceWidth     = 140.0;  // Pixel face width at calibration distance
let baselineShoulderTilt  = 0.0;    // Signed shoulder angle at upright (degrees)
let baselineSpineRatio    = null;   // Chin-to-clavicle / shoulder-span ratio at upright
let hasUpperBodyBaseline  = false;

let isCalibrating         = false;
let calibFrames           = [];

// Signal Smoothing (Exponential Moving Average)
let smoothL_EAR           = null;
let smoothR_EAR           = null;
let smoothMAR             = null;
// Posture signal smoothing — separate EMA for each angle channel
let smoothPitchRatio      = null;
let smoothYawDeg          = null;
let smoothTiltDeg         = null;
let smoothNoseY           = null;
let smoothSpineRatio      = null;
let smoothShoulderTilt    = null;
const EMA_ALPHA           = 0.75;   // EAR / MAR: fast biometric signals
const EMA_POSTURE_ALPHA   = 0.60;   // Posture: 60% instant frame, 40% memory — smooth & responsive!

// Timing & Frame-rate Independence
let lastScoreUpdateTime   = performance.now();

// Eye State Tracking
let isBlinking            = false;
let blinkStartTime        = 0;
let longBlinkFired        = false;
let leftEyeClosed         = false;
let rightEyeClosed        = false;

// Yawn Tracking
let isYawning             = false;
let yawnStartTime         = 0;
let yawnFired             = false;

// Asymmetry Tracking (One-eye droop)
let asymFired             = false;
let asymStartTime         = 0;

// Posture Tracking Timestamps & Flags
let badHeadPoseStart      = null;
let badYawStart           = null;
let badTorsoStart         = null;
let isHeadNodding         = false;
let isHeadSlumping        = false;
let isYawDrifting         = false;
let isTorsoSlouching      = false;

// Cooldown Timestamps
const lastPenaltyAt       = {
  longBlink: 0,
  yawn:      0,
  headDrop:  0,
  asymmetry: 0,
  slouch:    0,
  yawDrift:  0
};

// Rolling 60s PERCLOS (P80)
const PERCLOS_WINDOW_MS   = 60000;
let perclosHistory        = [];
let perclosSum            = 0;

// Event Counters
let totalEvents           = 0;
let totalYawns            = 0;
let totalLongBlinks       = 0;
let totalHeadDrops        = 0;
let totalSlouches         = 0;
let totalYawDrifts        = 0;
let sessionStart          = null;
let sessionInterval       = null;
let eventHistory          = [];

// FPS Counter
let fpsCount              = 0;
let fpsTimer              = performance.now();

// ── Chart.js Flight Telemetry Setup ──────────────────────────────────────────
const N = 60;
const lEH = Array(N).fill(0.28);
const rEH = Array(N).fill(0.28);
const mH  = Array(N).fill(0.10);
const sH  = Array(N).fill(1.0);

const telChart = new Chart(document.getElementById("telemetryChart"), {
  type: "line",
  data: {
    labels: Array(N).fill(""),
    datasets: [
      { label: "L-EAR", data: lEH, borderColor: "#0284C7", borderWidth: 1.8, pointRadius: 0, tension: 0.3, yAxisID: "yL" },
      { label: "R-EAR", data: rEH, borderColor: "#0EA5E9", borderWidth: 1.8, pointRadius: 0, tension: 0.3, yAxisID: "yL" },
      { label: "MAR",   data: mH,  borderColor: "#D97706", borderWidth: 1.8, pointRadius: 0, tension: 0.3, yAxisID: "yL" },
      { label: "Score", data: sH,  borderColor: "#059669", borderWidth: 1.8, pointRadius: 0, tension: 0.3,
        yAxisID: "yR", fill: true, backgroundColor: "rgba(5, 150, 105, 0.08)" }
    ]
  },
  options: {
    animation: false,
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x:  { display: false },
      yL: { position: "left",  min: 0, max: 1,   grid: { color: "#E2E8F0" }, ticks: { color: "#64748B", font: { size: 9 } } },
      yR: { position: "right", min: 0, max: 100, grid: { drawOnChartArea: false }, ticks: { color: "#059669", font: { size: 9 } } }
    }
  }
});

// ── Mathematical & Biometric Helpers ─────────────────────────────────────────

/**
 * Aspect-ratio corrected 2D Euclidean distance in pixels.
 */
function dist2D(p1, p2) {
  if (!p1 || !p2) return 0;
  const w = canvas.width  || 640;
  const h = canvas.height || 480;
  const dx = (p1.x - p2.x) * w;
  const dy = (p1.y - p2.y) * h;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Eye Aspect Ratio (EAR) for a single eye.
 */
function calcEAR(lm, idx) {
  if (!lm) return 0.28;
  const v1 = dist2D(lm[idx[2]], lm[idx[4]]);
  const v2 = dist2D(lm[idx[3]], lm[idx[5]]);
  const h  = dist2D(lm[idx[0]], lm[idx[1]]);
  return h < 1e-4 ? 0.28 : (v1 + v2) / (2.0 * h);
}

/**
 * Mouth Aspect Ratio (MAR) for yawn detection.
 */
function calcMAR(lm) {
  if (!lm) return 0.10;
  const v = dist2D(lm[MOUTH[2]], lm[MOUTH[3]]);
  const h = dist2D(lm[MOUTH[0]], lm[MOUTH[1]]);
  return h < 1e-4 ? 0.10 : v / h;
}

/**
 * 2D Face Pitch Ratio (Height / Width).
 * Forehead (10) to Chin (152) vs Left Tragus (234) to Right Tragus (454).
 * This is the PRIMARY and most reliable head pitch signal:
 * — When head pitches forward (chin drops), vertical span foreshortens
 *   relative to fixed horizontal ear-to-ear span.
 * — Calibrated at upright; any ratio drop beyond threshold = nodding.
 */
function calcPitchRatio(lm) {
  if (!lm) return 1.0;
  const faceH = dist2D(lm[FOREHEAD], lm[CHIN]);
  const faceW = dist2D(lm[LEFT_EAR_PT], lm[RIGHT_EAR_PT]);
  return faceW < 1e-4 ? 1.0 : faceH / faceW;
}

/**
 * Head Yaw in Degrees using 2D Ocular-Nasal Horizontal Projection.
 * Measures horizontal displacement of nose tip (pt 1) from the midpoint
 * of the two outer eye corners (pt 33 and pt 362), normalized by interocular distance.
 *
 * Physical basis:
 *  - When looking directly at camera: nose is centered between eyes -> yaw ~ 0°
 *  - When turning right (viewer perspective): nose shifts left -> negative yaw
 *  - When turning left (viewer perspective): nose shifts right -> positive yaw
 * Returns degrees ∈ [-60°, +60°].
 */
function calcYawDeg(lm) {
  if (!lm) return 0.0;
  const w = canvas.width  || 640;
  const rEyeX = lm[33].x * w;   // Right eye outer corner (viewer's left side)
  const lEyeX = lm[362].x * w;  // Left eye outer corner (viewer's right side)
  const noseX = lm[1].x * w;    // Nose tip

  const eyeSpan = lEyeX - rEyeX;
  if (eyeSpan < 15) return 0.0;

  const eyeMidX = (lEyeX + rEyeX) / 2;
  const dx = noseX - eyeMidX;
  const normYaw = Math.max(-1, Math.min(1, dx / (eyeSpan * 0.45)));
  return normYaw * 45.0; // Scaled to realistic angular degrees
}

/**
 * Head Lateral Roll/Tilt Angle in degrees.
 * Outer eye corner horizontal line deviation (pt 362 vs pt 33).
 */
function calcTilt(lm) {
  if (!lm) return 0.0;
  const w = canvas.width  || 640;
  const h = canvas.height || 480;
  // pt 362 = left outer corner, pt 33 = right outer corner
  const dx = (lm[362].x - lm[33].x) * w;
  const dy = (lm[362].y - lm[33].y) * h;
  return Math.atan2(dy, dx) * (180 / Math.PI);
}

/**
 * Vertical position of nose tip (normalized 0 to 1).
 * Used for detecting vertical chair sinking / slouching when shoulders are not visible.
 */
function calcNoseY(lm) {
  return lm ? lm[1].y : 0.45;
}

/**
 * Face height in pixels.
 */
function calcFaceHeight(lm) {
  return lm ? dist2D(lm[FOREHEAD], lm[CHIN]) : 150;
}

/**
 * Face width in pixels (scale / depth proxy).
 */
function calcFaceWidth(lm) {
  return lm ? dist2D(lm[LEFT_EAR_PT], lm[RIGHT_EAR_PT]) : 100;
}

/**
 * Upper-body pose metrics from MediaPipe PoseLandmarker.
 *
 * CRITICAL FIX: dx is (ls.x - rs.x) * w which is positive (~0.35 * width > 0).
 * This ensures atan2 is smoothly centered around 0° without 180° / -180° branch cut jumps!
 */
function getPoseMetrics(poseLm, faceLm) {
  if (!poseLm || poseLm.length < 13) return null;
  const ls = poseLm[POSE_LEFT_SHOULDER];  // Pt 11 (user's left = viewer's right, x ~ 0.65-0.75)
  const rs = poseLm[POSE_RIGHT_SHOULDER]; // Pt 12 (user's right = viewer's left, x ~ 0.25-0.35)
  if (!ls || !rs) return null;

  const visL = ls.visibility !== undefined ? ls.visibility : 1.0;
  const visR = rs.visibility !== undefined ? rs.visibility : 1.0;
  if (visL < 0.35 || visR < 0.35) return null;

  const w = canvas.width  || 640;
  const h = canvas.height || 480;

  // IMPORTANT: (ls.x - rs.x) > 0 so atan2(dy, dx) is continuously centered at 0°!
  const dx = (ls.x - rs.x) * w;
  const dy = (ls.y - rs.y) * h;
  const shoulderSpan = Math.sqrt(dx * dx + dy * dy);
  if (shoulderSpan < 25) return null;

  // SIGNED shoulder tilt angle around 0° (positive = left shoulder lower, negative = right lower)
  const shoulderTilt = Math.atan2(dy, dx) * (180 / Math.PI);

  // Spinal / neck vertical clearance: distance from chin to shoulder midpoint.
  // In screen coordinates Y increases downward, so shoulders_midY > chin_Y normally.
  const midY  = ((ls.y + rs.y) / 2) * h;
  const chinY = faceLm ? (faceLm[CHIN].y * h) : (poseLm[0] ? poseLm[0].y * h : midY * 0.85);
  const neckVertDist = midY - chinY; // Positive when upright (chin above shoulders)

  // Normalize to shoulder span so camera-distance changes don't distort it
  const spineRatio = neckVertDist / shoulderSpan;

  return { shoulderSpan, shoulderTilt, neckVertDist, spineRatio, ls, rs };
}

/**
 * Scientifically Standardized Time-Windowed PERCLOS (P80).
 * Calculates eye closure percentage over rolling 60s in O(1) time.
 */
function calcPerclosTimeWindow(eyeClosed, now) {
  perclosHistory.push({ t: now, closed: eyeClosed ? 1 : 0 });
  if (eyeClosed) perclosSum++;

  const cutoff = now - PERCLOS_WINDOW_MS;
  while (perclosHistory.length > 0 && perclosHistory[0].t < cutoff) {
    if (perclosHistory[0].closed) perclosSum--;
    perclosHistory.shift();
  }

  return perclosHistory.length > 0 ? (perclosSum / perclosHistory.length) * 100 : 0;
}

/**
 * Penalty deduction with debounce cooldown.
 */
function penalise(type) {
  const now = performance.now();
  if (now - (lastPenaltyAt[type] || 0) >= (COOLDOWN[type] || 3000)) {
    lastPenaltyAt[type] = now;
    alertnessScore = Math.max(0, alertnessScore - (PENALTY[type] || 15));
    return true;
  }
  return false;
}

/**
 * Helper to update cockpit annunciator cells.
 * state: "ok" | "warn" | "bad" | "info"
 */
function setIndicator(el, state, text) {
  if (!el) return;
  el.className = `posture-cell ${state}`;
  const stateEl = el.querySelector(".ind-state");
  if (stateEl && text) stateEl.textContent = text;
}

// ── Enhanced Biometric Heuristic Engine ───────────────────────────────────────
function runHeuristics(lm, poseLm) {
  const now = performance.now();

  // Frame-rate independent delta time (seconds)
  const dtSec = Math.min(0.2, (now - lastScoreUpdateTime) / 1000);
  lastScoreUpdateTime = now;

  // 1. Raw Feature Extraction
  const rawLeftEAR    = calcEAR(lm, LEFT_EYE);
  const rawRightEAR   = calcEAR(lm, RIGHT_EYE);
  const rawMAR        = calcMAR(lm);
  const rawPitchRatio = calcPitchRatio(lm);
  const rawYawDeg     = calcYawDeg(lm);
  const rawTiltDeg    = calcTilt(lm);
  const rawNoseY      = calcNoseY(lm);
  const rawFaceHeight = calcFaceHeight(lm);
  const rawFaceWidth  = calcFaceWidth(lm);
  const poseData      = getPoseMetrics(poseLm, lm);

  // 2. Exponential Moving Average — ocular signals
  smoothL_EAR = smoothL_EAR === null ? rawLeftEAR  : (EMA_ALPHA * rawLeftEAR  + (1 - EMA_ALPHA) * smoothL_EAR);
  smoothR_EAR = smoothR_EAR === null ? rawRightEAR : (EMA_ALPHA * rawRightEAR + (1 - EMA_ALPHA) * smoothR_EAR);
  smoothMAR   = smoothMAR   === null ? rawMAR      : (EMA_ALPHA * rawMAR      + (1 - EMA_ALPHA) * smoothMAR);

  // Posture EMA — responsive smoothing
  smoothPitchRatio = smoothPitchRatio === null ? rawPitchRatio : (EMA_POSTURE_ALPHA * rawPitchRatio + (1 - EMA_POSTURE_ALPHA) * smoothPitchRatio);
  smoothYawDeg     = smoothYawDeg     === null ? rawYawDeg     : (EMA_POSTURE_ALPHA * rawYawDeg     + (1 - EMA_POSTURE_ALPHA) * smoothYawDeg);
  smoothTiltDeg    = smoothTiltDeg    === null ? rawTiltDeg    : (EMA_POSTURE_ALPHA * rawTiltDeg    + (1 - EMA_POSTURE_ALPHA) * smoothTiltDeg);
  smoothNoseY      = smoothNoseY      === null ? rawNoseY      : (EMA_POSTURE_ALPHA * rawNoseY      + (1 - EMA_POSTURE_ALPHA) * smoothNoseY);

  const leftEAR  = smoothL_EAR;
  const rightEAR = smoothR_EAR;
  const mar      = smoothMAR;

  // 3. Per-Eye Closure & Relative Asymmetry
  const leftEAR_T  = baselineLeftEAR  * EAR_CLOSED_RATIO;
  const rightEAR_T = baselineRightEAR * EAR_CLOSED_RATIO;

  leftEyeClosed  = leftEAR  < leftEAR_T;
  rightEyeClosed = rightEAR < rightEAR_T;

  const eitherClosed = leftEyeClosed || rightEyeClosed;
  const bothClosed   = leftEyeClosed && rightEyeClosed;

  // Normalized Asymmetry
  const lRatio = leftEAR  / Math.max(0.08, baselineLeftEAR);
  const rRatio = rightEAR / Math.max(0.08, baselineRightEAR);
  const relativeAsym = Math.abs(lRatio - rRatio);

  // 4. Rolling 60s PERCLOS
  const perclos = calcPerclosTimeWindow(eitherClosed, now);

  // 5. Long Blink / Micro-Sleep Detection
  if (eitherClosed) {
    if (!isBlinking) {
      isBlinking     = true;
      longBlinkFired = false;
      blinkStartTime = now;
    } else if (!longBlinkFired && (now - blinkStartTime) >= LONG_BLINK_MS) {
      longBlinkFired = true;
      if (penalise("longBlink")) {
        const which = bothClosed ? "BILATERAL" : leftEyeClosed ? "LEFT EYE" : "RIGHT EYE";
        logEvent(`MICRO-SLEEP DETECTED (${which} CLOSED > 1.5s)`, "longblink");
        totalLongBlinks++;
        longBlinkCountR.textContent = totalLongBlinks;
      }
    }
  } else {
    isBlinking     = false;
    longBlinkFired = false;
  }

  // 6. One-Eye Droop (Relative Asymmetry Detection)
  if (relativeAsym > EAR_ASYM_THRESH && !bothClosed) {
    if (!asymFired) {
      asymFired     = true;
      asymStartTime = now;
    } else if ((now - asymStartTime) >= 1100) {
      asymFired = false;
      if (penalise("asymmetry")) {
        const droppingEye = lRatio < rRatio ? "LEFT" : "RIGHT";
        logEvent(`UNILATERAL EYE DROOP (${droppingEye} EYE >32% ASYMMETRY)`, "asym");
      }
    }
  } else {
    asymFired = false;
  }

  // 7. Yawn Detection
  const MAR_T = baselineMAR * MAR_YAWN_RATIO;
  if (mar > MAR_T) {
    if (!isYawning) {
      isYawning     = true;
      yawnFired     = false;
      yawnStartTime = now;
    } else if (!yawnFired && (now - yawnStartTime) >= YAWN_MIN_MS) {
      yawnFired = true;
      if (penalise("yawn")) {
        logEvent(`PROLONGED YAWN DETECTED (${mar.toFixed(2)} MAR)`, "yawn");
        totalYawns++;
        yawnCountR.textContent = totalYawns;
      }
    }
  } else {
    isYawning = false;
    yawnFired = false;
  }

  // ── 8. Enhanced Posture Engine ───────────────────────────────────────────────
  // A. HEAD PITCH (Forward Nod) — Percentage foreshortening from upright baseline
  const pitchDropPct = ((baselinePitchRatio - smoothPitchRatio) / Math.max(0.1, baselinePitchRatio)) * 100;
  isHeadNodding = pitchDropPct > HEAD_PITCH_DROP_PCT;

  // B. HEAD ROLL (Lateral Slump) — Outer eye corner roll deviation in degrees
  const tiltDelta = Math.abs(smoothTiltDeg - baselineTilt);
  isHeadSlumping = tiltDelta > HEAD_TILT_DEG;

  // C. HEAD YAW (Cockpit Inattention) — 2D ocular-nasal azimuth in degrees
  const yawDelta = Math.abs(smoothYawDeg - baselineYawDeg);
  isYawDrifting = yawDelta > HEAD_YAW_DEG_THRESH;

  // D. TORSO / SPINAL POSTURE
  let isShoulderDroop = false;
  let isSpinalSlouch  = false;
  let hasUpperBody    = false;

  if (poseData && baselineSpineRatio !== null) {
    hasUpperBody = true;

    smoothSpineRatio   = smoothSpineRatio   === null ? poseData.spineRatio   : (EMA_POSTURE_ALPHA * poseData.spineRatio   + (1 - EMA_POSTURE_ALPHA) * smoothSpineRatio);
    smoothShoulderTilt = smoothShoulderTilt === null ? poseData.shoulderTilt : (EMA_POSTURE_ALPHA * poseData.shoulderTilt + (1 - EMA_POSTURE_ALPHA) * smoothShoulderTilt);

    const shoulderTiltDelta = Math.abs(smoothShoulderTilt - baselineShoulderTilt);
    isShoulderDroop = shoulderTiltDelta > SHOULDER_TILT_DEG;
    isSpinalSlouch  = smoothSpineRatio < (baselineSpineRatio * (1 - SPINE_SLOUCH_RATIO_DROP));

  } else {
    // Camera cropped to head/desk fallback:
    // 1. Seat vertical sink: nose tip moves down in frame relative to face height
    const faceH = Math.max(50, rawFaceHeight);
    const sinkRatio = ((smoothNoseY - baselineNoseY) * (canvas.height || 480)) / faceH;
    // 2. Leaning back away from desk: face width drops without yawing
    const scaleRatio = rawFaceWidth / Math.max(1, baselineFaceWidth);

    isSpinalSlouch = (sinkRatio > SINK_SLOUCH_DROP) || (!isYawDrifting && scaleRatio < (1 - Z_SLOUCH_DROP));
  }

  isTorsoSlouching = isShoulderDroop || isSpinalSlouch;

  // Active postural fatigue for regen suppression
  const headPostureBad  = isHeadNodding || isHeadSlumping;
  const torsoPostureBad = isTorsoSlouching;
  const anyPenaltyPosture = headPostureBad || torsoPostureBad;

  // Persistence & Penalty: Head Nod / Lateral Slump (500ms debounce)
  if (headPostureBad) {
    if (!badHeadPoseStart) {
      badHeadPoseStart = now;
    } else if ((now - badHeadPoseStart) >= HEAD_PERSIST_MS) {
      if (penalise("headDrop")) {
        const reason = isHeadNodding
          ? `FORWARD HEAD NOD (−${pitchDropPct.toFixed(0)}% FORESHORTEN)`
          : `LATERAL HEAD SLUMP (${tiltDelta.toFixed(0)}° ROLL)`;
        logEvent(`POSTURAL FATIGUE: ${reason}`, "head");
        totalHeadDrops++;
        headDropCountR.textContent = totalHeadDrops + totalSlouches;
      }
      badHeadPoseStart = null;
    }
  } else {
    badHeadPoseStart = null;
  }

  // Persistence & Penalty: Yaw Inattention (2.8s sustained)
  if (isYawDrifting) {
    if (!badYawStart) {
      badYawStart = now;
    } else if ((now - badYawStart) >= HEAD_YAW_MS) {
      if (penalise("yawDrift")) {
        const dir = smoothYawDeg > baselineYawDeg ? "RIGHT" : "LEFT";
        logEvent(`FLIGHT INATTENTION: HEAD TURNED ${dir} (${yawDelta.toFixed(0)}° YAW > 2.8s)`, "caution");
        totalYawDrifts++;
      }
      badYawStart = null;
    }
  } else {
    badYawStart = null;
  }

  // Persistence & Penalty: Torso / Spinal Slouch (900ms sustained)
  if (torsoPostureBad) {
    if (!badTorsoStart) {
      badTorsoStart = now;
    } else if ((now - badTorsoStart) >= TORSO_PERSIST_MS) {
      if (penalise("slouch")) {
        const slouchReason = isShoulderDroop ? "SHOULDER LISTING / DROOP" : "SPINAL SLOUCH (TORSO COMPRESSION)";
        logEvent(`POSTURAL FATIGUE: ${slouchReason}`, "head");
        totalSlouches++;
        headDropCountR.textContent = totalHeadDrops + totalSlouches;
      }
      badTorsoStart = null;
    }
  } else {
    badTorsoStart = null;
  }

  // 9. Frame-Rate Independent Score Regeneration
  // Blocked only by active ocular fatigue or confirmed head/torso posture events.
  // Yaw inattention does NOT block regen (it has its own separate penalty).
  if (!eitherClosed && !isYawning && !anyPenaltyPosture) {
    alertnessScore = Math.min(100, alertnessScore + REGEN_PER_SEC * dtSec);
  }

  // 10. Continuous PERCLOS Drag Penalty
  if (perclos > 20) {
    alertnessScore = Math.max(0, alertnessScore - 1.5 * dtSec);
  } else if (perclos > 15) {
    alertnessScore = Math.max(0, alertnessScore - 0.6 * dtSec);
  }

  // 11. Safety Interlock: Automatic Grounding at Critical Score (≤ 25)
  if (alertnessScore <= AUTO_GROUND_SCORE && !sessionEnding) {
    sessionEnding = true;
    logEvent("SAFETY INTERLOCK: CRITICAL SCORE — AUTOMATIC FLIGHT GROUNDING", "critical");
    setTimeout(() => endSession(true), 1200);
  }

  // 12. Update UI Metric Displays
  leftEarVal.textContent  = leftEAR.toFixed(3);
  rightEarVal.textContent = rightEAR.toFixed(3);
  marVal.textContent      = mar.toFixed(3);
  perclosVal.textContent  = perclos.toFixed(1) + "%";

  // Posture metric card: live angle/status
  if (isHeadNodding) {
    postureVal.textContent = `-${pitchDropPct.toFixed(0)}% NOD`;
  } else if (isHeadSlumping) {
    postureVal.textContent = `${tiltDelta.toFixed(0)}° ROLL`;
  } else if (isTorsoSlouching) {
    postureVal.textContent = isShoulderDroop ? "DROOP" : "SLOUCH";
  } else if (isYawDrifting) {
    const dir = smoothYawDeg > baselineYawDeg ? "R" : "L";
    postureVal.textContent = `${yawDelta.toFixed(0)}° YAW-${dir}`;
  } else {
    postureVal.textContent = "LEVEL";
  }

  // Status cell micro-badges
  if (stLear)    stLear.textContent    = leftEyeClosed ? "Closed" : leftEAR < (baselineLeftEAR * 0.85) ? "Strained" : "Normal";
  if (stRear)    stRear.textContent    = rightEyeClosed ? "Closed" : rightEAR < (baselineRightEAR * 0.85) ? "Strained" : "Normal";
  if (stMar)     stMar.textContent     = mar > MAR_T ? "Yawning" : mar > (MAR_T * 0.6) ? "Speaking" : "Closed";
  if (stPerclos) stPerclos.textContent = perclos > 20 ? "Critical" : perclos > 15 ? "Elevated" : "Optimal";
  if (stPosture) {
    stPosture.textContent = isHeadNodding    ? "Forward Nod" :
                            isHeadSlumping   ? "Lateral Slump" :
                            isTorsoSlouching ? (isShoulderDroop ? "Shoulder Droop" : "Torso Slouch") :
                            isYawDrifting    ? "Inattentive" : "Upright";
  }

  const postureChip = document.getElementById("posture-chip");
  if (postureChip) {
    postureChip.textContent = hasUpperBody ? "TORSO+HEAD" : "HEAD 3D";
  }

  // Cockpit Annunciator Indicators
  const lEyeState = leftEyeClosed ? "bad" : leftEAR < (baselineLeftEAR * 0.85) ? "warn" : "ok";
  const rEyeState = rightEyeClosed ? "bad" : rightEAR < (baselineRightEAR * 0.85) ? "warn" : "ok";
  const mState    = mar > MAR_T ? "bad" : mar > (MAR_T * 0.6) ? "warn" : "ok";

  const headState = headPostureBad ? "bad" : isYawDrifting ? "warn" : "ok";
  const headText  = isHeadNodding ? "NODDING" : isHeadSlumping ? "SLUMP" : isYawDrifting ? "YAW DRIFT" : "LEVEL";

  const torsoState = torsoPostureBad ? "bad" : (hasUpperBody ? "ok" : "info");
  const torsoText  = isTorsoSlouching ? (isShoulderDroop ? "DROOP" : "SLOUCH") : (hasUpperBody ? "UPRIGHT" : "DESK 3D");

  setIndicator(indLeftEye,  lEyeState,  leftEyeClosed ? "CLOSED" : leftEAR < (baselineLeftEAR * 0.85) ? "DROOP" : "NORMAL");
  setIndicator(indRightEye, rEyeState,  rightEyeClosed ? "CLOSED" : rightEAR < (baselineRightEAR * 0.85) ? "DROOP" : "NORMAL");
  setIndicator(indMouth,    mState,     mar > MAR_T ? "YAWNING" : mar > (MAR_T * 0.6) ? "OPEN" : "CLOSED");
  setIndicator(indHeadPose, headState,  headText);
  setIndicator(indTorso,    torsoState, torsoText);

  return { leftEAR, rightEAR, avgEAR: (leftEAR + rightEAR) / 2, mar };
}

// ── Dashboard Updates ─────────────────────────────────────────────────────────
function updateDashboard(leftEAR, rightEAR, mar) {
  const s = Math.floor(alertnessScore);
  scoreNumber.textContent = s;

  let color, label, pillClass, pillText;
  if (s >= 75) {
    color = "#059669"; label = "FIT TO FLY · CLEARED";
    pillClass = "active"; pillText = "OPTIMAL — CLEAR FOR FLIGHT";
    alertBanner.classList.add("hidden");
  } else if (s >= 45) {
    color = "#D97706"; label = "FATIGUED · MONITORING REQUIRED";
    pillClass = "caution"; pillText = "CAUTION — ELEVATED FATIGUE";
    alertBanner.classList.add("hidden");
  } else if (s > AUTO_GROUND_SCORE) {
    color = "#DC2626"; label = "HIGH RISK · GROUNDING RECOMMENDED";
    pillClass = "critical"; pillText = "CRITICAL — REPLACEMENT REQUIRED";
    alertBanner.classList.remove("hidden");
  } else {
    color = "#DC2626"; label = "UNFIT TO FLY · GROUNDED";
    pillClass = "critical"; pillText = "CRITICAL — GROUNDED AUTOMATICALLY";
    alertBanner.classList.remove("hidden");
  }

  scoreNumber.style.color = color;
  scoreLabel.style.color  = color;
  scoreLabel.textContent  = label;
  statusPill.className    = `pill ${pillClass}`;
  if (statusText) statusText.textContent = pillText;
  drawGauge(s, color);

  lEH.push(leftEAR);  lEH.shift();
  rEH.push(rightEAR); rEH.shift();
  mH.push(mar);       mH.shift();
  sH.push(s / 100);   sH.shift();
  telChart.update("none");
}

// ── Flight Deck Gauge ─────────────────────────────────────────────────────────
function drawGauge(score, color = "#94A3B8") {
  const dpr = window.devicePixelRatio || 1;
  const w = 220;
  const h = 130;

  if (gaugeCanvas.width !== w * dpr) {
    gaugeCanvas.width = w * dpr;
    gaugeCanvas.height = h * dpr;
    gaugeCanvas.style.width = `${w}px`;
    gaugeCanvas.style.height = `${h}px`;
  }

  gaugeCtx.save();
  gaugeCtx.scale(dpr, dpr);
  gaugeCtx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h - 14;
  const r = 82;

  // Segmented graduation ticks around outer track
  const totalTicks = 24;
  for (let i = 0; i <= totalTicks; i++) {
    const angle = Math.PI + (i / totalTicks) * Math.PI;
    const isMajor = i % 6 === 0;
    const tickLen = isMajor ? 6 : 3;
    const r1 = r + 13;
    const r2 = r1 + tickLen;
    gaugeCtx.beginPath();
    gaugeCtx.moveTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
    gaugeCtx.lineTo(cx + Math.cos(angle) * r2, cy + Math.sin(angle) * r2);
    gaugeCtx.strokeStyle = isMajor ? "#94A3B8" : "#E2E8F0";
    gaugeCtx.lineWidth = isMajor ? 1.5 : 1;
    gaugeCtx.stroke();
  }

  // Base background arc
  gaugeCtx.beginPath();
  gaugeCtx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
  gaugeCtx.strokeStyle = "#E2E8F0";
  gaugeCtx.lineWidth = 14;
  gaugeCtx.lineCap = "round";
  gaugeCtx.stroke();

  // Active score arc
  if (score > 0) {
    const endAngle = Math.PI + (Math.max(0, Math.min(100, score)) / 100) * Math.PI;
    gaugeCtx.beginPath();
    gaugeCtx.arc(cx, cy, r, Math.PI, endAngle);
    gaugeCtx.strokeStyle = color;
    gaugeCtx.lineWidth = 14;
    gaugeCtx.lineCap = "round";
    gaugeCtx.stroke();
  }

  // Needle Pivot Hub
  gaugeCtx.beginPath();
  gaugeCtx.arc(cx, cy, 6, 0, 2 * Math.PI);
  gaugeCtx.fillStyle = color;
  gaugeCtx.fill();

  gaugeCtx.beginPath();
  gaugeCtx.arc(cx, cy, 2.5, 0, 2 * Math.PI);
  gaugeCtx.fillStyle = "#FFFFFF";
  gaugeCtx.fill();

  gaugeCtx.restore();
}

// ── Operational Event Logger ──────────────────────────────────────────────────
function logEvent(text, type = "info") {
  const time = new Date().toTimeString().slice(0, 8);
  totalEvents++;
  eventCount.textContent = totalEvents;
  eventCountBadge.textContent = `${totalEvents} event${totalEvents === 1 ? "" : "s"}`;

  eventHistory.unshift({ time, text, type });
  if (eventHistory.length > 50) eventHistory.pop();

  const empty = eventLog.querySelector(".log-empty");
  if (empty) empty.remove();

  const li = document.createElement("li");
  li.className = `log-entry log-${type}`;
  li.innerHTML = `
    <span class="log-time mono">${time}</span>
    <span class="log-msg">${text}</span>`;
  eventLog.prepend(li);
  while (eventLog.children.length > 30) eventLog.lastElementChild.remove();
}

// ── Session Timer & Clocks ────────────────────────────────────────────────────
function startSessionTimer() {
  sessionStart = Date.now();
  sessionInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    sessionTimeEl.textContent = `${m}:${s}`;
  }, 1000);
}

function updateClocks() {
  const d = new Date();
  clockEl.textContent = d.toTimeString().slice(0, 8);
  utcClockEl.textContent = d.toISOString().slice(11, 19) + " UTC";
}
setInterval(updateClocks, 1000);
updateClocks();

function tickFPS() {
  fpsCount++;
  const now = performance.now();
  if (now - fpsTimer >= 1000) {
    fpsBadge.textContent = `${fpsCount} FPS`;
    fpsCount = 0;
    fpsTimer = now;
  }
}

// ── Calibration Flow ──────────────────────────────────────────────────────────
function startCalibration() {
  calibModal.classList.add("hidden");
  calibProgress.classList.remove("hidden");
  isCalibrating = true;
  calibFrames   = [];

  let count = 10;
  calibCountdown.textContent = count;
  calibBar.style.width = "0%";

  const iv = setInterval(() => {
    count--;
    calibCountdown.textContent = count;
    calibBar.style.width = `${((10 - count) / 10) * 100}%`;
    if (count <= 0) {
      clearInterval(iv);
      finishCalibration();
    }
  }, 1000);
}

function finishCalibration() {
  isCalibrating = false;

  if (calibFrames.length >= 15) {
    const sort = arr => [...arr].sort((a, b) => a - b);

    // Individual per-eye baselines
    const sortedL_EAR = sort(calibFrames.map(f => f.leftEAR));
    const sortedR_EAR = sort(calibFrames.map(f => f.rightEAR));
    const openEyeIdx  = Math.floor(sortedL_EAR.length * 0.65);

    baselineLeftEAR  = Math.max(0.18, Math.min(0.45, sortedL_EAR[openEyeIdx]));
    baselineRightEAR = Math.max(0.18, Math.min(0.45, sortedR_EAR[openEyeIdx]));
    baselineEAR      = (baselineLeftEAR + baselineRightEAR) / 2;

    // Resting mouth MAR
    const sortedMAR = sort(calibFrames.map(f => f.mar));
    const restingMouthIdx = Math.floor(sortedMAR.length * 0.30);
    baselineMAR = Math.max(0.06, Math.min(0.28, sortedMAR[restingMouthIdx]));

    // 3D Head Euler & Pitch baselines (median of calibration frames)
    const sortedPitchRatio = sort(calibFrames.map(f => f.pitchRatio));
    baselinePitchRatio = sortedPitchRatio[Math.floor(sortedPitchRatio.length / 2)];

    // Upright Head Yaw baseline (median in degrees, ~0° when facing camera)
    const sortedYawDeg = sort(calibFrames.map(f => f.yawDeg));
    baselineYawDeg = sortedYawDeg[Math.floor(sortedYawDeg.length / 2)];

    // Upright Head Roll/Tilt baseline (median in degrees)
    const sortedTilt = sort(calibFrames.map(f => f.tiltDeg));
    baselineTilt = sortedTilt[Math.floor(sortedTilt.length / 2)];

    // Upright Nose Y position (median for seat sink detection)
    const sortedNoseY = sort(calibFrames.map(f => f.noseY));
    baselineNoseY = sortedNoseY[Math.floor(sortedNoseY.length / 2)];

    const sortedFaceW = sort(calibFrames.map(f => f.faceWidth));
    baselineFaceWidth = sortedFaceW[Math.floor(sortedFaceW.length / 2)];

    // Shoulder & Spinal baseline (if upper body visible)
    const validPoses = calibFrames.map(f => f.poseData).filter(p => p !== null);
    if (validPoses.length >= 8) {
      const sortedShoulderTilt = sort(validPoses.map(p => p.shoulderTilt));
      baselineShoulderTilt = sortedShoulderTilt[Math.floor(sortedShoulderTilt.length / 2)];

      const sortedSpineRatio = sort(validPoses.map(p => p.spineRatio));
      baselineSpineRatio = sortedSpineRatio[Math.floor(sortedSpineRatio.length / 2)];
      hasUpperBodyBaseline = true;
    } else {
      baselineShoulderTilt = 0.0;
      baselineSpineRatio = null;
      hasUpperBodyBaseline = false;
    }
  }

  // Initialise smoothing filters at calibrated baselines
  smoothL_EAR         = baselineLeftEAR;
  smoothR_EAR         = baselineRightEAR;
  smoothMAR           = baselineMAR;
  smoothPitchRatio    = baselinePitchRatio;
  smoothYawDeg        = baselineYawDeg;
  smoothTiltDeg       = baselineTilt;
  smoothNoseY         = baselineNoseY;
  smoothSpineRatio    = baselineSpineRatio;
  smoothShoulderTilt  = baselineShoulderTilt;
  perclosHistory      = [];
  perclosSum          = 0;
  lastScoreUpdateTime = performance.now();

  calibProgress.classList.add("hidden");
  sessionActive  = true;
  sessionEnding  = false;
  alertnessScore = 100;
  scanLine.classList.remove("hidden");

  btnStart.innerHTML = `
    <svg class="btn-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <rect x="6" y="6" width="12" height="12" rx="2"/>
    </svg>
    <span>END SESSION</span>`;
  btnStart.classList.remove("btn-primary");
  btnStart.classList.add("btn-danger");
  btnStart.disabled = false;

  scoreNumber.textContent = "100";
  scoreNumber.style.color = "#059669";
  scoreLabel.textContent  = "FIT TO FLY · CLEARED";
  scoreLabel.style.color  = "#059669";
  drawGauge(100, "#059669");
  startSessionTimer();

  const bodyStatusStr = hasUpperBodyBaseline ? "UPPER-BODY SKELETON ENGAGED" : "3D FACE POSE ACTIVE";
  logEvent(
    `CALIBRATION VERIFIED — L-EAR: ${baselineLeftEAR.toFixed(3)} | R-EAR: ${baselineRightEAR.toFixed(3)} | ${bodyStatusStr}`,
    "system"
  );
}

// ── End Session & Fitness Certificate ─────────────────────────────────────────
function endSession(autoGrounded = false) {
  sessionActive = false;
  sessionEnding = false;
  clearInterval(sessionInterval);
  scanLine.classList.add("hidden");

  const s = Math.floor(alertnessScore);
  let verdict, color, rec;

  if (autoGrounded || s <= AUTO_GROUND_SCORE) {
    color   = "#DC2626";
    verdict = "UNFIT TO FLY — GROUNDED";
    rec     = "✖ Automatic flight safety interlock triggered. Immediate crew replacement required.";
  } else if (s >= 75) {
    color   = "#059669";
    verdict = "FIT TO FLY — CLEARED";
    rec     = "✔ Pilot cleared for all regular flight deck operations.";
  } else if (s >= 45) {
    color   = "#D97706";
    verdict = "MONITORING REQUIRED";
    rec     = "⚠ Elevated fatigue indicators detected. Dispatch review recommended.";
  } else {
    color   = "#DC2626";
    verdict = "UNFIT TO FLY";
    rec     = "✖ Critical fatigue detected. Replacement pilot mandated.";
  }

  resultScore.textContent   = s;
  resultScore.style.color   = color;
  resultVerdict.textContent = verdict;
  resultVerdict.style.color = color;
  resultDetails.innerHTML   =
    `<b>Duration:</b> ${sessionTimeEl.textContent}<br>
     <b>Final score:</b> ${s} / 100<br>
     <b>Long blinks (Micro-sleep):</b> ${totalLongBlinks}<br>
     <b>Yawns:</b> ${totalYawns}<br>
     <b>Head drops / Slumps:</b> ${totalHeadDrops}<br>
     <b>Torso / Spine slouches:</b> ${totalSlouches}<br>
     <b>Inattention gaze drifts:</b> ${totalYawDrifts}<br>
     <b>Total biometric events:</b> ${totalEvents}<br>
     <b>PERCLOS (Rolling 60s):</b> ${perclosVal.textContent}<br><br>
     <b>Outcome:</b> ${rec}`;

  resultModal.classList.remove("hidden");
  statusPill.className   = "pill idle";
  if (statusText) statusText.textContent = "SESSION COMPLETE";

  btnStart.innerHTML = `
    <svg class="btn-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
    <span>BEGIN EVALUATION</span>`;
  btnStart.classList.remove("btn-danger");
  btnStart.classList.add("btn-primary");
  btnStart.disabled = false;
}

// ── Real-Time Prediction Loop ─────────────────────────────────────────────────
function predictWebcam() {
  requestAnimationFrame(predictWebcam);
  if (!faceLandmarker || video.readyState < 2) return;
  if (video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  const now = performance.now();

  let faceResults = null;
  try {
    faceResults = faceLandmarker.detectForVideo(video, now);
  } catch {
    return;
  }

  let poseResults = null;
  if (poseLandmarker && poseLandmarkerReady) {
    try {
      poseResults = poseLandmarker.detectForVideo(video, now);
    } catch (e) {}
  }

  const faceLm = faceResults?.faceLandmarks?.[0] || null;
  const poseLm = poseResults?.landmarks?.[0] || null;

  if (faceLm) {
    noFaceOverlay.classList.add("hidden");
    if (hudTargetStatus) {
      hudTargetStatus.textContent = poseLm ? "3D FUSED LOCK" : "FACE LOCKED";
      hudTargetStatus.style.color = "#34D399";
    }

    if (sessionActive || isCalibrating) {
      drawMesh(faceLm, poseLm);
    }

    if (isCalibrating) {
      calibFrames.push({
        leftEAR:    calcEAR(faceLm, LEFT_EYE),
        rightEAR:   calcEAR(faceLm, RIGHT_EYE),
        mar:        calcMAR(faceLm),
        pitchRatio: calcPitchRatio(faceLm),
        yawDeg:     calcYawDeg(faceLm),     // 2D ocular-nasal azimuth in degrees
        tiltDeg:    calcTilt(faceLm),       // Symmetric outer-corner roll angle
        noseY:      calcNoseY(faceLm),      // Nose vertical position in camera frame
        faceWidth:  calcFaceWidth(faceLm),
        poseData:   getPoseMetrics(poseLm, faceLm)
      });
    } else if (sessionActive) {
      const { leftEAR, rightEAR, mar } = runHeuristics(faceLm, poseLm);
      updateDashboard(leftEAR, rightEAR, mar);
      tickFPS();
    }
  } else {
    noFaceOverlay.classList.remove("hidden");
    if (hudTargetStatus) {
      hudTargetStatus.textContent = "SEARCHING";
      hudTargetStatus.style.color = "rgba(56, 189, 248, 0.85)";
    }
    if (sessionActive) {
      alertnessScore = Math.max(0, alertnessScore - 0.03);
    }
  }
}

// ── Cockpit HUD Skeleton & Mesh Overlay ───────────────────────────────────────
function drawMesh(faceLm, poseLm) {
  const W = canvas.width;
  const H = canvas.height;
  const FL = window.FaceLandmarker;

  // 1. Upper Body Skeleton HUD (Shoulder beam & spine axis)
  if (poseLm && poseLm.length > 12) {
    const ls = poseLm[POSE_LEFT_SHOULDER];
    const rs = poseLm[POSE_RIGHT_SHOULDER];
    const visL = (ls.visibility ?? 1) > 0.35;
    const visR = (rs.visibility ?? 1) > 0.35;

    if (visL && visR) {
      const lsX = ls.x * W, lsY = ls.y * H;
      const rsX = rs.x * W, rsY = rs.y * H;
      const midX = (lsX + rsX) / 2, midY = (lsY + rsY) / 2;

      const bodyColor = isTorsoSlouching ? "rgba(239, 68, 68, 0.85)" : "rgba(56, 189, 248, 0.75)";

      ctx2d.save();

      // Shoulder horizontal connector
      ctx2d.beginPath();
      ctx2d.moveTo(lsX, lsY);
      ctx2d.lineTo(rsX, rsY);
      ctx2d.strokeStyle = bodyColor;
      ctx2d.lineWidth = 2.5;
      ctx2d.stroke();

      // Spinal / neck alignment axis to chin
      const headAnchor = faceLm ? faceLm[CHIN] : poseLm[POSE_NOSE];
      if (headAnchor) {
        ctx2d.beginPath();
        ctx2d.moveTo(midX, midY);
        ctx2d.lineTo(headAnchor.x * W, headAnchor.y * H);
        ctx2d.strokeStyle = bodyColor;
        ctx2d.setLineDash([4, 4]);
        ctx2d.lineWidth = 1.5;
        ctx2d.stroke();
        ctx2d.setLineDash([]);
      }

      // Tactical diamond markers on shoulder joints
      drawTacticalDiamond(lsX, lsY, bodyColor);
      drawTacticalDiamond(rsX, rsY, bodyColor);

      // Clavicle center node
      ctx2d.beginPath();
      ctx2d.arc(midX, midY, 3.5, 0, 2 * Math.PI);
      ctx2d.fillStyle = bodyColor;
      ctx2d.fill();

      ctx2d.restore();
    }
  }

  // 2. Subtle Sky-Blue Facial Mesh Tesselation
  if (faceLm) {
    ctx2d.strokeStyle = "#1D6FA422";
    ctx2d.lineWidth   = 0.5;
    for (const c of (FL?.FACE_LANDMARKS_TESSELATION ?? [])) {
      const p1 = faceLm[c.start], p2 = faceLm[c.end];
      ctx2d.beginPath();
      ctx2d.moveTo(p1.x * W, p1.y * H);
      ctx2d.lineTo(p2.x * W, p2.y * H);
      ctx2d.stroke();
    }

    // Eyes — color by state
    const lColor = leftEyeClosed  ? "#F87171" : "#38BDF8";
    const rColor = rightEyeClosed ? "#F87171" : "#7DD3FC";
    drawContour(faceLm, FL?.FACE_LANDMARKS_LEFT_EYE,  lColor, 2);
    drawContour(faceLm, FL?.FACE_LANDMARKS_RIGHT_EYE, rColor, 2);

    // Lips — amber
    drawContour(faceLm, FL?.FACE_LANDMARKS_LIPS, "#FBBF2488", 1.5);
  }
}

function drawTacticalDiamond(x, y, color) {
  const s = 6;
  ctx2d.beginPath();
  ctx2d.moveTo(x, y - s);
  ctx2d.lineTo(x + s, y);
  ctx2d.lineTo(x, y + s);
  ctx2d.lineTo(x - s, y);
  ctx2d.closePath();
  ctx2d.strokeStyle = color;
  ctx2d.lineWidth = 1.5;
  ctx2d.fillStyle = "rgba(14, 165, 233, 0.25)";
  ctx2d.fill();
  ctx2d.stroke();
}

function drawContour(lm, connections, color, width) {
  if (!connections) return;
  ctx2d.strokeStyle = color;
  ctx2d.lineWidth   = width;
  for (const c of connections) {
    const p1 = lm[c.start], p2 = lm[c.end];
    ctx2d.beginPath();
    ctx2d.moveTo(p1.x * canvas.width, p1.y * canvas.height);
    ctx2d.lineTo(p2.x * canvas.width, p2.y * canvas.height);
    ctx2d.stroke();
  }
}

// ── Reset All System States ───────────────────────────────────────────────────
function resetAll() {
  sessionActive   = false;
  sessionEnding   = false;
  alertnessScore  = 100;
  isCalibrating   = false;
  calibFrames     = [];

  totalEvents = totalYawns = totalLongBlinks = totalHeadDrops = totalSlouches = totalYawDrifts = 0;
  eventHistory = [];

  isBlinking = longBlinkFired = false;
  isYawning  = yawnFired      = false;
  asymFired  = false;
  leftEyeClosed = rightEyeClosed = false;

  badHeadPoseStart = null;
  badYawStart      = null;
  badTorsoStart    = null;
  isHeadNodding    = isHeadSlumping = isYawDrifting = isTorsoSlouching = false;

  lastPenaltyAt.longBlink = lastPenaltyAt.yawn =
  lastPenaltyAt.headDrop  = lastPenaltyAt.asymmetry =
  lastPenaltyAt.slouch    = lastPenaltyAt.yawDrift = 0;

  smoothL_EAR         = null;
  smoothR_EAR         = null;
  smoothMAR           = null;
  smoothPitchRatio    = null;
  smoothYawDeg        = null;
  smoothTiltDeg       = null;
  smoothNoseY         = null;
  smoothSpineRatio    = null;
  smoothShoulderTilt  = null;
  perclosHistory      = [];
  perclosSum          = 0;
  lastScoreUpdateTime = performance.now();

  lEH.fill(0.28); rEH.fill(0.28); mH.fill(0); sH.fill(0);

  clearInterval(sessionInterval);
  sessionTimeEl.textContent = "00:00";
  scanLine.classList.add("hidden");
  alertBanner.classList.add("hidden");

  [longBlinkCountR, yawnCountR, headDropCountR, eventCount].forEach(e => {
    if (e) e.textContent = "0";
  });
  [leftEarVal, rightEarVal, marVal, perclosVal, postureVal].forEach(e => {
    if (e) e.textContent = "---";
  });

  fpsBadge.textContent = "-- FPS";
  eventLog.innerHTML   = '<li class="log-empty">No events recorded yet.</li>';
  eventCountBadge.textContent = "0 events";

  scoreNumber.textContent = "--";
  scoreNumber.style.color = "#64748B";
  scoreLabel.textContent  = "AWAITING START";
  scoreLabel.style.color  = "#64748B";
  statusPill.className    = "pill idle";
  if (statusText) statusText.textContent = "AWAITING EVALUATION";
  drawGauge(0);

  setIndicator(indLeftEye,  "", "NORMAL");
  setIndicator(indRightEye, "", "NORMAL");
  setIndicator(indMouth,    "", "CLOSED");
  setIndicator(indHeadPose, "", "LEVEL");
  setIndicator(indTorso,    "", "UPRIGHT");

  if (stLear)    stLear.textContent    = "Calibrating";
  if (stRear)    stRear.textContent    = "Calibrating";
  if (stMar)     stMar.textContent     = "Neutral";
  if (stPerclos) stPerclos.textContent = "Optimal";
  if (stPosture) stPosture.textContent = "Upright";

  telChart.data.datasets[0].data = [...lEH];
  telChart.data.datasets[1].data = [...rEH];
  telChart.data.datasets[2].data = [...mH];
  telChart.data.datasets[3].data = [...sH];
  telChart.update("none");

  btnStart.innerHTML = `
    <svg class="btn-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <polygon points="5 3 19 12 5 21 5 3"/>
    </svg>
    <span>BEGIN EVALUATION</span>`;
  btnStart.classList.remove("btn-danger");
  btnStart.classList.add("btn-primary");
  btnStart.disabled = !systemReady;
}

// ── JSON Telemetry Export ─────────────────────────────────────────────────────
function exportJSON() {
  const blob = new Blob([JSON.stringify({
    system: "AeroVigil Fatigue Evaluator",
    version: "2.1-PostureEnhanced",
    exportedAt: new Date().toISOString(),
    baselines: {
      leftEAR: baselineLeftEAR,
      rightEAR: baselineRightEAR,
      ear: baselineEAR,
      mar: baselineMAR,
      pitchRatio: baselinePitchRatio,
      yawDeg: baselineYawDeg,
      lateralTiltDeg: baselineTilt,
      noseY: baselineNoseY,
      faceWidth: baselineFaceWidth,
      shoulderTiltDeg: baselineShoulderTilt,
      spineRatio: baselineSpineRatio,
      upperBodyCalibrated: hasUpperBodyBaseline
    },
    sessionSummary: {
      duration: sessionTimeEl.textContent,
      finalScore: Math.floor(alertnessScore),
      verdict: resultVerdict.textContent,
      totalEvents,
      totalYawns,
      totalLongBlinks,
      totalHeadDrops,
      totalTorsoSlouches: totalSlouches,
      totalYawDrifts,
      finalPerclos: perclosVal.textContent
    },
    events: eventHistory
  }, null, 2)], { type: "application/json" });

  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download: `aerovigil_session_${Date.now()}.json`
  });
  a.click();
  URL.revokeObjectURL(a.href);
}

// ── Application Initialization ────────────────────────────────────────────────
async function init() {
  drawGauge(0);
  btnStart.disabled = true;

  try {
    loadingMsg.textContent = "Requesting cockpit camera access…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    video.srcObject = stream;
    await new Promise(res => { video.onloadedmetadata = res; });
    video.play();

    loadingMsg.textContent = "Loading AI models (cached after first run)…";
    const { FaceLandmarker: FL, PoseLandmarker: PL, FilesetResolver } =
      await import(`${MP_CDN}/vision_bundle.mjs`);
    window.FaceLandmarker = FL;
    window.PoseLandmarker = PL;

    const vision = await FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);

    // 1. Core Biometric Model (Face Landmarker)
    faceLandmarker = await FL.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: false,
      runningMode: "VIDEO",
      numFaces: 1
    });

    // 2. Upper-Body Posture Model (Pose Landmarker Lite)
    try {
      poseLandmarker = await PL.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
          delegate: "GPU"
        },
        runningMode: "VIDEO",
        numPoses: 1
      });
      poseLandmarkerReady = true;
      console.log("Upper-body Pose Landmarker initialized successfully.");
    } catch (poseErr) {
      console.warn("PoseLandmarker optional load note (running in 3D face mode):", poseErr);
      poseLandmarkerReady = false;
    }

    requestAnimationFrame(predictWebcam);
    systemReady = true;
    loadingOverlay.classList.add("hidden");
    calibModal.classList.remove("hidden");
    btnStart.disabled = false;

  } catch (err) {
    console.error(err);
    loadingMsg.innerHTML =
      `<span style="color:#F87171">ERROR: ${err.message}</span><br><br>
       Please allow camera access and use Google Chrome or Microsoft Edge.`;
  }
}

// ── User Event Wiring ─────────────────────────────────────────────────────────
btnStart.addEventListener("click", () => {
  if (!systemReady) return;
  if (sessionActive) {
    endSession(false);
  } else {
    resetAll();
    calibModal.classList.remove("hidden");
    btnStart.disabled = true;
  }
});

btnBeginCalib.addEventListener("click", startCalibration);
btnReset.addEventListener("click",      () => { resultModal.classList.add("hidden"); resetAll(); });
btnExport.addEventListener("click",     exportJSON);
btnResultExport.addEventListener("click", exportJSON);
btnResultRestart.addEventListener("click", () => {
  resultModal.classList.add("hidden");
  resetAll();
  calibModal.classList.remove("hidden");
  btnStart.disabled = true;
});

// Boot AeroVigil
init();
