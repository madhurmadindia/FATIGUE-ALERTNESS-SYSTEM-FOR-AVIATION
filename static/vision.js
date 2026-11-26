/**
 * AeroVigil — camera signals.
 *
 * Two MediaPipe models share one video stream:
 *   HandLandmarker — the pilot's answer (1-5 fingers) and the rest pose.
 *   FaceLandmarker — supporting signals only (face present, looking at screen,
 *                    eye closure). These never decide the verdict; they are
 *                    sleepiness/arousal signals and a data-quality check.
 *
 * The geometry helpers are pure and are exercised by tests/selftest.mjs.
 */

const MP_VERSION = "0.10.3";
const MP_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;

const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// ── Face landmark indices (carried over from the previous prototype) ─────────
const LEFT_EYE = [362, 263, 385, 387, 380, 373];
const RIGHT_EYE = [33, 133, 160, 158, 144, 153];
const FOREHEAD = 10;
const CHIN = 152;
const L_TRAGUS = 234;
const R_TRAGUS = 454;
const UPPER_LIP = 13;
const LOWER_LIP = 14;
const MOUTH_LEFT = 308;
const MOUTH_RIGHT = 78;

// ── Hand landmark indices ────────────────────────────────────────────────────
const WRIST = 0;
const MIDDLE_MCP = 9;
const PINKY_MCP = 17;
const TIP = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const PIP = { thumb: 3, index: 6, middle: 10, ring: 14, pinky: 18 };
const LONG_FINGERS = ["index", "middle", "ring", "pinky"];

export const REST_SIGN = 0; // closed fist

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Which fingers are extended.
 *
 * Decided from distance to the wrist rather than from vertical position on
 * screen, so a rotated or tilted hand still reads correctly — screen-height
 * tests fail as soon as the pilot angles their wrist.
 */
export function fingerStates(lm) {
  const w = lm[WRIST];
  const out = {};
  for (const name of LONG_FINGERS) {
    out[name] = dist(w, lm[TIP[name]]) > dist(w, lm[PIP[name]]) * 1.12;
  }
  // The thumb folds sideways, so it is measured across the palm instead.
  out.thumb = dist(lm[TIP.thumb], lm[PINKY_MCP]) > dist(lm[PIP.thumb], lm[PINKY_MCP]) * 1.1;
  return out;
}

/**
 * Map a hand to an answer.
 *   0     = rest (closed fist)
 *   1..4  = that many long fingers, thumb tucked
 *   5     = open palm
 *   null  = ambiguous — rejected rather than guessed, so a tracker failure
 *           never gets scored as a wrong answer by the pilot.
 */
export function handSign(lm) {
  if (!lm || lm.length < 21) return null;
  const f = fingerStates(lm);
  const n = LONG_FINGERS.filter((k) => f[k]).length;
  if (n === 0) return f.thumb ? null : REST_SIGN;
  if (n === 4) return f.thumb ? 5 : 4;
  return f.thumb ? null : n;
}

/** Rough hand scale in normalised units — used to spot a hand that is too far away. */
export function handScale(lm) {
  return dist(lm[WRIST], lm[MIDDLE_MCP]);
}

/** True when the hand overlaps the face box, which breaks the eye signals. */
export function handCoversFace(handLm, faceLm) {
  if (!handLm || !faceLm) return false;
  const xs = faceLm.map((p) => p.x);
  const ys = faceLm.map((p) => p.y);
  const pad = 0.02;
  const x0 = Math.min(...xs) - pad, x1 = Math.max(...xs) + pad;
  const y0 = Math.min(...ys) - pad, y1 = Math.max(...ys) + pad;
  return handLm.some((p) => p.x > x0 && p.x < x1 && p.y > y0 && p.y < y1);
}

// ── Face signals ─────────────────────────────────────────────────────────────

function dist2D(p1, p2, w, h) {
  return Math.hypot((p1.x - p2.x) * w, (p1.y - p2.y) * h);
}

/** Eye aspect ratio for one eye. */
export function eyeAspectRatio(lm, idx, w, h) {
  const v1 = dist2D(lm[idx[2]], lm[idx[4]], w, h);
  const v2 = dist2D(lm[idx[3]], lm[idx[5]], w, h);
  const span = dist2D(lm[idx[0]], lm[idx[1]], w, h);
  return span < 1e-4 ? null : (v1 + v2) / (2 * span);
}

/**
 * Horizontal head turn in degrees, from how far the nose sits off the midpoint
 * of the outer eye corners. Only needs 2D landmarks, so it survives poor depth.
 */
export function yawDegrees(lm, w) {
  const rEyeX = lm[33].x * w;
  const lEyeX = lm[362].x * w;
  const noseX = lm[1].x * w;
  const span = lEyeX - rEyeX;
  if (span < 15) return null;
  const off = (noseX - (lEyeX + rEyeX) / 2) / (span * 0.45);
  return Math.max(-1, Math.min(1, off)) * 45;
}

/** Face height-to-width ratio; drops when the head pitches forward. */
export function pitchRatio(lm, w, h) {
  const faceH = dist2D(lm[FOREHEAD], lm[CHIN], w, h);
  const faceW = dist2D(lm[L_TRAGUS], lm[R_TRAGUS], w, h);
  return faceW < 1e-4 ? null : faceH / faceW;
}

export function mouthAspectRatio(lm, w, h) {
  const v = dist2D(lm[UPPER_LIP], lm[LOWER_LIP], w, h);
  const span = dist2D(lm[MOUTH_LEFT], lm[MOUTH_RIGHT], w, h);
  return span < 1e-4 ? null : v / span;
}

export function faceSignals(lm, w, h) {
  if (!lm) return null;
  const left = eyeAspectRatio(lm, LEFT_EYE, w, h);
  const right = eyeAspectRatio(lm, RIGHT_EYE, w, h);
  const yaw = yawDegrees(lm, w);
  return {
    leftEar: left,
    rightEar: right,
    ear: left != null && right != null ? (left + right) / 2 : null,
    mar: mouthAspectRatio(lm, w, h),
    yaw,
    pitchRatio: pitchRatio(lm, w, h),
    looking: yaw != null && Math.abs(yaw) < 40,
  };
}

// ── Model loading and the per-frame loop ─────────────────────────────────────

export class Vision {
  constructor() {
    this.face = null;
    this.hand = null;
    this.ready = false;
    this.frame = 0;
    this.lastFace = null;
  }

  async load(onProgress = () => {}) {
    onProgress("Loading vision models…");
    const { FaceLandmarker, HandLandmarker, FilesetResolver } = await import(
      `${MP_CDN}/vision_bundle.mjs`
    );
    const fileset = await FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);

    this.hand = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
    });
    onProgress("Loading face model…");
    this.face = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      outputFaceBlendshapes: false,
      numFaces: 1,
    });
    this.ready = true;
  }

  /**
   * Run both models on the current video frame.
   *
   * The hand runs every frame because it carries the reaction time; the face
   * runs every third frame because it only supplies slow supporting signals,
   * and two models at full rate is too slow on a tablet.
   */
  detect(video, now, w, h) {
    if (!this.ready) return null;
    this.frame++;

    let handLm = null;
    try {
      handLm = this.hand.detectForVideo(video, now)?.landmarks?.[0] ?? null;
    } catch {
      handLm = null;
    }

    if (this.frame % 3 === 0) {
      try {
        this.lastFace = this.face.detectForVideo(video, now + 0.5)?.faceLandmarks?.[0] ?? null;
      } catch {
        this.lastFace = null;
      }
    }
    const faceLm = this.lastFace;

    return {
      t: now,
      handLm,
      faceLm,
      sign: handLm ? handSign(handLm) : null,
      handScale: handLm ? handScale(handLm) : null,
      face: faceSignals(faceLm, w, h),
      occluded: handCoversFace(handLm, faceLm),
    };
  }
}
