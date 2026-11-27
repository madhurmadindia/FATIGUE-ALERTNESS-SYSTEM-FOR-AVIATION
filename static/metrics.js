/**
 * AeroVigil — reaction-time metrics, personal baseline and verdict.
 *
 * Pure functions only: no DOM, no MediaPipe. Exercised by tests/selftest.mjs.
 *
 * Metric choice follows Basner & Dinges 2011, which found reciprocal reaction
 * time and lapse counts far more sensitive to sleep loss than mean or median RT.
 * Every threshold below is a PLACEHOLDER until the validation study in the
 * project report produces real cut-offs.
 */

// Touch block: adapted from PVT-B (Basner, Mollicone & Dinges 2011). The dot
// appears at a random position, adding scan + reach time vs a fixed-button lab
// PVT, so the lapse threshold is raised from the lab's 355 ms to 550 ms.
export const TOUCH_LAPSE_MS = 550;
export const FALSE_START_MS = 100;

// Gesture block: no published lapse threshold exists for a colour-to-hand-sign
// task, so we deliberately do not define one and score slowest-10% instead.
export const GESTURE_TIMEOUT_MS = 3500;

// shortcut: fixed verdict cut-offs on the pooled z-score, chosen to be
// conservative rather than validated. Replace with ROC-derived values once the
// sleep-restriction study has data.
export const Z_FATIGUED = 2.0;
export const Z_CLEAR = 1.0;

// A pilot's own session-to-session spread is never truly zero, and five
// sessions estimate it poorly. Without a floor, a consistent baseline would
// make every later session look wildly impaired. The floor is the larger of an
// absolute value and a fraction of the pilot's own mean, because the scale of
// these metrics differs between people.
// shortcut: both numbers are provisional, deliberately generous so the tool
// under-calls rather than over-calls fatigue. Replace once the study has data.
export const SD_FLOOR = { invRt: 0.2, slowestInvRt: 0.15, lapses: 1.5, medianMs: 30 };
export const SD_FLOOR_FRACTION = 0.08;

// Population norms (healthy rested adults) for a random-position dot on a tablet.
// Touch norms are adjusted from Basner & Dinges 2011 to account for the scan +
// reach time the random dot adds (~200 ms). Gesture norms are estimates.
// shortcut: replace with study-derived norms once the validation data exists.
export const POPULATION_NORMS = {
  touchInvRt:     { mean: 2.3, sd: 0.5 },
  touchSlowest:   { mean: 1.6, sd: 0.4 },
  touchLapses:    { mean: 3.0, sd: 2.5 },
  gestureMedian:  { mean: 950, sd: 200 },
  gestureSlowest: { mean: 1.0, sd: 0.3 },
};

export const BASELINE_TARGET = 5;      // rested sessions needed before a full verdict
export const FAMILIARISATION = 0;      // disabled for initial deployment

export const VERDICT = {
  NOT_FATIGUED: "NOT_FATIGUED",
  FATIGUED: "FATIGUED",
  RETEST: "RETEST",
};

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

function sd(xs) {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
}

function median(xs) {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.floor(s.length / 2);
  return s.length % 2 ? s[i] : (s[i - 1] + s[i]) / 2;
}

/**
 * Summarise one block of trials.
 *
 * A trial is { rt, falseStart, rejected, correct }. `rt` is null for a
 * no-response. Rejected trials (tracking lost) never count as pilot errors —
 * conflating the two would make a flaky hand tracker look like a tired pilot.
 */
export function blockStats(trials, { lapseMs = null } = {}) {
  const attempted = trials.filter((t) => !t.rejected);
  const falseStarts = attempted.filter((t) => t.falseStart).length;
  const scored = attempted.filter((t) => !t.falseStart && t.rt != null);
  const correct = scored.filter((t) => t.correct !== false);

  // Speed is scored on correct responses only: a wrong answer's RT measures a
  // different process. Accuracy is reported separately.
  const rts = correct.map((t) => t.rt);
  const stats = {
    trials: trials.length,
    rejected: trials.filter((t) => t.rejected).length,
    attempted: attempted.length,
    scored: scored.length,
    falseStarts,
    noResponse: attempted.filter((t) => !t.falseStart && t.rt == null).length,
    accuracy: scored.length ? correct.length / scored.length : null,
    rejectRate: trials.length ? trials.filter((t) => t.rejected).length / trials.length : 0,
    medianMs: median(rts),
    invRt: null,
    slowestInvRt: null,
    lapses: lapseMs == null ? null : rts.filter((rt) => rt >= lapseMs).length,
  };

  if (rts.length) {
    // Reciprocal RT in responses per second. Averaging 1/RT rather than RT stops
    // a handful of very slow trials from dominating, which is why Basner &
    // Dinges recommend it over mean RT.
    const inv = rts.map((rt) => 1000 / rt);
    stats.invRt = mean(inv);
    const n = Math.max(1, Math.ceil(inv.length * 0.1));
    stats.slowestInvRt = mean([...inv].sort((a, b) => a - b).slice(0, n));
  }
  return stats;
}

/** Fraction of frames where the face was present / looking at the screen, plus frame timing. */
export function qualityFromSignals({ frames, facePresent, looking, frameIntervals }) {
  const intervals = frameIntervals ?? [];
  return {
    frames,
    facePresentPct: frames ? facePresent / frames : 0,
    lookingPct: frames ? looking / frames : 0,
    meanFrameMs: intervals.length ? mean(intervals) : null,
    // Jitter, not mean delay, is what corrupts a reaction time: a constant lag is
    // absorbed by the pilot's own baseline, a varying one is noise.
    frameJitterMs: intervals.length > 1 ? sd(intervals) : null,
  };
}

/**
 * Decide whether a session is good enough to judge at all.
 * Returns { ok, reasons[] }.
 */
export function dataQuality({ touch, gesture, signals }) {
  const reasons = [];
  if (touch.scored < 8) reasons.push("too few valid taps in the touch block");
  if (gesture.scored < 12) reasons.push("too few valid trials in the hand-sign block");
  if (gesture.rejectRate > 0.35) reasons.push("hand tracking was lost too often");
  if (touch.falseStarts > Math.max(3, touch.attempted * 0.25)) reasons.push("too many early taps");
  if (signals.facePresentPct < 0.7) reasons.push("your face was out of view too often");
  if (signals.frameJitterMs != null && signals.frameJitterMs > 25) reasons.push("camera timing was unstable");
  return { ok: reasons.length === 0, reasons };
}

/**
 * Drowsiness indicators from accumulated face signals.
 * These run silently during both test blocks and contribute to the final
 * verdict without being shown to the pilot during the test.
 */
export function drowsinessScore({ earSamples, marSamples }) {
  if (!earSamples?.length) return { score: 0, perclos: null, yawns: 0, meanEar: null, details: [] };

  const EAR_CLOSED = 0.21;
  const MAR_YAWN = 0.55;

  const perclos = earSamples.filter((e) => e < EAR_CLOSED).length / earSamples.length;
  const meanEar = mean(earSamples);

  let yawns = 0, inYawn = false;
  for (const m of marSamples ?? []) {
    if (m > MAR_YAWN && !inYawn) { yawns++; inYawn = true; }
    if (m <= MAR_YAWN) inYawn = false;
  }

  const details = [];
  let score = 0;
  if (perclos > 0.08) {
    score += Math.min((perclos - 0.08) * 15, 2);
    details.push("elevated eye closure");
  }
  if (yawns >= 1) { score += 0.5; details.push(`${yawns} yawn${yawns > 1 ? "s" : ""} detected`); }
  if (yawns >= 3) score += 0.5;
  if (meanEar != null && meanEar < 0.22) { score += 0.5; details.push("low average eye opening"); }

  return { score: Math.min(score, 3), perclos, yawns, meanEar, details };
}

/** Mean and spread of one metric across stored baseline sessions. */
export function baselineFor(sessions, pick, floor) {
  const xs = sessions.map(pick).filter((x) => x != null && Number.isFinite(x));
  if (xs.length < 2) return null;
  const m = mean(xs);
  return {
    mean: m,
    sd: Math.max(sd(xs), floor, Math.abs(m) * SD_FLOOR_FRACTION),
    n: xs.length,
  };
}

/** Signed z-score where positive always means worse than baseline. */
export function zScore(value, base, { higherIsWorse }) {
  if (base == null || value == null || !Number.isFinite(value)) return null;
  const z = (value - base.mean) / base.sd;
  return higherIsWorse ? z : -z;
}

/**
 * Compare today's session against the pilot's own rested baseline.
 *
 * The touch block is weighted above the hand-sign block because simple reaction
 * time is the measure with published sleep-loss sensitivity; the hand-sign task
 * is still unvalidated (see the project report).
 */
export function compareToBaseline(today, baselineSessions) {
  const picks = [
    ["touchInvRt", (s) => s.touch.invRt, SD_FLOOR.invRt],
    ["touchSlowest", (s) => s.touch.slowestInvRt, SD_FLOOR.slowestInvRt],
    ["touchLapses", (s) => s.touch.lapses, SD_FLOOR.lapses],
    ["gestureMedian", (s) => s.gesture.medianMs, SD_FLOOR.medianMs],
    ["gestureSlowest", (s) => s.gesture.slowestInvRt, SD_FLOOR.slowestInvRt],
  ];

  let usingPopulationNorms = false;
  const b = {};
  for (const [key, pick, floor] of picks) {
    const personal = baselineFor(baselineSessions, pick, floor);
    if (personal) { b[key] = personal; }
    else { b[key] = POPULATION_NORMS[key]; usingPopulationNorms = true; }
  }

  const parts = [
    { key: "touchInvRt", label: "Tap speed", weight: 2, z: zScore(today.touch.invRt, b.touchInvRt, { higherIsWorse: false }) },
    { key: "touchSlowest", label: "Slowest taps", weight: 2, z: zScore(today.touch.slowestInvRt, b.touchSlowest, { higherIsWorse: false }) },
    { key: "touchLapses", label: "Lapses", weight: 1, z: zScore(today.touch.lapses, b.touchLapses, { higherIsWorse: true }) },
    { key: "gestureMedian", label: "Hand-sign speed", weight: 1, z: zScore(today.gesture.medianMs, b.gestureMedian, { higherIsWorse: true }) },
    { key: "gestureSlowest", label: "Slowest hand signs", weight: 1, z: zScore(today.gesture.slowestInvRt, b.gestureSlowest, { higherIsWorse: false }) },
  ].filter((p) => p.z != null);

  if (!parts.length) return { impairment: null, parts: [], baseline: b, usingPopulationNorms };

  const wsum = parts.reduce((a, p) => a + p.weight, 0);
  const impairment = parts.reduce((a, p) => a + p.z * p.weight, 0) / wsum;
  return { impairment, parts, baseline: b, usingPopulationNorms };
}

/**
 * Final verdict. Three outcomes, not two: at the low prevalence of real
 * impairment before a duty, forcing a yes/no answer on a noisy reading would
 * flag far more rested pilots than tired ones.
 */
export function decideVerdict({ impairment, quality, baselineCount, drowsiness }) {
  if (!quality.ok) {
    return { verdict: VERDICT.RETEST, reason: quality.reasons[0], reasons: quality.reasons };
  }
  if (impairment == null) {
    return { verdict: VERDICT.RETEST, reason: "not enough comparable data", reasons: [] };
  }

  // Drowsiness signals (PERCLOS, yawns, eye closure) adjust the impairment
  // upward — they can make a verdict more cautious but never less.
  let adjusted = impairment;
  if (drowsiness && drowsiness.score > 0) {
    adjusted = impairment + drowsiness.score * 0.3;
  }

  const provisional = baselineCount < BASELINE_TARGET;
  if (adjusted >= Z_FATIGUED) {
    return { verdict: VERDICT.FATIGUED, reason: "performance clearly below your normal range", reasons: [], provisional };
  }
  if (adjusted <= Z_CLEAR) {
    return { verdict: VERDICT.NOT_FATIGUED, reason: "performance within your normal range", reasons: [], provisional };
  }
  return { verdict: VERDICT.RETEST, reason: "result is borderline", reasons: [], provisional };
}

/**
 * Check-in answers never move the verdict (a pilot can under-report), but a
 * clear contradiction between what they said and how they performed is worth
 * recording for the operator's fatigue data.
 */
export function selfReportMismatch(checkin, impairment) {
  if (impairment == null || checkin?.rating == null) return null;
  if (checkin.rating <= 2 && impairment >= Z_FATIGUED) return "rated themselves alert but performed well below baseline";
  if (checkin.rating >= 6 && impairment <= Z_CLEAR) return "rated themselves exhausted but performed normally";
  return null;
}
