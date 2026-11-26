/**
 * Self-check for the pure logic behind the pre-flight check.
 * Run: node tests/selftest.mjs
 *
 * The modules are loaded from source text as data: URLs because the project
 * has no package.json, so node would otherwise read static/*.js as CommonJS.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const here = dirname(fileURLToPath(import.meta.url));
const load = (f) =>
  import("data:text/javascript;base64," +
    Buffer.from(readFileSync(join(here, "..", "static", f), "utf8")).toString("base64"));

const { handSign, fingerStates, handCoversFace, mouthAspectRatio, REST_SIGN } = await load("vision.js");
const M = await load("metrics.js");

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed++;
  } catch (err) {
    console.error(`FAIL  ${name}\n      ${err.message}`);
    process.exitCode = 1;
  }
}

// ── Hand landmark fixtures ───────────────────────────────────────────────────
// A right hand held palm-to-camera, fingers pointing up (y decreases upward).
function hand({ index = false, middle = false, ring = false, pinky = false, thumb = false } = {}) {
  const lm = new Array(21);
  lm[0] = { x: 0.50, y: 0.95 };                                   // wrist
  const cols = { index: 0.44, middle: 0.50, ring: 0.56, pinky: 0.62 };
  const mcpIdx = { index: 5, middle: 9, ring: 13, pinky: 17 };
  const up = { index, middle, ring, pinky };
  for (const [name, x] of Object.entries(cols)) {
    const b = mcpIdx[name];
    lm[b]     = { x, y: 0.70 };                                   // MCP
    lm[b + 1] = { x, y: 0.60 };                                   // PIP
    lm[b + 2] = { x, y: up[name] ? 0.53 : 0.65 };                 // DIP
    lm[b + 3] = { x, y: up[name] ? 0.46 : 0.68 };                 // TIP, curled back when folded
  }
  lm[1] = { x: 0.42, y: 0.88 };
  lm[2] = { x: 0.36, y: 0.82 };
  lm[3] = { x: 0.32, y: 0.78 };
  lm[4] = thumb ? { x: 0.26, y: 0.74 } : { x: 0.42, y: 0.74 };    // tucked thumb sits over the palm
  return lm;
}

check("fist reads as the rest pose", () => {
  assert.equal(handSign(hand()), REST_SIGN);
});

check("one finger reads as 1", () => {
  assert.equal(handSign(hand({ index: true })), 1);
});

check("two fingers read as 2", () => {
  assert.equal(handSign(hand({ index: true, middle: true })), 2);
});

check("three fingers read as 3", () => {
  assert.equal(handSign(hand({ index: true, middle: true, ring: true })), 3);
});

check("four fingers with the thumb tucked read as 4", () => {
  assert.equal(handSign(hand({ index: true, middle: true, ring: true, pinky: true })), 4);
});

check("open palm reads as 5", () => {
  assert.equal(handSign(hand({ index: true, middle: true, ring: true, pinky: true, thumb: true })), 5);
});

check("an ambiguous pose is rejected rather than guessed", () => {
  // Thumb out with two fingers is not in the answer set: it must not score as 2,
  // or a tracker wobble would be recorded as the pilot answering wrongly.
  assert.equal(handSign(hand({ index: true, middle: true, thumb: true })), null);
  assert.equal(handSign(null), null);
  assert.equal(handSign([{ x: 0, y: 0 }]), null);
});

check("finger states identify which fingers are up", () => {
  const f = fingerStates(hand({ index: true, pinky: true }));
  assert.equal(f.index, true);
  assert.equal(f.pinky, true);
  assert.equal(f.middle, false);
  assert.equal(f.thumb, false);
});

check("a hand over the face is detected", () => {
  const face = [{ x: 0.40, y: 0.30 }, { x: 0.60, y: 0.70 }];
  const over = [{ x: 0.50, y: 0.50 }];
  const beside = [{ x: 0.85, y: 0.50 }];
  assert.equal(handCoversFace(over, face), true);
  assert.equal(handCoversFace(beside, face), false);
  assert.equal(handCoversFace(null, face), false);
});

// ── Block statistics ─────────────────────────────────────────────────────────
const t = (rt, extra = {}) => ({ rt, falseStart: false, rejected: false, correct: true, ...extra });

check("block stats separate rejects, false starts and lapses", () => {
  const s = M.blockStats([
    t(200), t(300), t(400),
    t(null, { falseStart: true }),
    t(null),
    t(999, { rejected: true }),
  ], { lapseMs: 550 });

  assert.equal(s.trials, 6);
  assert.equal(s.rejected, 1);
  assert.equal(s.attempted, 5);
  assert.equal(s.scored, 3);
  assert.equal(s.falseStarts, 1);
  assert.equal(s.noResponse, 1);
  assert.equal(s.lapses, 0, "no trial exceeds 550 ms lapse threshold");
  assert.equal(s.medianMs, 300);
  // Reciprocal RT in responses per second: mean of 5, 3.33 and 2.5.
  assert.ok(Math.abs(s.invRt - 3.6111) < 1e-3, `invRt was ${s.invRt}`);
  // Slowest 10% of 3 trials rounds up to 1 trial: the 400 ms one.
  assert.ok(Math.abs(s.slowestInvRt - 2.5) < 1e-9);
});

check("a rejected trial never counts as a wrong answer", () => {
  const s = M.blockStats([t(500), t(null, { rejected: true, correct: false })]);
  assert.equal(s.accuracy, 1, "the rejected trial must not reduce accuracy");
  assert.ok(Math.abs(s.rejectRate - 0.5) < 1e-9);
});

check("wrong answers are excluded from the speed metrics", () => {
  const s = M.blockStats([t(300), t(900, { correct: false })]);
  assert.equal(s.medianMs, 300, "a wrong answer's RT measures a different process");
  assert.equal(s.accuracy, 0.5);
});

check("an empty block does not throw", () => {
  const s = M.blockStats([], { lapseMs: 550 });
  assert.equal(s.scored, 0);
  assert.equal(s.medianMs, null);
  assert.equal(s.invRt, null);
  assert.equal(s.accuracy, null);
});

// ── Baselines and z-scores ───────────────────────────────────────────────────
check("positive z always means worse, whichever way the metric runs", () => {
  const base = { mean: 4, sd: 0.5 };
  // Reciprocal RT: lower is slower, so lower must give positive z.
  assert.ok(M.zScore(3, base, { higherIsWorse: false }) > 0);
  assert.ok(M.zScore(5, base, { higherIsWorse: false }) < 0);
  // Lapse count: higher is worse.
  assert.ok(M.zScore(5, base, { higherIsWorse: true }) > 0);
  assert.equal(M.zScore(null, base, { higherIsWorse: true }), null);
  assert.equal(M.zScore(3, null, { higherIsWorse: true }), null);
});

check("the baseline spread has a floor", () => {
  // Five identical sessions would give sd 0 and make any later session look
  // infinitely impaired; the floor stops that.
  const sessions = Array.from({ length: 5 }, () => ({ touch: { invRt: 4 } }));
  const b = M.baselineFor(sessions, (s) => s.touch.invRt, M.SD_FLOOR.invRt);
  assert.ok(b.sd > 0);
  assert.equal(b.n, 5);
  // The floor scales with the pilot's own mean, because these metrics have no
  // common scale between people.
  assert.equal(b.sd, 4 * M.SD_FLOOR_FRACTION);
  const small = M.baselineFor(
    Array.from({ length: 5 }, () => ({ touch: { invRt: 0.1 } })),
    (s) => s.touch.invRt, M.SD_FLOOR.invRt,
  );
  assert.equal(small.sd, M.SD_FLOOR.invRt, "the absolute floor still applies to tiny means");
});

check("a consistent baseline does not produce absurd z-scores", () => {
  // Regression: an absolute-only floor made a modestly slower session score
  // past z = 20, which is meaningless from five baseline sessions.
  const flat = Array.from({ length: 5 }, () => session(4.0, 760));
  const { impairment } = M.compareToBaseline(session(2.6, 1150), flat);
  assert.ok(impairment > 2, "a clearly slower session must still read as impaired");
  assert.ok(impairment < 12, `impairment was implausibly large: ${impairment}`);
});

check("a single session is not a baseline", () => {
  assert.equal(M.baselineFor([{ touch: { invRt: 4 } }], (s) => s.touch.invRt, 0.1), null);
});

function session(touchInv, gestureMs) {
  return {
    touch: { invRt: touchInv, slowestInvRt: touchInv * 0.7, lapses: 1, medianMs: 1000 / touchInv },
    gesture: { medianMs: gestureMs, slowestInvRt: 1000 / (gestureMs * 1.4) },
  };
}
const baseline = [session(4.0, 760), session(3.9, 780), session(4.1, 750), session(4.0, 770), session(3.95, 765)];

check("a session like the baseline scores near zero", () => {
  const { impairment } = M.compareToBaseline(session(4.0, 765), baseline);
  assert.ok(Math.abs(impairment) < 1, `impairment was ${impairment}`);
});

check("a clearly slower session scores positive", () => {
  const { impairment } = M.compareToBaseline(session(2.6, 1150), baseline);
  assert.ok(impairment > 2, `impairment was ${impairment}`);
});

check("comparison with no baseline falls back to population norms", () => {
  const { impairment, parts, usingPopulationNorms } = M.compareToBaseline(session(4.0, 765), []);
  assert.ok(impairment != null, "population norms should provide a comparison");
  assert.ok(parts.length > 0, "should have metric parts from population norms");
  assert.equal(usingPopulationNorms, true);
});

check("population norms give a reasonable score for average performance", () => {
  const { impairment } = M.compareToBaseline(session(2.3, 950), []);
  assert.ok(Math.abs(impairment) < 1.5, `average performance should be near zero, got ${impairment}`);
});

// ── Data quality and verdict ─────────────────────────────────────────────────
const goodSignals = { facePresentPct: 0.95, lookingPct: 0.9, frameJitterMs: 5 };
const goodTouch = { scored: 20, falseStarts: 1, attempted: 21 };
const goodGesture = { scored: 30, rejectRate: 0.05 };

check("clean data passes the quality gate", () => {
  const q = M.dataQuality({ touch: goodTouch, gesture: goodGesture, signals: goodSignals });
  assert.equal(q.ok, true, q.reasons.join("; "));
});

check("lost hand tracking fails the quality gate", () => {
  const q = M.dataQuality({
    touch: goodTouch,
    gesture: { scored: 30, rejectRate: 0.5 },
    signals: goodSignals,
  });
  assert.equal(q.ok, false);
  assert.match(q.reasons[0], /hand tracking/);
});

check("an absent face fails the quality gate", () => {
  const q = M.dataQuality({
    touch: goodTouch, gesture: goodGesture,
    signals: { ...goodSignals, facePresentPct: 0.3 },
  });
  assert.equal(q.ok, false);
});

check("bad data forces a retest even when performance looks fine", () => {
  const d = M.decideVerdict({
    impairment: 0,
    quality: { ok: false, reasons: ["camera timing was unstable"] },
    baselineCount: M.BASELINE_TARGET,
  });
  assert.equal(d.verdict, M.VERDICT.RETEST);
  assert.match(d.reason, /camera timing/);
});

check("an incomplete baseline gives a provisional verdict, not a retest", () => {
  const d = M.decideVerdict({
    impairment: 5,
    quality: { ok: true, reasons: [] },
    baselineCount: M.BASELINE_TARGET - 1,
  });
  assert.equal(d.verdict, M.VERDICT.FATIGUED);
  assert.equal(d.provisional, true);
});

check("the three verdict bands behave", () => {
  const q = { ok: true, reasons: [] };
  const n = M.BASELINE_TARGET;
  const noDrow = { score: 0, details: [] };
  assert.equal(M.decideVerdict({ impairment: 0.2, quality: q, baselineCount: n, drowsiness: noDrow }).verdict, M.VERDICT.NOT_FATIGUED);
  assert.equal(M.decideVerdict({ impairment: 1.5, quality: q, baselineCount: n, drowsiness: noDrow }).verdict, M.VERDICT.RETEST);
  assert.equal(M.decideVerdict({ impairment: 2.4, quality: q, baselineCount: n, drowsiness: noDrow }).verdict, M.VERDICT.FATIGUED);
  assert.equal(M.decideVerdict({ impairment: M.Z_CLEAR, quality: q, baselineCount: n, drowsiness: noDrow }).verdict, M.VERDICT.NOT_FATIGUED);
  assert.equal(M.decideVerdict({ impairment: M.Z_FATIGUED, quality: q, baselineCount: n, drowsiness: noDrow }).verdict, M.VERDICT.FATIGUED);
});

check("drowsiness signals push a borderline verdict toward fatigued", () => {
  const q = { ok: true, reasons: [] };
  const n = M.BASELINE_TARGET;
  const drowsy = { score: 2.5, details: ["elevated eye closure"] };
  // Impairment 1.5 is borderline (RETEST), but drowsiness adds 2.5*0.3 = 0.75 → 2.25 → FATIGUED
  assert.equal(M.decideVerdict({ impairment: 1.5, quality: q, baselineCount: n, drowsiness: drowsy }).verdict, M.VERDICT.FATIGUED);
});

check("drowsiness scoring detects elevated eye closure", () => {
  const earSamples = Array.from({ length: 100 }, (_, i) => i < 20 ? 0.15 : 0.30);
  const d = M.drowsinessScore({ earSamples, marSamples: [] });
  assert.ok(d.perclos > 0.1, `perclos was ${d.perclos}`);
  assert.ok(d.score > 0, "elevated PERCLOS should produce a positive score");
});

check("drowsiness scoring counts yawns from MAR spikes", () => {
  const marSamples = [0.3, 0.3, 0.7, 0.7, 0.3, 0.3, 0.8, 0.3];
  const d = M.drowsinessScore({ earSamples: [0.3, 0.3], marSamples });
  assert.equal(d.yawns, 2);
});

check("self-report mismatches are flagged both ways", () => {
  assert.match(M.selfReportMismatch({ rating: 1 }, 3.0), /alert but performed/);
  assert.match(M.selfReportMismatch({ rating: 7 }, 0.1), /exhausted but performed/);
  assert.equal(M.selfReportMismatch({ rating: 4 }, 0.5), null);
  assert.equal(M.selfReportMismatch({ rating: null }, 3.0), null);
  assert.equal(M.selfReportMismatch({ rating: 1 }, null), null);
});

check("frame jitter is reported separately from mean frame time", () => {
  const s = M.qualityFromSignals({ frames: 100, facePresent: 90, looking: 80, frameIntervals: [16, 17, 16, 51] });
  assert.equal(s.facePresentPct, 0.9);
  assert.ok(s.frameJitterMs > 10, "a dropped frame should show up as jitter");
});

console.log(process.exitCode ? `\n${passed} passed, with failures above.` : `All ${passed} checks passed.`);
