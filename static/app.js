/**
 * AeroVigil Pre-Flight Check — flow controller.
 *
 * Eight screens, two timed blocks, one verdict against the pilot's own baseline.
 * The verdict comes from measured performance; the camera supplies supporting
 * signals and data quality only. See reports/mentor for the rationale.
 */

import { Vision, REST_SIGN } from "./vision.js";
import {
  blockStats, qualityFromSignals, dataQuality, compareToBaseline, decideVerdict,
  selfReportMismatch, TOUCH_LAPSE_MS, FALSE_START_MS, GESTURE_TIMEOUT_MS,
  BASELINE_TARGET, FAMILIARISATION, VERDICT,
} from "./metrics.js";

// ── Block parameters ─────────────────────────────────────────────────────────
const TOUCH_MS = 60_000;          // touch block length
const TOUCH_ISI = [1000, 4000];   // PVT-B style inter-stimulus interval
const TOUCH_TIMEOUT_MS = 3000;
const GESTURE_TRIALS = 20;
const GESTURE_ISI = [900, 2500];  // measured from the moment the fist is settled
const PRACTICE_TRIALS = 5;
const STABLE_FRAMES = 3;          // frames a sign must hold before it counts

const SHAPES = [
  { n: 1, name: "Triangle", colour: "#EA580C" },
  { n: 2, name: "Circle",   colour: "#0284C7" },
  { n: 3, name: "Square",   colour: "#059669" },
  { n: 4, name: "Diamond",  colour: "#CA8A04" },
  { n: 5, name: "Star",     colour: "#7C3AED" },
];

const STEPS = ["welcome", "checkin", "setup", "practice", "touch", "gesture", "analysing", "result"];
const STEP_LABELS = {
  welcome: "Welcome", checkin: "Check-in", setup: "Setup", practice: "Practice",
  touch: "Touch test", gesture: "Hand-sign test", analysing: "Analysing", result: "Result",
};

const STORE_KEY = "aerovigil.v1";
const $ = (sel) => document.querySelector(sel);
const rand = (min, max) => min + Math.random() * (max - min);

// ── Stored sessions (this device only) ───────────────────────────────────────

function loadStore() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    const parsed = raw ? JSON.parse(raw) : null;
    if (parsed && Array.isArray(parsed.sessions)) return parsed;
  } catch {
    // Corrupt or unavailable storage must not stop the check: start fresh.
  }
  return { sessions: [] };
}

function saveStore(store) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(store));
  } catch {
    // Private browsing or a full quota — the session still completes, it just
    // does not contribute to the baseline.
  }
}

const baselineSessions = (store) => store.sessions.filter((s) => s.baseline);

// ── Drawing helpers ──────────────────────────────────────────────────────────

function shapeSvg(shape, size = 160) {
  const c = shape.colour;
  const body = {
    1: `<polygon points="50,10 92,88 8,88" fill="${c}"/>`,
    2: `<circle cx="50" cy="50" r="42" fill="${c}"/>`,
    3: `<rect x="10" y="10" width="80" height="80" rx="6" fill="${c}"/>`,
    4: `<polygon points="50,6 94,50 50,94 6,50" fill="${c}"/>`,
    5: `<polygon points="50,5 61,38 96,38 68,59 79,92 50,71 21,92 32,59 4,38 39,38" fill="${c}"/>`,
  }[shape.n];
  return `<svg viewBox="0 0 100 100" width="${size}" height="${size}" role="img" aria-label="${shape.name}">${body}</svg>`;
}

/** Simple hand glyph: raised fingers filled, folded fingers stubbed. */
function handSvg(n, size = 54) {
  const up = (i) => (n === 5 ? true : i < n);
  let f = "";
  for (let i = 0; i < 4; i++) {
    const x = 20 + i * 15;
    const h = up(i) ? 34 : 10;
    const y = 46 - h;
    f += `<rect x="${x}" y="${y}" width="10" height="${h + 8}" rx="5" fill="${up(i) ? "#0F172A" : "#CBD5E1"}"/>`;
  }
  const thumb = n === 5
    ? `<rect x="4" y="40" width="10" height="24" rx="5" transform="rotate(-35 9 52)" fill="#0F172A"/>`
    : `<rect x="8" y="48" width="12" height="9" rx="4.5" fill="#CBD5E1"/>`;
  return `<svg viewBox="0 0 90 80" width="${size}" height="${size * 0.9}" role="img" aria-label="${n} fingers">
    ${thumb}${f}<rect x="16" y="44" width="62" height="30" rx="12" fill="#0F172A"/></svg>`;
}

function legendHtml() {
  return SHAPES.map((s) => `
    <div class="legend-item">
      <span class="legend-shape">${shapeSvg(s, 34)}</span>
      <span class="legend-num mono">${s.n}</span>
      <span class="legend-hand">${handSvg(s.n, 40)}</span>
      <span class="legend-name">${s.name}</span>
    </div>`).join("");
}

// ── Application ──────────────────────────────────────────────────────────────

class App {
  constructor() {
    this.store = loadStore();
    this.vision = new Vision();
    this.video = $("#webcam");
    this.overlay = $("#overlay");
    this.octx = this.overlay.getContext("2d");
    this.screen = "welcome";
    this.frameHandler = null;   // set by whichever block is running
    this.lastFrameT = null;
    this.signals = { frames: 0, facePresent: 0, looking: 0, frameIntervals: [], earSamples: [], occluded: 0 };
    this.session = null;
    this.lastResult = null;
  }

  // ── Boot ───────────────────────────────────────────────────────────────────
  async boot() {
    this.wireUi();
    this.renderSteps();
    this.go("welcome");
    setInterval(() => {
      $("#utc").textContent = new Date().toISOString().slice(11, 19) + "Z";
    }, 1000);

    try {
      $("#boot-msg").textContent = "Requesting camera access…";
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
        audio: false,
      });
      this.video.srcObject = stream;
      await new Promise((res) => { this.video.onloadedmetadata = res; });
      await this.video.play();

      await this.vision.load((m) => { $("#boot-msg").textContent = m; });
      $("#boot").classList.add("hidden");
      requestAnimationFrame(() => this.loop());
    } catch (err) {
      $("#boot-msg").innerHTML =
        `<span class="error">Could not start: ${err.message}</span><br><br>
         Allow camera access and use a Chromium-based browser (Chrome or Edge).`;
    }
  }

  // ── Per-frame loop ─────────────────────────────────────────────────────────
  loop() {
    requestAnimationFrame(() => this.loop());
    if (!this.vision.ready || this.video.readyState < 2) return;

    const now = performance.now();
    const w = this.video.videoWidth || 1280;
    const h = this.video.videoHeight || 720;
    const f = this.vision.detect(this.video, now, w, h);
    if (!f) return;

    if (this.lastFrameT != null) this.signals.frameIntervals.push(now - this.lastFrameT);
    this.lastFrameT = now;

    this.drawOverlay(f);
    this.updateCamBadge(f);
    if (this.frameHandler) this.frameHandler(f);
  }

  drawOverlay(f) {
    const cv = this.overlay;
    const r = cv.getBoundingClientRect();
    if (cv.width !== Math.round(r.width) || cv.height !== Math.round(r.height)) {
      cv.width = Math.round(r.width);
      cv.height = Math.round(r.height);
    }
    const ctx = this.octx;
    ctx.clearRect(0, 0, cv.width, cv.height);
    if (!f.handLm) return;

    // The preview is mirrored so the pilot sees themselves naturally.
    const px = (p) => (1 - p.x) * cv.width;
    const py = (p) => p.y * cv.height;
    const ok = f.sign != null;
    ctx.strokeStyle = ok ? "rgba(14,165,233,.9)" : "rgba(217,119,6,.9)";
    ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 2;
    const bones = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
                   [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
    for (const [a, b] of bones) {
      ctx.beginPath();
      ctx.moveTo(px(f.handLm[a]), py(f.handLm[a]));
      ctx.lineTo(px(f.handLm[b]), py(f.handLm[b]));
      ctx.stroke();
    }
    for (const p of f.handLm) {
      ctx.beginPath();
      ctx.arc(px(p), py(p), 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  updateCamBadge(f) {
    const dot = $("#cam-dot");
    const label = $("#cam-label");
    let state = "ok", text = "Tracking";
    if (!f.face) { state = "warn"; text = "No face"; }
    else if (f.occluded) { state = "warn"; text = "Hand covers face"; }
    else if (!f.handLm) { state = "idle"; text = "No hand"; }
    else if (f.sign == null) { state = "warn"; text = "Unclear sign"; }
    else if (f.sign === REST_SIGN) { state = "ok"; text = "Rest"; }
    else { state = "ok"; text = `Sign ${f.sign}`; }
    dot.className = `dot ${state}`;
    label.textContent = text;
  }

  /** Accumulate supporting signals while a scored block is running. */
  recordSignals(f) {
    this.signals.frames++;
    if (f.face) {
      this.signals.facePresent++;
      if (f.face.looking) this.signals.looking++;
      if (f.face.ear != null) this.signals.earSamples.push(f.face.ear);
    }
    if (f.occluded) this.signals.occluded++;
  }

  // ── Screens ────────────────────────────────────────────────────────────────
  renderSteps() {
    $("#steps").innerHTML = STEPS.filter((s) => s !== "analysing")
      .map((s) => `<span class="step" data-step="${s}">${STEP_LABELS[s]}</span>`).join("");
  }

  go(name) {
    this.frameHandler = null;
    this.screen = name;
    document.querySelectorAll(".screen").forEach((el) => {
      el.classList.toggle("active", el.dataset.screen === name);
    });
    const idx = STEPS.indexOf(name);
    document.querySelectorAll(".step").forEach((el) => {
      const i = STEPS.indexOf(el.dataset.step);
      el.classList.toggle("done", i < idx);
      el.classList.toggle("now", i === idx);
    });
    // Move the single camera element into this screen's slot, if it has one.
    const slot = document.querySelector(`.screen[data-screen="${name}"] .cam-slot`);
    const dock = $("#camera-dock");
    if (slot) slot.appendChild(dock); else document.body.appendChild(dock);
    dock.classList.toggle("offscreen", !slot);

    if (name === "welcome") this.renderWelcome();
    if (name === "setup") this.runSetup();
    if (name === "practice") this.runPractice();
    if (name === "touch") this.runTouch();
    if (name === "gesture") this.runGesture();
  }

  renderWelcome() {
    const done = this.store.sessions.length;
    const base = baselineSessions(this.store).length;
    const el = $("#baseline-status");
    if (done < FAMILIARISATION) {
      el.className = "status-strip info";
      el.innerHTML = `<strong>Familiarisation.</strong> Your first ${FAMILIARISATION} sessions are practice and are
        not used as a baseline, because people get faster at any new task for the first few goes.`;
    } else if (base < BASELINE_TARGET) {
      el.className = "status-strip info";
      el.innerHTML = `<strong>Building your baseline: ${base} of ${BASELINE_TARGET} rested sessions.</strong>
        Until it is complete the app cannot judge a session, and will say <em>Retest needed</em>.`;
    } else {
      el.className = "status-strip ok";
      el.innerHTML = `<strong>Baseline ready</strong> — built from ${base} rested sessions.`;
    }

    const recent = this.store.sessions.slice(-5).reverse();
    $("#history-strip").innerHTML = recent.length
      ? `<h3>Recent checks</h3>` + recent.map((s) => {
          const v = s.verdict ?? "—";
          const cls = v === VERDICT.NOT_FATIGUED ? "ok" : v === VERDICT.FATIGUED ? "bad" : "warn";
          return `<div class="hist-row">
            <span class="mono">${new Date(s.at).toLocaleString()}</span>
            <span class="chip ${cls}">${verdictLabel(v)}</span>
            <span class="mono muted">${s.touch?.medianMs ? Math.round(s.touch.medianMs) + " ms" : "—"}</span>
          </div>`;
        }).join("")
      : "";
  }

  // ── Check-in ───────────────────────────────────────────────────────────────
  wireUi() {
    $("#btn-start").onclick = () => {
      this.session = { at: Date.now(), checkin: null };
      this.signals = { frames: 0, facePresent: 0, looking: 0, frameIntervals: [], earSamples: [], occluded: 0 };
      // The rested declaration is only asked for while the baseline is still
      // being built; afterwards it would just be another gameable input.
      const enrolling = baselineSessions(this.store).length < BASELINE_TARGET
        && this.store.sessions.length >= FAMILIARISATION;
      $("#rested-wrap").classList.toggle("hidden", !enrolling);
      $("#chk-rested").checked = false;
      this.go("checkin");
    };

    for (const [id, out, unit] of [["sl-24", "out-24", "h"], ["sl-72", "out-72", "h"], ["sl-awake", "out-awake", "h"]]) {
      const input = $("#" + id);
      const render = () => { $("#" + out).textContent = `${input.value} ${unit}`; };
      input.oninput = render;
      render();
    }

    const labels = ["Fully alert", "Alert", "Mostly alert", "Neither", "A bit tired", "Very tired", "Exhausted"];
    $("#rating-row").innerHTML = labels.map((l, i) =>
      `<button class="rate" role="radio" aria-checked="false" data-rate="${i + 1}">
         <span class="rate-n mono">${i + 1}</span><span class="rate-l">${l}</span></button>`).join("");
    $("#rating-row").onclick = (e) => {
      const b = e.target.closest(".rate");
      if (!b) return;
      document.querySelectorAll(".rate").forEach((x) => {
        const on = x === b;
        x.classList.toggle("on", on);
        x.setAttribute("aria-checked", String(on));
      });
    };

    $("#btn-checkin-next").onclick = () => {
      const sel = document.querySelector(".rate.on");
      this.session.checkin = {
        sleep24: Number($("#sl-24").value),
        sleep72: Number($("#sl-72").value),
        hoursAwake: Number($("#sl-awake").value),
        rating: sel ? Number(sel.dataset.rate) : null,
        restedClaim: $("#chk-rested").checked,
      };
      this.go("setup");
    };

    document.querySelectorAll("[data-back]").forEach((b) => {
      b.onclick = () => this.go(STEPS[Math.max(0, STEPS.indexOf(this.screen) - 1)]);
    });

    $("#btn-setup-next").onclick = () => this.go("practice");
    $("#btn-practice-next").onclick = () => this.go("touch");
    $("#btn-retest").onclick = () => this.go("welcome");
    $("#btn-export").onclick = () => this.exportJson();

    $("#practice-legend").innerHTML = legendHtml();
    $("#gesture-legend").innerHTML = legendHtml();

    $("#btn-demo").onclick = () => $("#demo-panel").classList.toggle("hidden");
    $("#demo-panel").onclick = (e) => {
      const act = e.target.closest("[data-demo]")?.dataset.demo;
      if (!act) return;
      if (act === "close") $("#demo-panel").classList.add("hidden");
      if (act === "seed") { this.seedBaseline(); this.renderWelcome(); }
      if (act === "clear") { this.store = { sessions: [] }; saveStore(this.store); this.renderWelcome(); }
      if (act.startsWith("v-")) {
        const v = act === "v-not" ? VERDICT.NOT_FATIGUED : act === "v-fat" ? VERDICT.FATIGUED : VERDICT.RETEST;
        this.previewVerdict(v);
      }
    };
  }

  // ── Setup ──────────────────────────────────────────────────────────────────
  runSetup() {
    const checks = { face: false, light: false, hand: false, clear: false };
    let stable = 0;
    this.frameHandler = (f) => {
      checks.face = !!f.face;
      // A usable eye reading is the practical light test: if EAR can be
      // computed at all, the eyes are resolved well enough.
      checks.light = !!(f.face && f.face.ear != null);
      checks.hand = !!f.handLm && f.handScale > 0.04;
      checks.clear = !f.occluded;
      for (const [k, v] of Object.entries(checks)) {
        document.querySelector(`#setup-checks [data-check="${k}"]`)?.classList.toggle("ok", v);
      }
      stable = Object.values(checks).every(Boolean) ? stable + 1 : 0;
      $("#btn-setup-next").disabled = stable < 15;
    };
  }

  // ── Trial engine shared by practice and the gesture block ──────────────────
  /**
   * Runs one colour/hand-sign trial and resolves with its record.
   * Phases: wait for a settled fist → random delay → stimulus → answer.
   */
  gestureTrial(shape, { onPhase, record = false }) {
    return new Promise((resolve) => {
      let phase = "rest";
      let restFrames = 0, stimulusAt = 0, onsetAt = null, isiUntil = 0;
      let lastSign = null, sameFor = 0, seen = 0, handFrames = 0;

      onPhase("rest");
      this.frameHandler = (f) => {
        const now = f.t;
        if (record) this.recordSignals(f);
        if (phase !== "rest") { seen++; if (f.handLm) handFrames++; }

        if (phase === "rest") {
          restFrames = f.sign === REST_SIGN ? restFrames + 1 : 0;
          if (restFrames >= STABLE_FRAMES) {
            phase = "isi";
            isiUntil = now + rand(...GESTURE_ISI);
            onPhase("ready");
          }
          return;
        }

        if (phase === "isi") {
          if (now >= isiUntil) {
            phase = "stimulus";
            stimulusAt = now;
            onPhase("stimulus");
          }
          return;
        }

        if (phase !== "stimulus") return;

        // Movement onset: the first frame the hand leaves the fist. Closer to
        // decision time than the confirmed sign, which also includes the time
        // taken to form the shape.
        if (onsetAt == null && f.sign != null && f.sign !== REST_SIGN) onsetAt = now;

        if (f.sign != null && f.sign !== REST_SIGN) {
          sameFor = f.sign === lastSign ? sameFor + 1 : 1;
          lastSign = f.sign;
          if (sameFor >= STABLE_FRAMES) {
            phase = "done";
            this.frameHandler = null;
            resolve({
              shape: shape.n, answer: f.sign, correct: f.sign === shape.n,
              rt: now - stimulusAt,
              onsetRt: onsetAt != null ? onsetAt - stimulusAt : null,
              rejected: false, falseStart: false,
            });
          }
        } else {
          sameFor = 0;
          lastSign = null;
        }

        if (now - stimulusAt > GESTURE_TIMEOUT_MS) {
          phase = "done";
          this.frameHandler = null;
          // Lost tracking is a rejected trial, not a pilot error: scoring a
          // flaky tracker as a wrong answer would make the pilot look impaired.
          const lostTracking = seen > 0 && handFrames / seen < 0.4;
          resolve({
            shape: shape.n, answer: null, correct: false, rt: null, onsetRt: null,
            rejected: lostTracking, falseStart: false,
          });
        }
      };
    });
  }

  async runPractice() {
    $("#btn-practice-next").disabled = true;
    const stim = $("#practice-stimulus");
    const prompt = $("#practice-prompt");
    const fb = $("#practice-feedback");

    for (let i = 0; i < PRACTICE_TRIALS; i++) {
      const shape = SHAPES[Math.floor(Math.random() * SHAPES.length)];
      const t = await this.gestureTrial(shape, {
        onPhase: (p) => {
          if (p === "rest") { stim.innerHTML = ""; prompt.textContent = "Make a fist to begin."; }
          if (p === "ready") { stim.innerHTML = ""; prompt.textContent = "Wait…"; }
          if (p === "stimulus") { stim.innerHTML = shapeSvg(shape); prompt.textContent = "Show the number."; }
        },
      });
      if (this.screen !== "practice") return;
      fb.textContent = t.rejected ? "Hand not seen — keep it in view."
        : t.rt == null ? "No answer in time."
        : t.correct ? `Good — ${Math.round(t.rt)} ms`
        : `That was ${t.answer}, the shape needed ${t.shape}.`;
      fb.className = `feedback ${t.correct ? "good" : "bad"}`;
    }
    stim.innerHTML = "";
    prompt.textContent = "Practice complete.";
    $("#btn-practice-next").disabled = false;
  }

  // ── Touch block ────────────────────────────────────────────────────────────
  runTouch() {
    const trials = [];
    const area = $("#touch-area");
    const dot = $("#touch-dot");
    const msg = $("#touch-msg");
    const start = performance.now();
    let shownAt = null, nextAt = start + rand(...TOUCH_ISI), finished = false;

    const show = () => {
      dot.classList.remove("hidden");
      msg.textContent = "";
      // Stamp the stimulus on the frame that paints it, not the frame that
      // asked for it: this block is the precise anchor the hand-sign block is
      // measured against, so a frame of slack here matters.
      requestAnimationFrame(() => { shownAt = performance.now(); });
    };
    const hide = () => { shownAt = null; dot.classList.add("hidden"); };

    const onTap = (e) => {
      e.preventDefault();
      if (finished) return;
      // event.timeStamp shares the performance.now() timebase in modern
      // browsers, so it is closer to the real touch than the handler's own clock.
      const now = Number.isFinite(e.timeStamp) && e.timeStamp > 0 ? e.timeStamp : performance.now();
      if (shownAt == null) {
        trials.push({ rt: null, falseStart: true, rejected: false, correct: true });
        msg.textContent = "Too early — wait for the circle.";
        nextAt = performance.now() + rand(...TOUCH_ISI);
        return;
      }
      const rt = now - shownAt;
      trials.push({ rt, falseStart: false, rejected: false, correct: true });
      $("#touch-last").textContent = `${Math.round(rt)} ms`;
      hide();
      msg.textContent = "";
      nextAt = performance.now() + rand(...TOUCH_ISI);
    };
    area.addEventListener("pointerdown", onTap);

    // Supporting camera signals ride along with the vision loop, but the block
    // itself is driven by its own animation frame so that slow vision
    // processing cannot stretch the intervals or delay the stimulus.
    this.frameHandler = (f) => this.recordSignals(f);

    const tick = () => {
      if (finished || this.screen !== "touch") return;
      requestAnimationFrame(tick);
      const now = performance.now();
      const elapsed = now - start;
      $("#touch-progress").style.width = `${Math.min(100, (elapsed / TOUCH_MS) * 100)}%`;
      const left = Math.max(0, Math.ceil((TOUCH_MS - elapsed) / 1000));
      $("#touch-timer").textContent = `0:${String(left).padStart(2, "0")}`;

      if (shownAt == null && now >= nextAt) show();
      if (shownAt != null && now - shownAt > TOUCH_TIMEOUT_MS) {
        trials.push({ rt: null, falseStart: false, rejected: false, correct: true });
        hide();
        nextAt = now + rand(...TOUCH_ISI);
      }

      if (elapsed >= TOUCH_MS) {
        finished = true;
        area.removeEventListener("pointerdown", onTap);
        hide();
        this.frameHandler = null;
        this.session.touchTrials = trials;
        this.go("gesture");
      }
    };
    requestAnimationFrame(tick);
  }

  // ── Gesture block ──────────────────────────────────────────────────────────
  async runGesture() {
    const trials = [];
    const stim = $("#gesture-stimulus");
    const prompt = $("#gesture-prompt");
    // A fixed order of five shapes repeated in shuffled batches keeps each
    // answer equally frequent, so the pilot cannot gain by guessing.
    const bag = [];
    for (let i = 0; i < GESTURE_TRIALS; i++) {
      if (!bag.length) bag.push(...SHAPES.slice().sort(() => Math.random() - 0.5));
      const shape = bag.pop();
      $("#gesture-trial").textContent = `Trial ${i + 1} / ${GESTURE_TRIALS}`;
      $("#gesture-progress").style.width = `${(i / GESTURE_TRIALS) * 100}%`;

      const t = await this.gestureTrial(shape, {
        record: true,
        onPhase: (p) => {
          if (p === "rest") { stim.innerHTML = ""; prompt.textContent = "Return to a fist."; }
          if (p === "ready") { stim.innerHTML = ""; prompt.textContent = "Wait…"; }
          if (p === "stimulus") { stim.innerHTML = shapeSvg(shape); prompt.textContent = ""; }
        },
      });
      if (this.screen !== "gesture") return;
      trials.push(t);
    }
    stim.innerHTML = "";
    this.session.gestureTrials = trials;
    this.finish();
  }

  // ── Scoring and result ─────────────────────────────────────────────────────
  finish() {
    this.go("analysing");
    setTimeout(() => this.score(), 1100);
  }

  score() {
    const touch = blockStats(this.session.touchTrials ?? [], { lapseMs: TOUCH_LAPSE_MS });
    const gesture = blockStats(this.session.gestureTrials ?? []);
    const signals = qualityFromSignals(this.signals);
    const quality = dataQuality({ touch, gesture, signals });

    const base = baselineSessions(this.store);
    const { impairment, parts } = compareToBaseline({ touch, gesture }, base);
    const decision = decideVerdict({ impairment, quality, baselineCount: base.length });
    const mismatch = selfReportMismatch(this.session.checkin, impairment);

    const isFamiliarisation = this.store.sessions.length < FAMILIARISATION;
    const record = {
      at: this.session.at,
      checkin: this.session.checkin,
      touch, gesture, signals,
      impairment,
      verdict: decision.verdict,
      reason: decision.reason,
      mismatch,
      familiarisation: isFamiliarisation,
      // A session only joins the baseline if the pilot declared themselves
      // rested AND the data was clean AND it is past familiarisation.
      baseline: !isFamiliarisation && quality.ok && !!this.session.checkin?.restedClaim,
    };
    this.store.sessions.push(record);
    saveStore(this.store);

    this.lastResult = { record, parts, quality, decision, baselineCount: base.length };
    this.renderResult(this.lastResult);
    this.go("result");
  }

  renderResult({ record, parts, quality, decision, baselineCount }) {
    const v = decision.verdict;
    const cls = v === VERDICT.NOT_FATIGUED ? "ok" : v === VERDICT.FATIGUED ? "bad" : "warn";
    $("#verdict-card").className = `verdict ${cls}`;
    $("#verdict-title").textContent = verdictLabel(v);
    $("#verdict-line").textContent = capitalise(decision.reason ?? "");

    const actions = {
      [VERDICT.NOT_FATIGUED]: [],
      [VERDICT.FATIGUED]: ["Rest before you fly, and retest after a break.", "Tell your supervisor.",
        "This is an indicator, not a decision — your own assessment still applies."],
      [VERDICT.RETEST]: decision.baselineBuilding
        ? ["Keep doing checks when you are rested until the baseline is complete."]
        : ["Fix the problem above and run the check again."],
    }[v] ?? [];
    $("#verdict-actions").innerHTML = actions.length
      ? `<ul>${actions.map((a) => `<li>${a}</li>`).join("")}</ul>` : "";

    // Today vs baseline, one bar per metric, as a z-score away from the pilot's own mean.
    $("#bars").innerHTML = parts.length
      ? parts.map((p) => {
          // Centre is the pilot's own average; the bar runs right when slower
          // than usual, left when faster. Clamped at 3 standard deviations.
          const z = Math.max(-3, Math.min(3, p.z));
          const half = (Math.abs(z) / 3) * 50;
          const left = z >= 0 ? 50 : 50 - half;
          const tone = p.z >= 2 ? "bad" : p.z >= 1 ? "warn" : "ok";
          return `<div class="bar-row">
            <span class="bar-label">${p.label}</span>
            <span class="bar-track"><span class="bar-mid"></span>
              <span class="bar-fill ${tone}" style="left:${left}%;width:${half}%"></span></span>
            <span class="bar-z mono">${fmtZ(p.z)}</span>
          </div>`;
        }).join("") + `<p class="fine">Bars show how far today sits from your own rested average, in units of
            your own session-to-session spread. 0 is typical for you; right is slower or less accurate.</p>`
      : `<p class="muted">No baseline comparison yet (${baselineCount} of ${BASELINE_TARGET} rested sessions).</p>`;

    const t = record.touch, g = record.gesture;
    $("#result-facts").innerHTML = facts([
      ["Touch median", t.medianMs ? `${Math.round(t.medianMs)} ms` : "—"],
      ["Touch lapses", `${t.lapses ?? "—"} of ${t.scored}`],
      ["Early taps", String(t.falseStarts)],
      ["Hand-sign median", g.medianMs ? `${Math.round(g.medianMs)} ms` : "—"],
      ["Hand-sign accuracy", g.accuracy != null ? `${Math.round(g.accuracy * 100)}%` : "—"],
      ["Trials rejected", `${g.rejected} of ${g.trials}`],
      ["Data quality", quality.ok ? "Usable" : "Not usable"],
    ]);

    const c = record.checkin ?? {};
    $("#result-extra").innerHTML = facts([
      ["Face present", pct(record.signals.facePresentPct)],
      ["Looking at screen", pct(record.signals.lookingPct)],
      ["Mean frame interval", record.signals.meanFrameMs ? `${record.signals.meanFrameMs.toFixed(1)} ms` : "—"],
      ["Frame jitter", record.signals.frameJitterMs ? `${record.signals.frameJitterMs.toFixed(1)} ms` : "—"],
      ["Reported sleep (24 h)", c.sleep24 != null ? `${c.sleep24} h` : "—"],
      ["Reported hours awake", c.hoursAwake != null ? `${c.hoursAwake} h` : "—"],
      ["Self-rating", c.rating != null ? `${c.rating} of 7` : "—"],
      ["Self-report mismatch", record.mismatch ?? "none"],
      ["Counted toward baseline", record.baseline ? "yes" : "no"],
    ]);
  }

  previewVerdict(v) {
    $("#demo-panel").classList.add("hidden");
    const demoTrials = (n, ms) => Array.from({ length: n }, () => ({
      rt: ms + Math.random() * 60, falseStart: false, rejected: false, correct: true,
    }));
    const slow = v === VERDICT.FATIGUED;
    const touch = blockStats(demoTrials(22, slow ? 430 : 270), { lapseMs: TOUCH_LAPSE_MS });
    const gesture = blockStats(demoTrials(30, slow ? 1150 : 760));
    const signals = qualityFromSignals({ frames: 1800, facePresent: 1760, looking: 1690, frameIntervals: [16, 17, 16, 17] });
    const base = baselineSessions(this.store);
    const { impairment, parts } = compareToBaseline({ touch, gesture }, base);
    const quality = v === VERDICT.RETEST
      ? { ok: false, reasons: ["hand tracking was lost too often"] }
      : { ok: true, reasons: [] };
    const decision = v === VERDICT.RETEST
      ? { verdict: VERDICT.RETEST, reason: quality.reasons[0], reasons: quality.reasons }
      : { verdict: v, reason: v === VERDICT.FATIGUED ? "performance clearly below your normal range"
                                                     : "performance within your normal range" };
    this.lastResult = {
      record: { at: Date.now(), checkin: { sleep24: slow ? 4 : 7.5, hoursAwake: slow ? 17 : 2, rating: slow ? 6 : 2 },
                touch, gesture, signals, impairment, verdict: decision.verdict, mismatch: null,
                baseline: false, demo: true },
      parts, quality, decision, baselineCount: base.length,
    };
    this.renderResult(this.lastResult);
    this.go("result");
  }

  seedBaseline() {
    // Synthetic sessions so the result screen can be demonstrated before any
    // real baseline exists. Marked demo:true so exports never pass as real data.
    const mk = (daysAgo) => {
      // Each session gets its own offset as well as per-trial noise: real people
      // vary between sessions, and a baseline without that variation would make
      // every later session look impaired.
      const dayT = (Math.random() - 0.5) * 40;
      const dayG = (Math.random() - 0.5) * 120;
      const t = Array.from({ length: 24 }, () => ({ rt: 255 + dayT + Math.random() * 80, falseStart: false, rejected: false, correct: true }));
      const g = Array.from({ length: 32 }, () => ({ rt: 720 + dayG + Math.random() * 200, falseStart: false, rejected: false, correct: Math.random() > 0.05 }));
      return {
        at: Date.now() - daysAgo * 86_400_000,
        demo: true, baseline: true, familiarisation: false, verdict: VERDICT.NOT_FATIGUED,
        touch: blockStats(t, { lapseMs: TOUCH_LAPSE_MS }), gesture: blockStats(g),
        signals: qualityFromSignals({ frames: 1800, facePresent: 1780, looking: 1700, frameIntervals: [16, 17] }),
        checkin: { restedClaim: true, rating: 2 },
      };
    };
    for (let i = 0; i < BASELINE_TARGET; i++) this.store.sessions.push(mk(BASELINE_TARGET - i));
    saveStore(this.store);
  }

  exportJson() {
    if (!this.lastResult) return;
    const { record, decision } = this.lastResult;
    const payload = {
      system: "AeroVigil Pre-Flight Check",
      version: "3.0-prototype",
      note: "Advisory alertness/performance indicator. Thresholds provisional and not validated.",
      exportedAt: new Date().toISOString(),
      demo: !!record.demo,
      verdict: decision.verdict,
      reason: decision.reason,
      impairment: record.impairment,
      touch: record.touch,
      gesture: record.gesture,
      signals: record.signals,
      checkin: record.checkin,
      selfReportMismatch: record.mismatch,
      countedTowardBaseline: record.baseline,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = Object.assign(document.createElement("a"), {
      href: URL.createObjectURL(blob),
      download: `aerovigil_check_${new Date(record.at).toISOString().slice(0, 19).replace(/[:T]/g, "")}.json`,
    });
    a.click();
    URL.revokeObjectURL(a.href);
  }
}

// ── Small formatting helpers ─────────────────────────────────────────────────
const verdictLabel = (v) => ({
  [VERDICT.NOT_FATIGUED]: "NOT FATIGUED",
  [VERDICT.FATIGUED]: "FATIGUED",
  [VERDICT.RETEST]: "RETEST NEEDED",
}[v] ?? "—");

const capitalise = (s) => (s ? s[0].toUpperCase() + s.slice(1) : "");
// Beyond 3 the exact figure is meaningless — the baseline spread is estimated
// from a handful of sessions — so show it as off the scale rather than "+22".
const fmtZ = (z) => (Math.abs(z) > 3 ? (z > 0 ? ">+3" : "<-3") : `${z > 0 ? "+" : ""}${z.toFixed(1)}`);
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const facts = (rows) => rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

const app = new App();
// Exposed on purpose: the tablet has no convenient devtools, and the validation
// study needs a way to inspect and export a session from the console.
window.aerovigil = app;
app.boot();
