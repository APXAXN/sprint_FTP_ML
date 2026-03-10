import { useState } from "react";
import { ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Area } from "recharts";

const VO2_GREEN = "#A8D5A2";

const ENERGY_DATA = Array.from({ length: 41 }, (_, i) => {
  const s = i * 3;
  const aerobic = Math.min(96, 20 + 50 * (1 - Math.exp(-s / 55)));
  // Dual-phase VO₂ kinetics (Poole & Jones 2012), normalized so asymptote = 100%
  const vo2Raw = s <= 15
    ? 10 * (1 - Math.exp(-s / 20))
    : 10 + 65 * (1 - Math.exp(-(s - 15) / 35));
  const vo2 = Math.min(100, +(vo2Raw / 75 * 100).toFixed(1));
  const o2deficit = +Math.max(0, +aerobic.toFixed(1) - vo2).toFixed(1);
  return {
    s,
    aerobic: +aerobic.toFixed(1),
    anaerobic: +(100 - aerobic).toFixed(1),
    vo2,
    vo2_base: vo2,
    o2deficit,
  };
});

const vo2SteadyPt = ENERGY_DATA.find(d => d.vo2 >= 90);

const ACCENT = "#E8C547";
const AEROBIC = "#5B9BD5";
const ANAEROBIC = "#C0504D";

export default function App() {
  const [sprint1s, setSprint1s] = useState(900);
  const [sprint30s, setSprint30s] = useState(620);
  const [power2min, setPower2min] = useState(340);
  const [weight, setWeight] = useState(75);
  const sprintFTP = Math.round(sprint30s * 0.38 + sprint1s * 0.08 + weight * 0.4 + 40);
  const subMaxFTP = Math.round(power2min * 0.78 + sprint30s * 0.09 + weight * 0.18 + 15);
  const shift = subMaxFTP - sprintFTP;
  const shiftPct = ((shift / sprintFTP) * 100).toFixed(1);
  const profile =
    shift > 15
      ? { label: "AEROBIC ENGINE", color: AEROBIC, desc: "2-min outperforms sprint prediction. Strong oxidative base — aerobic capacity is your edge." }
      : shift < -15
      ? { label: "NEUROMUSCULAR DOMINANT", color: ANAEROBIC, desc: "Sprint overpredicts. Aerobic engagement at 2 min underperforms. Prioritize VO\u2082 kinetics & base volume." }
      : { label: "BALANCED PROFILE", color: ACCENT, desc: "Systems well-integrated. Sprint and aerobic capacity proportionally developed. Train to race demands." };
  return (
    <div style={{ background: "#0d0d12", minHeight: "100vh", color: "#e8e8e8", fontFamily: "'IBM Plex Mono', monospace", padding: "40px 28px", maxWidth: 920, margin: "0 auto" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;600&family=Playfair+Display:ital,wght@0,700;1,400&display=swap');
        * { box-sizing: border-box; }
        input[type=range] { -webkit-appearance: none; width: 100%; height: 2px; background: #2a2a3a; outline: none; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #E8C547; cursor: pointer; }
      `}</style>
      <div style={{ borderBottom: "1px solid #1e1e30", paddingBottom: 28, marginBottom: 44 }}>
        <div style={{ fontSize: 10, letterSpacing: "0.2em", color: ACCENT, marginBottom: 10 }}>ULTRATHON.IO · SPRINT FTP PREDICTION</div>
        <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 34, fontWeight: 700, margin: 0 }}>The Physiological Switch</h1>
        <p style={{ fontFamily: "'Playfair Display', serif", fontStyle: "italic", color: "#888", fontSize: 15, margin: "8px 0 0" }}>What the math looks like</p>
      </div>
      <div style={{ maxWidth: 720, marginBottom: 48, paddingBottom: 40, borderBottom: "1px solid #1e1e30" }}>
        {/* Eyebrow + opening rule */}
        <div style={{ fontSize: 9, letterSpacing: "0.2em", color: "#333333", marginBottom: 12, fontFamily: "'IBM Plex Mono', monospace" }}>INTRODUCTION</div>
        <hr style={{ border: "none", borderTop: "1px solid #1e1e30", width: 72, margin: "0 0 28px 0" }} />

        {/* Paragraph 1 */}
        <p style={{ fontFamily: "'Playfair Display', serif", fontStyle: "italic", fontWeight: 400, fontSize: 13, color: ACCENT, letterSpacing: "0.04em", marginBottom: 10, marginTop: 0 }}>The Sprint Has a Ceiling</p>
        <p style={{ fontSize: 14, lineHeight: 1.85, color: "#aaaaaa", marginBottom: 0, marginTop: 0 }}>
          <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 42, fontWeight: 400, float: "left", lineHeight: 0.85, marginRight: 6, marginTop: 4, color: "#e8e8e8" }}>A</span>
          {" 30-second sprint tells you something real about a cyclist. It captures neuromuscular peak power, phosphocreatine capacity, and the explosive force-velocity output that decides a field sprint or a criterium attack. Across 4,768 cyclists, that single effort explains roughly half the variance in 20-minute FTP — a meaningful signal, and a hard ceiling. The gap in that ceiling is visible in the chart above: when the sprint ends at 30 seconds, VO\u2082 is still climbing. The aerobic engine is mid-ignition. The sprint test ends before the system it can't measure has finished turning on."}
        </p>

        <hr style={{ border: "none", borderTop: "1px solid #1e1e30", margin: "24px 0", clear: "both" }} />

        {/* Paragraph 2 */}
        <p style={{ fontFamily: "'Playfair Display', serif", fontStyle: "italic", fontWeight: 400, fontSize: 13, color: ACCENT, letterSpacing: "0.04em", marginBottom: 10, marginTop: 0 }}>The Ceiling Breaks at Two Minutes</p>
        <p style={{ fontSize: 14, lineHeight: 1.85, color: "#aaaaaa", marginBottom: 0, marginTop: 0 }}>{"The ceiling breaks at two minutes. A single 2-minute sub-maximal effort raises FTP prediction accuracy from R\u00b2=0.516 to R\u00b2=0.704 — an 18.8 percentage point jump, the largest single gain across the entire duration progression. By that point, aerobic metabolism has become the majority ATP supplier (~65%), VO\u2082 has completed its primary kinetic rise, and the O\u2082 deficit that accumulated during the sprint has largely resolved. The 2-minute effort and the 20-minute FTP test are physiological neighbors. The 30-second sprint belongs to a different domain entirely."}</p>

        <hr style={{ border: "none", borderTop: "1px solid #1e1e30", margin: "24px 0" }} />

        {/* Paragraph 3 */}
        <p style={{ fontFamily: "'Playfair Display', serif", fontStyle: "italic", fontWeight: 400, fontSize: 13, color: ACCENT, letterSpacing: "0.04em", marginBottom: 10, marginTop: 0 }}>The Math Is the Prescription</p>
        <p style={{ fontSize: 14, lineHeight: 1.85, color: "#aaaaaa", marginBottom: 0, marginTop: 0 }}>{"The Athlete Shift Calculator below quantifies what that transition means for you individually. Enter your sprint numbers and your 2-minute power — the shift score reveals the gap between what your neuromuscular output predicts and what your aerobic system actually delivers. A negative shift means your sprint outpaces your oxidative engagement; the aerobic engine needs work. A positive shift means the opposite — your aerobic base is the asset, and your training should leverage it. The math is the prescription."}</p>

        {/* Closing mark */}
        <div style={{ textAlign: "center", marginTop: 28, fontSize: 12, color: "#333333", fontFamily: "'IBM Plex Mono', monospace" }}>×</div>
      </div>
      {/* ── Formula block ── */}
      <div style={{ marginTop: 40, marginBottom: 56 }}>
        {/* Section header */}
        <div style={{ fontSize: 9, letterSpacing: "0.2em", color: "#333333", marginBottom: 12, fontFamily: "'IBM Plex Mono', monospace" }}>THE MATH</div>
        <hr style={{ border: "none", borderTop: "1px solid #1e1e30", width: 72, margin: "0 0 28px 0" }} />

        {/* 2×2 grid */}
        <div style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 1,
          background: "#1e1e30",
          border: "1px solid #1e1e30",
        }}>
          {[
            {
              label: "MODEL EXPLANATORY POWER",
              eq: (
                <span>
                  <span style={{ color: ACCENT }}>R²</span>
                  {" = 1 − "}
                  <span style={{ display: "inline-block", textAlign: "center", verticalAlign: "middle", lineHeight: 1.3 }}>
                    <span style={{ display: "block", borderBottom: "1px solid #888", paddingBottom: 2, marginBottom: 2 }}>
                      Σ(y<span style={{ fontSize: 11 }}>ᵢ</span> − ŷ<span style={{ fontSize: 11 }}>ᵢ</span>)²
                    </span>
                    <span style={{ display: "block" }}>
                      Σ(y<span style={{ fontSize: 11 }}>ᵢ</span> − ȳ)²
                    </span>
                  </span>
                </span>
              ),
            },
            {
              label: "INDIVIDUAL SHIFT SCORE",
              eq: (
                <span>
                  <span style={{ color: ACCENT }}>Δ<span style={{ fontSize: 12 }}>shift</span></span>
                  {" = FTP̂"}
                  <span style={{ fontSize: 12 }}>2min</span>
                  {" − FTP̂"}
                  <span style={{ fontSize: 12 }}>sprint</span>
                </span>
              ),
            },
            {
              label: "NORMALIZED SHIFT (%)",
              eq: (
                <span>
                  <span style={{ color: ACCENT }}>Δ<span style={{ fontSize: 12 }}>shift,%</span></span>
                  {" = "}
                  <span style={{ display: "inline-block", textAlign: "center", verticalAlign: "middle", lineHeight: 1.3 }}>
                    <span style={{ display: "block", borderBottom: "1px solid #888", paddingBottom: 2, marginBottom: 2 }}>
                      FTP̂<span style={{ fontSize: 12 }}>2min</span>
                      {" − FTP̂"}
                      <span style={{ fontSize: 12 }}>sprint</span>
                    </span>
                    <span style={{ display: "block" }}>
                      FTP̂<span style={{ fontSize: 12 }}>sprint</span>
                    </span>
                  </span>
                  {" × 100"}
                </span>
              ),
            },
            {
              label: "CONTINUUM POSITION",
              eq: (
                <span>
                  <span style={{ color: ACCENT }}>z<span style={{ fontSize: 12 }}>shift</span></span>
                  {" = "}
                  <span style={{ display: "inline-block", textAlign: "center", verticalAlign: "middle", lineHeight: 1.3 }}>
                    <span style={{ display: "block", borderBottom: "1px solid #888", paddingBottom: 2, marginBottom: 2 }}>
                      Δ<span style={{ fontSize: 12 }}>shift,%</span>
                      {" − μ"}
                      <span style={{ fontSize: 12 }}>Δ</span>
                    </span>
                    <span style={{ display: "block" }}>
                      σ<span style={{ fontSize: 12 }}>Δ</span>
                    </span>
                  </span>
                </span>
              ),
            },
          ].map((f, i) => (
            <div key={i} style={{ background: "#0d0d12", padding: "24px 20px" }}>
              <div style={{ fontSize: 9, letterSpacing: "0.18em", color: "#444444", fontFamily: "'IBM Plex Mono', monospace", marginBottom: 12 }}>{f.label}</div>
              <div style={{ fontFamily: "'Playfair Display', serif", fontStyle: "italic", fontSize: 17, color: "#e8e8e8", lineHeight: 1.5 }}>{f.eq}</div>
            </div>
          ))}
        </div>

        {/* Citation */}
        <div style={{ textAlign: "right", marginTop: 12, fontSize: 9, color: "#333333", fontFamily: "'IBM Plex Mono', monospace" }}>
          † Medbø &amp; Tabata 1989 · Gastin 2001 · Poole &amp; Jones 2012
        </div>
      </div>

      <div style={{ marginBottom: 52 }}>
        <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 20, fontWeight: 700, marginBottom: 2 }}>01 — Energy System Crossover</div>
        <div style={{ fontSize: 10, color: "#555", letterSpacing: "0.08em", marginBottom: 20 }}>Aerobic % contribution by effort duration · Medbø &amp; Tabata (1989) + Gastin (2001)</div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 14, marginBottom: 24 }}>
          {[
            { s: "30s", val: "~40%", note: "PCr + glycolysis dominant", col: ANAEROBIC },
            { s: "75s", val: "~50%", note: "crossover point", col: "#aaa" },
            { s: "2 min", val: "~65%", note: "aerobic majority \u2713", col: AEROBIC },
          ].map(d => (
            <div key={d.s} style={{ background: "#13131f", border: "1px solid #1e1e30", padding: "18px 20px" }}>
              <div style={{ fontSize: 10, letterSpacing: "0.12em", color: "#555", marginBottom: 6 }}>At {d.s}</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: d.col }}>{d.val}</div>
              <div style={{ fontSize: 10, color: "#444", marginTop: 4 }}>{d.note}</div>
            </div>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={ENERGY_DATA} margin={{ top: 4, right: 54, left: -18, bottom: 0 }}>
            <CartesianGrid strokeDasharray="2 6" stroke="#1a1a26" vertical={false} />
            <XAxis dataKey="s" tick={{ fill: "#555", fontSize: 10 }} axisLine={false} tickLine={false}
              tickFormatter={v => v === 0 ? "0s" : v === 30 ? "30s" : v === 75 ? "75s" : v === 120 ? "2min" : ""} />
            <YAxis yAxisId="left" domain={[0, 100]} tick={{ fill: "#444", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => v + "%"} />
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={[0, 100]}
              ticks={[0, 25, 50, 75, 100]}
              tick={{ fill: VO2_GREEN + "99", fontSize: 9 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={v => v + "%"}
              label={{ value: "% VO\u2082max", angle: 90, position: "insideRight", fill: VO2_GREEN, fontSize: 9, dx: 16 }}
            />
            <ReferenceLine yAxisId="left" x={30} stroke={ACCENT} strokeDasharray="4 4"
              label={{ value: "sprint ends", fill: ACCENT, fontSize: 9, position: "insideTopRight" }} />
            <ReferenceLine yAxisId="left" x={75} stroke="#ffffff22" strokeDasharray="4 4" />
            {vo2SteadyPt && (
              <ReferenceLine yAxisId="left" x={vo2SteadyPt.s} stroke={VO2_GREEN + "55"} strokeDasharray="3 3"
                label={{ value: "VO\u2082 ~steady state", fill: VO2_GREEN, fontSize: 8, position: "insideTopLeft" }} />
            )}
            <ReferenceLine yAxisId="left" x={48} stroke="transparent"
              label={{ value: "O\u2082 deficit", fill: VO2_GREEN + "aa", fontSize: 8, position: "insideTopLeft", fontFamily: "IBM Plex Mono" }} />
            <Tooltip
              contentStyle={{ background: "#13131f", border: "1px solid #2a2a3a", fontFamily: "IBM Plex Mono", fontSize: 11 }}
              formatter={(v, n) => {
                if (n === "aerobic") return [`${v}%`, "Aerobic"];
                if (n === "anaerobic") return [`${v}%`, "Anaerobic"];
                if (n === "vo2") return [`${v}%`, "VO\u2082 (% max)"];
                return [null, null];
              }}
            />
            {/* Stacked energy areas */}
            <Area yAxisId="left" type="monotone" dataKey="aerobic" stackId="1" stroke={AEROBIC} fill={AEROBIC + "55"} />
            <Area yAxisId="left" type="monotone" dataKey="anaerobic" stackId="1" stroke={ANAEROBIC} fill={ANAEROBIC + "44"} />
            {/* O₂ deficit shading: transparent VO₂ base + colored gap up to aerobic */}
            <Area yAxisId="left" type="monotone" dataKey="vo2_base" stackId="deficit" stroke="none" fill="transparent" activeDot={false} legendType="none" isAnimationActive={false} />
            <Area yAxisId="left" type="monotone" dataKey="o2deficit" stackId="deficit" stroke="none" fill={VO2_GREEN + "28"} activeDot={false} legendType="none" isAnimationActive={false} />
            {/* VO₂ kinetics line */}
            <Line yAxisId="right" type="monotone" dataKey="vo2" stroke={VO2_GREEN} strokeWidth={2} strokeDasharray="6 3" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
        <div style={{ display: "flex", gap: 24, marginTop: 10, paddingLeft: 10, alignItems: "center" }}>
          <span style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 9, color: "#555" }}>
            <svg width="18" height="8"><line x1="0" y1="4" x2="18" y2="4" stroke={VO2_GREEN} strokeWidth="2" strokeDasharray="5 3" /></svg>
            VO₂ rise (τ≈35s)
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: 7, fontSize: 9, color: "#555" }}>
            <span style={{ display: "inline-block", width: 14, height: 8, background: VO2_GREEN + "28", border: `1px solid ${VO2_GREEN}55` }}></span>
            O₂ deficit
          </span>
        </div>
      </div>
      <div style={{
        background: "linear-gradient(to top, #1a1a6e 0%, #0d0d2b 40%, #0a0a0f 100%)",
        border: "2px solid #E8C547",
        padding: "56px 48px",
        margin: "72px 0",
        position: "relative",
      }}>
        <div style={{ fontSize: 9, letterSpacing: "0.2em", color: "#E8C547", marginBottom: 14, fontFamily: "'IBM Plex Mono', monospace" }}>INTERACTIVE TOOL</div>
        <div style={{ fontFamily: "'Playfair Display', serif", fontSize: 32, fontWeight: 700, marginBottom: 8, lineHeight: 1.15 }}>02 — Athlete Shift Calculator</div>
        <div style={{ fontSize: 13, color: "#666", letterSpacing: "0.04em", marginBottom: 40 }}>Dial in your numbers · See where you sit on the sprint-to-endurance continuum</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 48 }}>
          <div>
            {[
              { label: "1s Peak Power", val: sprint1s, set: setSprint1s, min: 400, max: 1800, unit: "W" },
              { label: "30s Power", val: sprint30s, set: setSprint30s, min: 200, max: 1200, unit: "W" },
              { label: "2-Min Power", val: power2min, set: setPower2min, min: 100, max: 600, unit: "W" },
              { label: "Body Weight", val: weight, set: setWeight, min: 45, max: 120, unit: "kg" },
            ].map(d => (
              <div key={d.label} style={{ marginBottom: 32 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: "#888", marginBottom: 10 }}>
                  <span>{d.label}</span>
                  <span style={{ color: "#e8e8e8", fontWeight: 600, fontSize: 18 }}>{d.val}{d.unit}</span>
                </div>
                <input
                  type="range"
                  min={d.min}
                  max={d.max}
                  value={d.val}
                  onChange={e => d.set(+e.target.value)}
                  style={{
                    width: "100%",
                    appearance: "none",
                    height: 3,
                    background: `linear-gradient(to right, #E8C547 0%, #E8C547 ${((d.val - d.min) / (d.max - d.min)) * 100}%, #2a2a3a ${((d.val - d.min) / (d.max - d.min)) * 100}%, #2a2a3a 100%)`,
                    outline: "none",
                    cursor: "pointer",
                    borderRadius: 2,
                  }}
                />
              </div>
            ))}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            {[
              { label: "Sprint-Derived FTP", val: sprintFTP + "W", color: ANAEROBIC, tint: false },
              { label: "2-Min-Derived FTP", val: subMaxFTP + "W", color: AEROBIC, tint: false },
              { label: "Shift Score Δ", val: `${shift > 0 ? "+" : ""}${shift}W (${shift > 0 ? "+" : ""}${shiftPct}%)`, color: ACCENT, tint: true },
            ].map(d => (
              <div key={d.label} style={{
                background: d.tint ? "#E8C54708" : "#13131f",
                border: "1px solid #1e1e30",
                padding: "24px 28px",
              }}>
                <div style={{ fontSize: 11, letterSpacing: "0.12em", color: "#555", marginBottom: 8 }}>{d.label}</div>
                <div style={{ fontSize: 32, fontWeight: 600, color: d.color }}>{d.val}</div>
              </div>
            ))}
            <div style={{
              background: `${profile.color}05`,
              border: `2px solid ${profile.color}`,
              padding: "24px 28px",
            }}>
              <div style={{ fontSize: 9, letterSpacing: "0.2em", color: "#666", marginBottom: 6, fontFamily: "'IBM Plex Mono', monospace" }}>YOUR PHENOTYPE</div>
              <div style={{ fontSize: 13, fontWeight: 700, letterSpacing: "0.08em", color: profile.color, marginBottom: 8 }}>{profile.label}</div>
              <div style={{ fontSize: 13, color: "#888", lineHeight: 1.8 }}>{profile.desc}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
