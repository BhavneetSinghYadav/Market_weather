# Market_weather

**Market_weather** is a modular, Colab-first system that computes **market “weather”**—a real-time assessment of trading conditions—using **entropy** and **finite-time Lyapunov exponents (FTLE)** on **Polygon.io Currencies (FX/metals)** data (e.g., `C:XAUUSD`).  
It outputs a **GREEN / YELLOW / RED** state per minute plus diagnostics and a decision log.  
The same pipeline later runs on a small server with Polygon WebSockets.

> **What it is:** objective conditions & risk gating.  
> **What it isn’t:** trade calls or financial advice.

---

## Table of Contents
- [Design Goals](#design-goals)
- [Architecture Overview](#architecture-overview)
- [Data Contract](#data-contract)
- [Math & Model](#math--model)
- [Repository Layout](#repository-layout)
- [Quickstart (Colab)](#quickstart-colab)
- [Quickstart (Local Dev)](#quickstart-local-dev)
- [Running the Live Minute Loop](#running-the-live-minute-loop)
- [Artifacts & Outputs](#artifacts--outputs)
- [Parameters & Tuning](#parameters--tuning)
- [Quality, Safety & Ops](#quality-safety--ops)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License & Disclaimer](#license--disclaimer)

---

## Design Goals
1. **Causal**: uses only data available up to minute *t* (no look-ahead).  
2. **Deterministic**: same inputs → same outputs; everything versioned and hashed.  
3. **Resilient**: graceful under API hiccups; explicit stale/insufficient-window banners.  
4. **Auditable**: every minute’s decision is logged with inputs and reasons.  
5. **Modular**: adapters, canonicalizer, features, scoring, loop, viz are separable.

---

## Architecture Overview

**Layers**
- **Adapters**: fetch FX/minute aggregates (REST today; add quotes & WebSockets later).  
- **Canonicalizer**: ET→UTC, strict 1-min grid, explicit gaps, OHLC integrity checks; Parquet + metadata.  
- **Feature Engine**: first-order (Permutation Entropy, FTLE), second-order (Δ/Δ², tension), third-order (hazard proxies).  
- **Scoring**: \(\mathcal T \in [0,1]\) → **RED / YELLOW / GREEN** with hysteresis.  
- **Live Loop**: poll → append → compute → score → log → plot (per minute).  
- **Viz**: price with shaded background + diagnostics panels.  
- **Persistence**: Parquet (bars, features), CSV (decision log), JSON (session summary).

---

## Data Contract

**Index**: tz-aware `DatetimeIndex` in **UTC**, strictly increasing, **1-minute regular grid**.  
**Columns** (floats unless noted):
- `open, high, low, close, volume` *(FX “volume” is activity proxy from quotes)*  
- optional: `bid, ask`  
- flags: `is_session` (bool), `is_gap` (bool), `quality_score` (0–1)  
- helper: `minute_of_day` (0–1439)

**Rules**
- **No forward-fill** of OHLC; gaps remain `NaN` and break windows.  
- Duplicates → keep **last**, log counts.  
- Downsample only (1→5 min: O=first, H=max, L=min, C=last, V=sum).  
- FX sessions 24×5; weekends masked.

**Persistence**
- Parquet per day: `data/canonical/XAUUSD/1min/XAUUSD_1min_YYYY-MM-DD.parquet`  
- Sidecar meta (`.meta.json`): source, source_time_basis, loaded_at, row/gap/dup/clip counts, sha256, `contract_version`.

---

## Math & Model

**Inputs**: 1-min closes \(C_t\), returns \(r_t=\ln(C_t/C_{t-1})\).

### First-Order Diagnostics (micro scale v1)
- **Permutation Entropy (PE)**  
  - Window \(W=60\), embedding \(m=3\), delay \(τ=1\).  
  - Ordinal patterns \(\pi\) → frequencies \(p(\pi)\) →  
    \(E^{\text{PE}} = -\sum p(\pi)\log p(\pi) / \log(m!) \in [0,1]\).
- **Finite-Time Lyapunov Exponent (Rosenstein)**  
  - Embedding \(m=3\), \(τ=1\); horizon \(h=10\); Theiler exclusion \(w_T\ge τ\).  
  - Delay-embed → nearest neighbor (exclude temporal neighbors) → fit slope of \(\ln d(k)\) vs \(k\) → **median** slope = \(\Lambda\).

**Normalization & Smoothing**
- Causal min-max (v1) or time-of-day percentiles (v2) ⇒ \(\widehat{E},\widehat{\Lambda}\in[0,1]\).  
- EMA(3–5) on diagnostics (never smooth price).

**Second-Order**
- Velocities \(V_E,V_\Lambda\), curvatures \(C_E,C_\Lambda\), tension \(T=\alpha(1-\widehat E)-\beta\widehat\Lambda\).

**Tradability Score & State**
- \(\mathcal T = 0.6(1-\widehat E) + 0.4(1-\widehat{\Lambda})\).  
- Thresholds: **GREEN** \(\ge 0.65\), **YELLOW** \(0.45–0.65\), **RED** \(<0.45\) with hysteresis (e.g., `k_up=2`, `k_down=1`, min flip spacing).

---

## Repository Layout

market-weather/
notebooks/
00_config_and_paths.ipynb
10_ingest_canonicalize.ipynb
20_features_first_order.ipynb
40_tradability_scoring_and_state.ipynb
50_live_minute_loop_colab.ipynb
mw/
adapters/ # polygon REST (v2 aggs; v3 quotes later)
io/ # canonicalizer, manifests
features/ # entropy.py, ftle.py, scaling.py, smoothing.py
scoring/ # tradability.py (score + hysteresis), sizing (later)
live/ # minute_loop.py orchestrator, health.py
viz/ # plots.py (price + bands; diagnostics)
utils/ # time, ohlc_checks, persistence
params/ # params_v1.json (windows, thresholds, hysteresis)
tests/ # unit tests & causality checks
data/ # (gitignored) Parquet, logs, summaries
.github/workflows/ci.yml
.pre-commit-config.yaml
.gitattributes
.gitignore
requirements.txt
README.md
LICENSE

---

## Quickstart (Colab)

1. Open `notebooks/50_live_minute_loop_colab.ipynb`.  
2. `pip install -r requirements.txt` (or use pre-installed in Colab).  
3. Set your Polygon key as an **environment variable** in Colab:
   ```python
   import os
   os.environ["POLYGON_API_KEY"] = "YOUR_KEY"
4. Run the warm-up cell (pull last ~1000 1-min bars for C:XAUUSD via v2 aggs).
5. Start the minute loop: poll at t+3–5s, append-if-advanced, compute PE/FTLE, score, plot, and log.

---

## Quickstart (Local Dev)

python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# store your key locally (never commit .env)
echo "POLYGON_API_KEY=YOUR_KEY" > .env

# optional formatting/lint hooks
pip install pre-commit
pre-commit install
pre-commit run --all-files

---

## Running the Live Minute Loop

# Flow per minute:
1. Poll /v2/aggs/ticker/C:XAUUSD/range/1/minute/{from}/{to} at t+3–5s.
2. Append only if timestamp advanced; no partial bars.
3. Compute micro PE(60) & FTLE(200) → normalize → EMA → 𝑇
4. Map to R/Y/G with hysteresis.
5. Update plots & write a decision log row.

Staleness: `if no new bar for >180s, freeze state and show STALE banner`.

---

## Parameters & Tuning

File: `params/params_v1.json (single source of truth)`.
1. pe.window, pe.m, pe.tau
2. ftle.window, ftle.m, ftle.tau, ftle.horizon, ftle.theiler
3. smoothing.ema_span
4. score.{w1,w2,tau_y,tau_g,k_up,k_down,min_flip_spacing}
5. normalization.method = minmax (v1) or tod_percentile (v2)

Calibration (later): `use walk-forward tests, purged K-fold with embargo; choose thresholds to reduce whipsaws & drawdown`.

---

## Quality, Safety & Ops

1. Integrity checks: L ≤ min(O,C) ≤ max(O,C) ≤ H; clip/drop + log.
2. Gaps: explicit NaN; diagnostics skip windows spanning gaps.
3. Rate limits / 5xx: exponential backoff + jitter; circuit breaker after 3 fails.
4. Causality guard: no look-ahead; replay equality test ensures live==batch.
5. Line endings: LF in repo; notebooks tracked as binary via .gitattributes.
6. CI: GitHub Actions runs black/isort/flake8 + tests on push/PR.
7. Secrets: never commit API keys; use .env locally and Colab os.environ.

---

## Roadmap

1. Macro/system scales (SampEn/MSE, longer-horizon FTLE).
2. Cross-scale coherence & “tension” composites.
3. Hazard models (whipsaw/tail probabilities) from offline calibration.
4. WebSocket client on a small VM; quotes stream for nano diagnostics.
5. Telegram bot: 09:00 outlook + 15-min weather + hourly recap.
6. Size bands (Pro tier) conditioned on 𝑇 and tail risk.

---

## Troubleshooting

1. No bar arrived: `heck clock; poll at t+3–5s. Quiet minutes may be empty—your grid will show gaps`.
2. State flicker: `increase EMA span or hysteresis (k_up, min_flip_spacing)`.
3. “Volume” confusion: `FX aggregates are quote-derived; treat as activity proxy`.
4. Timezone weirdness: `canonicalizer converts ET→UTC; never compute on ET`.

---

## License & Disclaimer

MIT License (fill in LICENSE).
This project is for research/education. `It provides market conditions, not trading advice. No warranty; you are responsible for your risk`.

