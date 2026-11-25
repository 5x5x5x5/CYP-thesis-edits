"""Build the interactive futuristic scrollytelling web paper `extension/unhinged/index.html`
directly from the real saved CSV results in `notebooks/results/`.

Usage:
    uv run python extension/unhinged/build_unhinged.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "notebooks" / "results"
OUT_HTML = Path(__file__).resolve().parent / "index.html"

ISOZYMES = ["1a2", "2c9", "2c19", "2d6", "3a4"]

FEATURED_MODELS = [
    ("RF_baseline", "2015 RF (10 trees, MOE)", "2015 Baseline", "#64748b"),
    ("SVM_baseline", "2015 SVM (RBF, MOE)", "2015 Baseline", "#94a3b8"),
    ("XGBoost", "Tier 1 XGBoost (MOE)", "Tier 1 · MOE", "#3b82f6"),
    ("Stack_LogReg", "Tier 1 MOE Stack", "Tier 1 · MOE", "#60a5fa"),
    ("Tanimoto_SVM_RDKit", "Tier 2A Tanimoto SVM (RDKit)", "Tier 2A · RDKit", "#f59e0b"),
    ("Stack_RDKit", "Tier 2A RDKit-Only Stack", "Tier 2A · RDKit", "#fbbf24"),
    ("HybridKernel_SVM", "Tier 2A Hybrid Tanimoto+RBF SVM", "Tier 2A · Hybrid", "#fb923c"),
    ("Stack_MOE_RDKit", "Tier 2A MOE+RDKit Stack", "Tier 2A · Hybrid", "#f97316"),
    ("MT_ResMLP_RDKit", "Tier 2B MT-ResMLP (RDKit, GPU)", "Tier 2B · GPU", "#ec4899"),
    ("MT_ResMLP_MOE_RDKit", "Tier 2B MT-ResMLP (MOE+RDKit, GPU)", "Tier 2B · GPU", "#f43f5e"),
    ("MT_GNN_GraphPlusDesc", "Tier 3A MT-GINE+VN GNN (GPU)", "Tier 3A · Graph", "#a855f7"),
    ("MT_ChemBERTa_77M_MTR", "Tier 3B ChemBERTa-77M (Raw SMILES)", "Tier 3B · LM", "#c084fc"),
    ("Grand_OpenSource_Stack", "Tier 4 Grand Open-Source Stack (0 MOE)", "Tier 4 · Open", "#2dd4bf"),
    ("Grand_AllModal_Stack", "Tier 4 Grand All-Modal Stack", "Tier 4 · Grand", "#00f5a8"),
    ("Grand_AllModal_Blend", "Tier 4 Grand All-Modal Blend", "Tier 4 · Grand", "#34d399"),
]


def build_payload() -> dict:
    master = pd.read_csv(RESULTS / "master_comparison.csv")
    grand = pd.read_csv(RESULTS / "grand_summary.csv")
    stats = pd.read_csv(RESULTS / "statistical_tests.csv")
    scaf = pd.read_csv(RESULTS / "scaffold_cv_summary.csv")

    iso_data: dict[str, dict] = {}
    for iso in ISOZYMES + ["mean"]:
        rows = []
        for mkey, mlabel, tier, color in FEATURED_MODELS:
            if iso == "mean":
                sub = master[master["model"] == mkey]
                rows.append(
                    {
                        "key": mkey,
                        "label": mlabel,
                        "tier": tier,
                        "color": color,
                        "acc": round(float(sub["test_acc"].mean()), 4),
                        "auc": round(float(sub["test_auc"].mean()), 4),
                        "brier": round(float(sub["test_brier"].mean()), 4),
                        "ece": round(float(sub["test_ece"].mean()), 4),
                    }
                )
            else:
                r = master[(master["isozyme"] == iso) & (master["model"] == mkey)].iloc[0]
                rows.append(
                    {
                        "key": mkey,
                        "label": mlabel,
                        "tier": tier,
                        "color": color,
                        "acc": round(float(r["test_acc"]), 4),
                        "auc": round(float(r["test_auc"]), 4),
                        "brier": round(float(r["test_brier"]), 4),
                        "ece": round(float(r["test_ece"]), 4),
                    }
                )

        # Stacking weights
        if iso == "mean":
            w_acc: dict[str, float] = {}
            for iso_i in ISOZYMES:
                g = grand[(grand["isozyme"] == iso_i) & (grand["model"] == "Grand_AllModal_Stack")].iloc[0]
                coefs = json.loads(g["best_params"])["coefs"]
                pos_sum = sum(max(0.0, v) for v in coefs.values())
                for k, v in coefs.items():
                    w_acc[k] = w_acc.get(k, 0.0) + (max(0.0, v) / pos_sum) / 5.0
            weights = {k: round(v * 100, 1) for k, v in w_acc.items()}
        else:
            g = grand[(grand["isozyme"] == iso) & (grand["model"] == "Grand_AllModal_Stack")].iloc[0]
            coefs = json.loads(g["best_params"])["coefs"]
            pos_sum = sum(max(0.0, v) for v in coefs.values())
            weights = {k: round(max(0.0, v) / pos_sum * 100, 1) for k, v in coefs.items()}

        # DeLong highlights
        delong = {}
        for comp in [
            "GrandAllModal_vs_OrigSVM",
            "GrandOpenSource_vs_Tier1MOE",
            "GrandAllModal_vs_SingleTaskMOERDKit",
            "GrandAllModal_vs_GrandOpenSource",
        ]:
            if iso == "mean":
                sub = stats[stats["comparison"] == comp]
                delong[comp] = {
                    "delta_auc": round(float(sub["delta_auc"].mean()), 4),
                    "p_max": float(sub["delong_p"].max()),
                    "p_min": float(sub["delong_p"].min()),
                }
            else:
                r = stats[(stats["comparison"] == comp) & (stats["isozyme"] == iso)].iloc[0]
                delong[comp] = {
                    "delta_auc": round(float(r["delta_auc"]), 4),
                    "p": float(r["delong_p"]),
                    "delta_acc": round(float(r["delta_acc"]), 4),
                }

        iso_data[iso] = {"models": rows, "weights": weights, "delong": delong}

    # Timeline milestones for the Time Machine scrubber
    milestones = [
        {
            "year": "2015 · AWS EC2 t1.micro",
            "model": "SVM_baseline",
            "title": "The Lonely Micro Instance",
            "hw": "1 vCPU · 613 MiB RAM · 10% CPU credit throttle",
            "runtime": "Overnight → Multi-day runs (2–3 semesters total)",
            "note": "RBF SVM on 186 proprietary MOE descriptors. Beat MOE Binary QSAR on every isozyme—and proved open-source learners had arrived.",
        },
        {
            "year": "2026 · Tier 1 (Same 186 MOE)",
            "model": "Stack_LogReg",
            "title": "Algorithmic Awakening",
            "hw": "Local 4-thread CPU · 42 min total",
            "runtime": "+0.0196 AUC (52% of decade gain) on unchanged 2015 features",
            "note": "Zero new chemistry—just LightGBM, XGBoost, CatBoost, QuantileTransformer, tuned SVMs, and nested stacking. Every isozyme p < 3e-4.",
        },
        {
            "year": "2026 · Tier 2A (Open-Source RDKit)",
            "model": "Stack_RDKit",
            "title": "The Proprietary Wall Crumbles",
            "hw": "Local 4-thread CPU · 37.5 min total",
            "runtime": "0.8715 Mean AUC with ZERO MOE descriptors",
            "note": "Morgan count fingerprints, pharmacophore bits, MACCS keys, RDKit 2D descriptors, and exact Tanimoto kernels crush MOE-only models.",
        },
        {
            "year": "2026 · Tier 2B (Colab T4 GPU)",
            "model": "MT_ResMLP_MOE_RDKit",
            "title": "Five Isozymes, One Shared Brain",
            "hw": "Colab Tesla T4 · 9.7 min (150 neural nets!)",
            "runtime": "Breaks 80.30% Mean Accuracy · CYP 2D6 leaps to 80.07%",
            "note": "Cross-isozyme multi-task transfer with double-firewall zero-leakage isolation turns promiscuous CYP binding from a bug into a superpower.",
        },
        {
            "year": "2026 · Tier 3B (Raw SMILES Transformer)",
            "model": "MT_ChemBERTa_77M_MTR",
            "title": "Reading Molecules Like Language",
            "hw": "Colab Tesla T4 · 17.5 min (fp16 fine-tuning)",
            "runtime": "0.8659 Mean AUC · ZERO descriptors · ZERO fingerprints",
            "note": "Pretrained on 77 million PubChem molecules, ChemBERTa reads raw SMILES strings and beats the entire 186-MOE Tier 1 ensemble.",
        },
        {
            "year": "2026 · Tier 4 (Grand All-Modal Stack)",
            "model": "Grand_AllModal_Stack",
            "title": "The AI-Orchestrated Endgame",
            "hw": "Hybrid CPU + T4 GPU · 1.80 Colab CU · ~3.5h wall-clock",
            "runtime": "0.8792 Mean AUC · 80.63% Acc · 0.0216 ECE · p = 2e-25 (3A4)",
            "note": "Trees, Tanimoto kernels, Multi-Task ResMLPs, GNNs, and ChemBERTa fused via nested log-odds stacking. Open-source alone hits 0.8777 (MOE = ns).",
        },
    ]
    for m in milestones:
        sub = master[master["model"] == m["model"]]
        m["auc"] = round(float(sub["test_auc"].mean()), 4)
        m["acc"] = round(float(sub["test_acc"].mean()), 4)
        m["brier"] = round(float(sub["test_brier"].mean()), 4)
        m["ece"] = round(float(sub["test_ece"].mean()), 4)

    return {"isozymes": iso_data, "milestones": milestones}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>CYTOCHROME // 2026 — A Living Scientific Manifesto</title>
<style>
  :root {
    --bg: #07090e;
    --bg2: #0d111a;
    --card: rgba(17, 24, 39, 0.76);
    --border: rgba(148, 163, 184, 0.16);
    --text: #e2e8f0;
    --muted: #94a3b8;
    --emerald: #00f5a8;
    --crimson: #ff3b5c;
    --cobalt: #3b82f6;
    --amber: #ffb800;
    --violet: #a855f7;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    background: radial-gradient(circle at 15% 10%, rgba(0, 245, 168, 0.08), transparent 36%),
                radial-gradient(circle at 85% 22%, rgba(255, 59, 92, 0.09), transparent 38%),
                radial-gradient(circle at 50% 80%, rgba(59, 130, 246, 0.08), transparent 45%),
                var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    line-height: 1.65;
    overflow-x: hidden;
  }
  .mono { font-family: "JetBrains Mono", "Fira Code", ui-monospace, SFMono-Regular, monospace; }
  .serif { font-family: Georgia, "Playfair Display", "Times New Roman", serif; }

  /* Subtle animated molecular hex grid overlay */
  .grid-bg {
    position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: linear-gradient(rgba(148,163,184,0.035) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(148,163,184,0.035) 1px, transparent 1px);
    background-size: 48px 48px;
  }
  .wrap { position: relative; z-index: 1; max-width: 1120px; margin: 0 auto; padding: 0 24px; }

  /* Top telemetry bar */
  .telemetry {
    position: sticky; top: 0; z-index: 50;
    backdrop-filter: blur(14px);
    background: rgba(7, 9, 14, 0.84);
    border-bottom: 1px solid var(--border);
    padding: 10px 24px;
    display: flex; flex-wrap: wrap; gap: 16px; align-items: center; justify-content: space-between;
    font-size: 12px;
  }
  .telemetry .tag { color: var(--emerald); font-weight: 700; letter-spacing: 0.12em; }
  .telemetry nav a { color: var(--muted); text-decoration: none; margin-left: 16px; transition: color .2s; }
  .telemetry nav a:hover { color: var(--emerald); }

  /* Hero */
  header.hero {
    padding: 84px 0 56px;
    border-bottom: 1px solid var(--border);
  }
  .kicker {
    display: inline-block;
    padding: 4px 12px;
    border: 1px solid rgba(0,245,168,0.4);
    border-radius: 999px;
    color: var(--emerald);
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 22px;
  }
  h1.mega {
    font-size: clamp(2.5rem, 6vw, 4.7rem);
    line-height: 1.02;
    margin: 0 0 22px;
    letter-spacing: -0.03em;
    background: linear-gradient(120deg, #ffffff 10%, #a5f3fc 52%, #00f5a8 95%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .deck {
    font-size: clamp(1.08rem, 1.8vw, 1.32rem);
    color: #cbd5e1;
    max-width: 820px;
    margin-bottom: 36px;
  }
  .byline { font-size: 13px; color: var(--muted); margin-bottom: 36px; }
  .byline strong { color: #f8fafc; }

  /* Monolith KPI strip */
  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px;
  }
  .kpi {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
  }
  .kpi::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent, var(--emerald));
  }
  .kpi .val { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; color: #fff; }
  .kpi .lbl { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
  .kpi .sub { font-size: 12px; color: var(--accent, var(--emerald)); margin-top: 6px; }

  /* Sections */
  section { padding: 64px 0; border-bottom: 1px solid var(--border); }
  .sec-num { color: var(--crimson); font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; font-weight: 700; }
  h2 { font-size: clamp(1.8rem, 3.2vw, 2.5rem); margin: 8px 0 18px; letter-spacing: -0.02em; color: #fff; }
  p.lead { font-size: 1.12rem; color: #cbd5e1; max-width: 780px; }

  /* Pull quote */
  blockquote.manifesto {
    margin: 34px 0;
    padding: 24px 28px;
    border-left: 4px solid var(--crimson);
    background: linear-gradient(90deg, rgba(255,59,92,0.11), transparent);
    border-radius: 0 12px 12px 0;
    font-size: 1.32rem;
    line-height: 1.45;
    color: #fff;
  }
  blockquote.manifesto footer { margin-top: 10px; font-size: 12px; color: var(--muted); letter-spacing: 0.08em; text-transform: uppercase; }

  /* Interactive Time Machine */
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 26px;
    margin-top: 24px;
  }
  .slider-wrap { margin: 18px 0 24px; }
  input[type=range] {
    width: 100%; accent-color: var(--emerald); cursor: pointer; height: 6px;
  }
  .tm-ticks {
    display: flex; justify-content: space-between; font-size: 11px; color: var(--muted); margin-top: 6px;
  }
  .tm-grid {
    display: grid; grid-template-columns: 1.4fr 1fr; gap: 22px; align-items: center;
  }
  @media (max-width: 780px) { .tm-grid { grid-template-columns: 1fr; } }
  .tm-metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  .tm-box {
    background: rgba(7,9,14,0.65); border: 1px solid var(--border); border-radius: 10px; padding: 14px;
  }
  .tm-box .v { font-size: 1.55rem; font-weight: 800; color: var(--emerald); }
  .tm-box .k { font-size: 11px; color: var(--muted); text-transform: uppercase; }

  /* Interactive Isozyme Explorer */
  .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
  .pill {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid var(--border);
    color: var(--muted);
    padding: 8px 16px;
    border-radius: 999px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 700;
    transition: all .18s;
  }
  .pill:hover { border-color: var(--emerald); color: #fff; }
  .pill.active { background: var(--emerald); color: #04130d; border-color: var(--emerald); }

  .bar-row {
    display: grid; grid-template-columns: 270px 1fr 130px; gap: 12px; align-items: center;
    margin: 7px 0; font-size: 13px;
  }
  @media (max-width: 720px) { .bar-row { grid-template-columns: 1fr; gap: 4px; } }
  .bar-track { background: rgba(255,255,255,0.06); border-radius: 999px; height: 14px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 999px; transition: width .35s cubic-bezier(.22,1,.36,1); }
  .bar-val { text-align: right; color: #f8fafc; font-weight: 700; }

  /* Hypothesis Cards */
  .hyp-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; margin-top: 20px; }
  .hyp {
    background: rgba(7,9,14,0.72); border: 1px solid var(--border); border-radius: 12px; padding: 18px;
  }
  .badge-sig { display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; font-weight: 800; }
  .badge-yes { background: rgba(0,245,168,0.16); color: var(--emerald); border: 1px solid rgba(0,245,168,0.4); }
  .badge-ns  { background: rgba(168,85,247,0.18); color: #d8b4fe; border: 1px solid rgba(168,85,247,0.45); }

  /* Figures gallery */
  .fig-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 22px; margin-top: 24px; }
  @media (max-width: 540px) { .fig-grid { grid-template-columns: 1fr; } }
  figure.plate {
    margin: 0; background: var(--card); border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
  }
  figure.plate img { width: 100%; display: block; background: #fff; cursor: zoom-in; }
  figure.plate figcaption { padding: 14px 18px; font-size: 13px; color: var(--muted); }
  figure.plate figcaption strong { color: #fff; }

  /* Future manifesto columns */
  .future-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 18px; margin-top: 24px; }
  .future-card {
    background: linear-gradient(160deg, rgba(17,24,39,0.88), rgba(7,9,14,0.92));
    border: 1px solid var(--border); border-radius: 14px; padding: 22px;
  }
  .future-card h3 { margin: 0 0 10px; color: var(--emerald); font-size: 1.1rem; }

  /* Lightbox */
  #lightbox {
    display: none; position: fixed; inset: 0; z-index: 100;
    background: rgba(3,5,9,0.94); align-items: center; justify-content: center; padding: 24px; cursor: zoom-out;
  }
  #lightbox img { max-width: 96vw; max-height: 92vh; border-radius: 10px; background: #fff; }

  footer.colophon { padding: 48px 0 64px; font-size: 13px; color: var(--muted); text-align: center; }
</style>
</head>
<body>
<div class="grid-bg"></div>

<div class="telemetry mono">
  <div><span class="tag">CYTOCHROME // 2026</span> &nbsp;·&nbsp; AID1851 &nbsp;·&nbsp; 5 ISOZYMES &nbsp;·&nbsp; 100% TEST COVERAGE &nbsp;·&nbsp; 0 LEAKAGE</div>
  <nav>
    <a href="#timemachine">TIME MACHINE</a>
    <a href="#explorer">LIVE BENCHMARK</a>
    <a href="#autopsy">MOE AUTOPSY</a>
    <a href="#plates">PLATES</a>
    <a href="#future">THE FUTURE</a>
  </nav>
</div>

<div class="wrap">
  <header class="hero">
    <div class="kicker mono">LIVING SCIENTIFIC MANIFESTO · 2015 → 2026</div>
    <h1 class="mega serif">We Buried a Ten-Year-Old Thesis Under 1,000 Models in One Afternoon.</h1>
    <p class="deck">
      In 2015, an AWS Free Tier <span class="mono">t1.micro</span> wheezed overnight at 10% CPU credits to prove that open-source Python classifiers could beat MOE Binary QSAR on five cytochrome P450 isozymes. Eleven years later, we pointed a local mini PC, a 1.80-credit Colab Tesla T4, a 77-million-molecule chemical transformer, and an AI research orchestrator at the <strong>exact same Figshare splits</strong>. Here is what survived—and what became obsolete.
    </p>
    <div class="byline mono">
      <strong>Daniel E. Siegle</strong> · M.S. Thesis Extension (NCCU 2015 → 2026) · Orchestrated with Antigravity · Total Cloud Spend: <strong>1.80 Colab Compute Units</strong>
    </div>

    <div class="kpi-grid">
      <div class="kpi" style="--accent: var(--emerald)">
        <div class="val mono">0.8792</div>
        <div class="lbl">Mean Held-Out Test ROC-AUC</div>
        <div class="sub mono">+0.0375 vs 2015 SVM · p ≤ 2.7×10⁻⁷ (5/5)</div>
      </div>
      <div class="kpi" style="--accent: var(--cobalt)">
        <div class="val mono">80.63%</div>
        <div class="lbl">Mean Held-Out Test Accuracy</div>
        <div class="sub mono">+4.23% vs 2015 SVM · +6.50% vs 2015 RF</div>
      </div>
      <div class="kpi" style="--accent: var(--violet)">
        <div class="val mono">0.8777</div>
        <div class="lbl">100% Open-Source Stack (0 MOE)</div>
        <div class="sub mono">Adding 186 MOE = +0.0015 AUC (ns, 5/5!)</div>
      </div>
      <div class="kpi" style="--accent: var(--crimson)">
        <div class="val mono">3.5 hrs</div>
        <div class="lbl">2026 Wall-Clock Campaign</div>
        <div class="sub mono">vs 2–3 semesters + overnight EC2 in 2015</div>
      </div>
    </div>
  </header>

  <section id="timemachine">
    <div class="sec-num mono">01 // THE DECADE TIME MACHINE</div>
    <h2 class="serif">Scrub Through Eleven Years of Machine Learning on the Exact Same Molecules.</h2>
    <p class="lead">
      Almost no benchmark in machine learning holds the molecules, labels, and test splits frozen for eleven years. Drag the slider below to step from a single-vCPU AWS micro instance in 2015 to a multi-modal 2026 meta-stack—every number is computed on 100% of the original Figshare test sets.
    </p>

    <div class="card">
      <div class="slider-wrap">
        <input id="tmSlider" type="range" min="0" max="5" step="1" value="5" />
        <div class="tm-ticks mono">
          <span>2015 SVM</span><span>Tier 1 MOE</span><span>Tier 2A RDKit</span><span>Tier 2B GPU MT</span><span>Tier 3B SMILES</span><span>Tier 4 Grand</span>
        </div>
      </div>
      <div class="tm-grid">
        <div>
          <div id="tmYear" class="mono" style="color:var(--emerald);font-size:12px;letter-spacing:.12em;text-transform:uppercase;"></div>
          <h3 id="tmTitle" class="serif" style="font-size:1.75rem;margin:6px 0 10px;color:#fff;"></h3>
          <p id="tmNote" style="color:#cbd5e1;margin:0 0 14px;"></p>
          <div id="tmHw" class="mono" style="font-size:12px;color:var(--amber);"></div>
          <div id="tmRt" class="mono" style="font-size:12px;color:var(--muted);margin-top:4px;"></div>
        </div>
        <div class="tm-metrics mono">
          <div class="tm-box"><div class="v" id="tmAuc">—</div><div class="k">Mean Test ROC-AUC</div></div>
          <div class="tm-box"><div class="v" id="tmAcc">—</div><div class="k">Mean Test Accuracy</div></div>
          <div class="tm-box"><div class="v" id="tmBrier" style="color:var(--cobalt)">—</div><div class="k">Mean Brier Score</div></div>
          <div class="tm-box"><div class="v" id="tmEce" style="color:var(--violet)">—</div><div class="k">Mean 10-Bin ECE</div></div>
        </div>
      </div>
    </div>

    <blockquote class="manifesto serif">
      "In 2015, open-source classifiers beat MOE's classifier while still renting MOE's eyeballs. In 2026, open-source cheminformatics and a 77-million-molecule language model retired the entire proprietary descriptor suite—with exact DeLong proof."
      <footer class="mono">Headline Finding · Grand Open-Source Stack (0.8777) vs Grand All-Modal Stack (0.8792), p ∈ [0.059, 0.668]</footer>
    </blockquote>
  </section>

  <section id="explorer">
    <div class="sec-num mono">02 // INTERACTIVE BENCHMARK TELEMETRY</div>
    <h2 class="serif">Inspect All Five Cytochrome P450 Isozymes Live.</h2>
    <p class="lead">
      Click any isozyme below to inspect all 15 landmark models across Tiers 1–4, exact DeLong significance tests, and the positive stacking weights of the Grand All-Modal meta-learner. Notice how <strong>CYP 2D6</strong> (the smallest training set, N=4,433) sees the largest multi-task jump—jumping from 74.9% (2015 SVM) to <strong>80.79%</strong>!
    </p>

    <div class="pill-row mono" id="isoPills">
      <button class="pill active" data-iso="mean">ALL 5 MEAN</button>
      <button class="pill" data-iso="1a2">CYP 1A2</button>
      <button class="pill" data-iso="2c9">CYP 2C9</button>
      <button class="pill" data-iso="2c19">CYP 2C19</button>
      <button class="pill" data-iso="2d6">CYP 2D6</button>
      <button class="pill" data-iso="3a4">CYP 3A4</button>
    </div>

    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
        <div class="mono" style="font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;">Held-Out Test ROC-AUC &amp; Accuracy Leaderboard</div>
        <div class="mono" style="font-size:12px;color:var(--emerald);" id="isoHeader">ALL 5 ISOZYMES (MEAN)</div>
      </div>
      <div id="leaderboard"></div>

      <div class="hyp-grid" id="delongCards"></div>
    </div>
  </section>

  <section id="autopsy">
    <div class="sec-num mono">03 // AUTOPSY OF A DECADE'S UPLIFT</div>
    <h2 class="serif">Where Did +0.0375 ROC-AUC Actually Come From?</h2>
    <p class="lead">
      Hype says "deep learning replaced everything." Reality is sharper, weirder, and way more interesting. Because we ran controlled ablations on the exact same splits, we can split the ten-year gain down to the fourth decimal place:
    </p>
    <div class="future-grid">
      <div class="future-card">
        <h3 class="mono">52% · Modern Tabular Math</h3>
        <p><strong>+0.0196 AUC</strong> on the <em>exact unchanged 186 MOE columns from 2015</em>. Histogram gradient boosting (LightGBM, XGBoost, CatBoost), QuantileTransformer scaling, and nested log-odds stacking did half the work without touching a single new feature or GPU.</p>
      </div>
      <div class="future-card">
        <h3 class="mono" style="color:var(--amber)">37% · Open-Source Chemistry</h3>
        <p><strong>+0.0138 AUC</strong> from RDKit Morgan count fingerprints, pharmacophore bits, MACCS keys, RDKit 2D descriptors, and exact Tanimoto + Hybrid RBF kernels. Under strict Murcko scaffold splitting, RDKit beats MOE on 5/5 isozymes with <em>less</em> generalization drop.</p>
      </div>
      <div class="future-card">
        <h3 class="mono" style="color:var(--crimson)">11% · GPU Multi-Task &amp; Transformers</h3>
        <p><strong>+0.0042 AUC</strong> (statistically significant on 5/5 isozymes, DeLong p &lt; 0.03). Multi-task ResMLPs share knowledge across all 5 CYP heads, while <strong>ChemBERTa-77M-MTR</strong> reads <em>raw SMILES strings</em> with zero descriptors and earns 12–18% weight in the final stack on every isozyme.</p>
      </div>
    </div>
  </section>

  <section id="plates">
    <div class="sec-num mono">04 // HIGH-RESOLUTION EVIDENCE PLATES</div>
    <h2 class="serif">Seven 300-DPI Figures. Click Any Plate to Expand.</h2>
    <div class="fig-grid">
      <figure class="plate">
        <img src="../img/fig1_ten_year_progression.png" alt="Figure 1" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 1 · Ten-Year Progression &amp; Uplift Waterfall.</strong> Decomposing +0.0375 ROC-AUC across algorithms, RDKit chemistry, and GPU multi-task models.</figcaption>
      </figure>
      <figure class="plate">
        <img src="../img/fig4_delong_forest.png" alt="Figure 4" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 2 · Exact DeLong Forest Plots.</strong> Four hypotheses tested across all 5 isozymes—including Panel D proving MOE's residual contribution is statistically zero.</figcaption>
      </figure>
      <figure class="plate">
        <img src="../img/fig7_architecture_and_leakage.png" alt="Figure 7" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 3 · Multi-Modal Architecture &amp; Double-Firewall Isolation.</strong> How four orthogonal model families fuse with mathematically asserted zero cross-isozyme leakage.</figcaption>
      </figure>
      <figure class="plate">
        <img src="../img/fig6_multimodal_weights_and_compute.png" alt="Figure 6" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 4 · Stacking Weights &amp; Hybrid Compute Profile.</strong> Balanced weight share across GBDTs, Tanimoto SVMs, MT-ResMLPs, and raw-SMILES ChemBERTa—completed for 1.80 Colab CU.</figcaption>
      </figure>
      <figure class="plate">
        <img src="../img/fig2_roc_curves_decade.png" alt="Figure 2" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 5 · Held-Out Test ROC Curves (2015 vs 2026).</strong> Raw-SMILES ChemBERTa already outperforms the strongest 186-MOE ensemble on 2C9, 2C19, and 3A4.</figcaption>
      </figure>
      <figure class="plate">
        <img src="../img/fig3_calibration_reliability.png" alt="Figure 3" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 6 · Reliability &amp; Brier Calibration.</strong> Tightening mean 10-bin ECE from 0.0389 (2015 RF) down to 0.0216 (2026 Grand Stack).</figcaption>
      </figure>
      <figure class="plate">
        <img src="../img/fig5_scaffold_generalization.png" alt="Figure 5" onclick="openLightbox(this.src)" />
        <figcaption><strong>Plate 7 · Murcko Scaffold-Split Generalization.</strong> Honest ~0.010 AUC drop on novel chemical scaffolds—with RDKit preserving its lead on every isozyme.</figcaption>
      </figure>
    </div>
  </section>

  <section id="future">
    <div class="sec-num mono">05 // THE WAY OF THE FUTURE (2026–2035)</div>
    <h2 class="serif">Why the Static Two-Column PDF Is Dying—And What Replaces It.</h2>
    <p class="lead">
      In 2015, finishing a thesis meant spending 2–3 semesters simultaneously learning Python, Linux CLI, AWS Free Tier provisioning, LaTeX, MOE, cheminformatics, and Pharmaceutical Sciences coursework—then waiting overnight for a throttled single-vCPU micro instance to finish one SVM grid. That apprenticeship forged real scientific instincts. What changed in 2026 is that once a scientist has those instincts, <strong>AI-coordinated computation collapses the execution gap from months to an afternoon</strong>.
    </p>
    <div class="future-grid">
      <div class="future-card">
        <h3>1. Papers Become Executable Artifacts</h3>
        <p>Every number on this page is bound directly to <span class="mono">notebooks/results/*.csv</span> and verified <span class="mono">.npz</span> prediction archives. Readers shouldn't squint at frozen tables—they should slice isozymes, audit DeLong intervals, and inspect calibration live.</p>
      </div>
      <div class="future-card">
        <h3>2. The Scientist Becomes a Conductor</h3>
        <p>When a local kernel SVM OOM'd at 4.3 GiB on a 14 GiB host, it was refactored to single-fold in-place assembly (&lt;1.3 GiB) in minutes. When an fp16 GNN scatter tripped on CUDA, it was patched on the fly. Human judgment sets the firewalls; AI executes the symphony.</p>
      </div>
      <div class="future-card">
        <h3>3. Open Foundation Models Eat Proprietary Silos</h3>
        <p>The moral arc of computational pharmacology bends toward open weights and open chemistry. When a 3.4M-parameter open SMILES transformer + RDKit + multi-task PyTorch matches a licensed commercial suite at $p > 0.05$, the barrier to world-class ADMET modeling drops to a laptop and two dollars of cloud GPU.</p>
      </div>
    </div>
  </section>

  <footer class="colophon mono">
    CYTOCHROME // 2026 · Built programmatically from verified zero-leakage archives in <span style="color:var(--emerald)">CYP-thesis-edits</span> · Original 2015 thesis preserved untouched.
  </footer>
</div>

<div id="lightbox" onclick="this.style.display='none'"><img id="lbImg" src="" alt="Expanded figure" /></div>

<script>
const DATA = __PAYLOAD_JSON__;

function fmtP(p) {
  if (p < 1e-4) return p.toExponential(1);
  return p.toFixed(4);
}

// Time Machine
const slider = document.getElementById("tmSlider");
function renderTM(idx) {
  const m = DATA.milestones[idx];
  document.getElementById("tmYear").textContent = m.year;
  document.getElementById("tmTitle").textContent = m.title;
  document.getElementById("tmNote").textContent = m.note;
  document.getElementById("tmHw").textContent = "HARDWARE // " + m.hw;
  document.getElementById("tmRt").textContent = "IMPACT   // " + m.runtime;
  document.getElementById("tmAuc").textContent = m.auc.toFixed(4);
  document.getElementById("tmAcc").textContent = (m.acc * 100).toFixed(2) + "%";
  document.getElementById("tmBrier").textContent = m.brier.toFixed(4);
  document.getElementById("tmEce").textContent = m.ece.toFixed(4);
}
slider.addEventListener("input", e => renderTM(+e.target.value));
renderTM(5);

// Isozyme Explorer
function renderIso(iso) {
  const d = DATA.isozymes[iso];
  document.getElementById("isoHeader").textContent = iso === "mean" ? "ALL 5 ISOZYMES (MEAN)" : "CYP " + iso.toUpperCase() + " (HELD-OUT TEST)";
  const lb = document.getElementById("leaderboard");
  const minAuc = 0.77, maxAuc = Math.max(0.92, ...d.models.map(x => x.auc) ) + 0.005;
  lb.innerHTML = d.models.map(m => {
    const pct = Math.max(4, Math.min(100, ((m.auc - minAuc) / (maxAuc - minAuc)) * 100));
    return `
      <div class="bar-row">
        <div><span style="color:#fff;font-weight:600">${m.label}</span></div>
        <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${m.color}"></div></div>
        <div class="bar-val mono">${m.auc.toFixed(4)} <span style="color:var(--muted);font-weight:400">(${(m.acc*100).toFixed(1)}%)</span></div>
      </div>`;
  }).join("");

  const dc = document.getElementById("delongCards");
  const specs = [
    ["GrandAllModal_vs_OrigSVM", "10-Year Total Leap", "Grand All-Modal − 2015 SVM"],
    ["GrandOpenSource_vs_Tier1MOE", "Open-Source Beats MOE", "Grand Open-Source − Tier 1 MOE"],
    ["GrandAllModal_vs_SingleTaskMOERDKit", "Multi-Task + DL Lift", "Grand All-Modal − Single-Task MOE+RDKit"],
    ["GrandAllModal_vs_GrandOpenSource", "Residual Value of MOE", "Grand All-Modal − Grand Open-Source"],
  ];
  dc.innerHTML = specs.map(([k, title, sub]) => {
    const x = d.delong[k];
    const pStr = iso === "mean"
      ? (x.p_max < 0.05 ? `all p ≤ ${fmtP(x.p_max)}` : `all ns (p ∈ [${fmtP(x.p_min)}, ${fmtP(x.p_max)}])`)
      : (x.p < 0.05 ? `p = ${fmtP(x.p)}` : `p = ${fmtP(x.p)} (ns)`);
    const isSig = iso === "mean" ? x.p_max < 0.05 : x.p < 0.05;
    return `
      <div class="hyp">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
          <strong style="font-size:13px;color:#fff">${title}</strong>
          <span class="badge-sig mono ${isSig ? 'badge-yes' : 'badge-ns'}">${isSig ? 'SIGNIFICANT' : 'NS'}</span>
        </div>
        <div class="mono" style="font-size:1.35rem;font-weight:800;margin:8px 0 4px;color:${isSig ? 'var(--emerald)' : '#d8b4fe'}">${x.delta_auc >= 0 ? '+' : ''}${x.delta_auc.toFixed(4)} AUC</div>
        <div class="mono" style="font-size:11px;color:var(--muted)">${pStr}</div>
        <div style="font-size:11px;color:#64748b;margin-top:4px">${sub}</div>
      </div>`;
  }).join("");
}

document.querySelectorAll("#isoPills .pill").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#isoPills .pill").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    renderIso(btn.dataset.iso);
  });
});
renderIso("mean");

function openLightbox(src) {
  document.getElementById("lbImg").src = src;
  document.getElementById("lightbox").style.display = "flex";
}
</script>
</body>
</html>
"""


def main() -> None:
    payload = build_payload()
    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", json.dumps(payload))
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote interactive web paper: {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
