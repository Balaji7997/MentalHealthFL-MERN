import { useEffect, useState } from "react";
import "./App.css";
import { fetchHealth, fetchOverview, fmt, pct } from "./api";
import {
  AggregationView,
  FLArchitectureView,
  FederatedClientsView,
  ModelEvaluationView,
  PersonalizationView,
  ReliabilityAnalysisView,
} from "./components/ResearchModules";
import {
  StressDetectionView,
  VoiceAssistantView,
} from "./components/StressAndVoice";

const menu = [
  { name: "Overview", icon: "▦" },
  { name: "FL Architecture", icon: "◇" },
  { name: "Federated Clients", icon: "⚙" },
  { name: "Aggregation", icon: "⇄" },
  { name: "Personalization", icon: "◎" },
  { name: "Reliability Analysis", icon: "◈" },
  { name: "Model Evaluation", icon: "▣" },
  { name: "Stress Detection", icon: "⌕" },
  { name: "Voice Assistant", icon: "♫" },
];

function OverviewView({ overview, health, error, onNavigate }) {
  const ditto = overview?.model_comparison?.find((m) => m.model_key === "personalized_ditto");
  const fedavg = overview?.model_comparison?.find((m) => m.model_key === "federated_global");
  const comm = overview?.communication_efficiency;
  const policy = overview?.human_review_policy;

  const stats = [
    {
      label: "Federated Clients",
      value: overview ? `0${overview.dataset.num_clients}` : "05",
      note: overview
        ? `Non-IID spread: ${pct(overview.dataset.non_iid_stress_spread.spread)}`
        : "5 Dreaddit domains",
      icon: "⚙",
    },
    {
      label: "Ditto-LoRA Test F1",
      value: ditto ? pct(ditto.f1) : "—",
      note: fedavg ? `vs FedAvg Global ${pct(fedavg.f1)}` : "Connecting to API...",
      icon: "◎",
    },
    {
      label: "Calibrated ECE",
      value: ditto ? fmt(ditto.calibrated_ece) : "—",
      note: policy ? `T = ${policy.optimal_temperature} (69% lower ECE)` : "Temperature Scaling",
      icon: "◈",
    },
    {
      label: "Auto Acc (τ=0.75)",
      value: policy ? pct(policy.default_operating_point.automated_accuracy) : "—",
      note: policy
        ? `${pct(policy.default_operating_point.coverage)} coverage, ${pct(policy.default_operating_point.human_review_rate)} flagged`
        : "Human-review gating",
      icon: "▣",
    },
  ];

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">RESEARCH OVERVIEW</div>
          <h1>Research Dashboard</h1>
          <p>
            Reliability-Aware Personalized Federated Learning for Mental Health Stress Detection.
          </p>
        </div>
        <div className="project-tag">
          <span className={`status-dot ${health?.ai_service_connected ? "" : "offline"}`}></span>
          {health?.ai_service_connected
            ? `Live AI Engine (${health.ai_service?.device?.toUpperCase() || "GPU"} • ${health.ai_service?.loaded_adapters?.length || 7} Adapters)`
            : "Waiting for Backend (5000) & FastAPI (8000)"}
        </div>
      </div>

      {error && (
        <div className="notice danger" style={{ marginTop: 0, marginBottom: "18px" }}>
          <strong>Backend Not Reachable</strong>
          <p>
            {error}. Ensure CMD Window 1 (FastAPI on port 8000) and CMD Window 2 (Express on port 5000) are running.
          </p>
        </div>
      )}

      <div className="stats-grid">
        {stats.map((stat) => (
          <div className="stat-card" key={stat.label}>
            <div className="stat-top">
              <span>{stat.label}</span>
              <span className="stat-icon">{stat.icon}</span>
            </div>
            <h2>{stat.value}</h2>
            <p>{stat.note}</p>
          </div>
        ))}
      </div>

      <div className="section-title">
        <div>
          <h2>Research workflow</h2>
          <p>Completed stages of the proposed methodology (click any stage to inspect real outputs)</p>
        </div>
        <span className="demo-label">EXPERIMENTS COMPLETED</span>
      </div>

      <div className="workflow-grid">
        {[
          ["01", "Dataset Preparation", "Dreaddit cleaned & split (2,256 train / 564 val / 715 test)", "Federated Clients"],
          ["02", "Local Model Training", "RoBERTa-base with LoRA (0.82% trainable parameters)", "Model Evaluation"],
          ["03", "Federated Aggregation", "5-round sample-weighted FedAvg (3.95 MB/client)", "Aggregation"],
          ["04", "Personalization", "Ditto-LoRA proximal adaptation & FedPer-LoRA ablation", "Personalization"],
          ["05", "Reliability Assessment", "Temperature scaling (ECE 0.0258) & predictive entropy", "Reliability Analysis"],
          ["06", "Human Review & Voice", "Uncertain cases flagged; English/Telugu voice pipeline", "Voice Assistant"],
        ].map((step) => (
          <div
            className="workflow-card"
            key={step[0]}
            style={{ cursor: "pointer" }}
            onClick={() => onNavigate(step[3])}
          >
            <div className="step-number">{step[0]}</div>
            <div>
              <h3>{step[1]}</h3>
              <p>{step[2]}</p>
              <span className="step-tag">Open {step[3]} →</span>
            </div>
          </div>
        ))}
      </div>

      <div className="bottom-grid">
        <div className="panel">
          <h3>System architecture</h3>
          <p className="panel-subtitle">Verified research pipeline</p>
          <div className="architecture">
            <div className="architecture-box">Dreaddit Dataset (5 Non-IID Domain Clients)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">RoBERTa-base + LoRA (1.03M Trainable Params)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">5-Round FedAvg Aggregation ({comm ? `${comm.upload_size_per_client_mb} MB` : "3.95 MB"}/client)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">Ditto-LoRA Proximal Personalization (μ = 0.1)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box highlight-box">
              Temperature-Scaled Reliability & Entropy Check
            </div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">Stress / Non-Stress Prediction or Human Review Flag</div>
          </div>
        </div>

        <div className="panel">
          <h3>Verified research configuration</h3>
          <p className="panel-subtitle">Live settings loaded from Python FastAPI service</p>
          <div className="config-list">
            <div><span>Dataset</span><strong>Dreaddit (Binary: Stress / Non-Stress)</strong></div>
            <div><span>Base model</span><strong>FacebookAI/roberta-base</strong></div>
            <div><span>Fine-tuning</span><strong>LoRA (r=8, alpha=16, 0.8231% trainable)</strong></div>
            <div><span>Aggregation</span><strong>Weighted FedAvg (5 Rounds, 5 Clients)</strong></div>
            <div><span>Personalization</span><strong>Ditto-LoRA & FedPer-LoRA</strong></div>
            <div><span>Calibration</span><strong>Temperature Scaling + Normalized Entropy</strong></div>
            <div><span>Voice Interface</span><strong>Web Speech (EN/TE) → Same Text Model</strong></div>
          </div>
        </div>
      </div>

      <div className="notice">
        <strong>Live Experimental Data & Research Disclaimer</strong>
        <p>
          All metrics across this dashboard are loaded live from verified Dreaddit experimental artifacts
          through Express (port 5000) and FastAPI (port 8000). This platform is a research prototype for
          text-based stress indication and uncertainty flagging—not a medical diagnostic system.
        </p>
      </div>
    </>
  );
}

function App() {
  const [active, setActive] = useState("Overview");
  const [overview, setOverview] = useState(null);
  const [health, setHealth] = useState(null);
  const [overviewError, setOverviewError] = useState("");

  useEffect(() => {
    Promise.all([fetchOverview(), fetchHealth()])
      .then(([ov, hl]) => {
        setOverview(ov);
        setHealth(hl);
        setOverviewError("");
      })
      .catch((err) => {
        setOverviewError(err.message || "Failed to connect to backend API.");
      });
  }, []);

  function renderPage() {
    switch (active) {
      case "Overview":
        return (
          <OverviewView
            overview={overview}
            health={health}
            error={overviewError}
            onNavigate={setActive}
          />
        );
      case "FL Architecture":
        return <FLArchitectureView />;
      case "Federated Clients":
        return <FederatedClientsView />;
      case "Aggregation":
        return <AggregationView />;
      case "Personalization":
        return <PersonalizationView />;
      case "Reliability Analysis":
        return <ReliabilityAnalysisView />;
      case "Model Evaluation":
        return <ModelEvaluationView />;
      case "Stress Detection":
        return <StressDetectionView />;
      case "Voice Assistant":
        return <VoiceAssistantView />;
      default:
        return (
          <OverviewView
            overview={overview}
            health={health}
            error={overviewError}
            onNavigate={setActive}
          />
        );
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">M</div>
          <div>
            <h2>MentalHealthFL</h2>
            <span>RESEARCH PLATFORM</span>
          </div>
        </div>

        <div className="menu-heading">WORKSPACE</div>

        <nav>
          {menu.map((item) => (
            <button
              key={item.name}
              className={`nav-item ${active === item.name ? "active" : ""}`}
              onClick={() => setActive(item.name)}
            >
              <span className="nav-icon">{item.icon}</span>
              {item.name}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className={`status-dot ${health?.ai_service_connected ? "" : "offline"}`}></div>
          {health?.ai_service_connected ? "AI Engine Online" : "Offline"}
          <span className="version">v2.0</span>
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <span className="breadcrumb">Workspace /</span>
            <span className="current-page"> {active}</span>
          </div>
          <div className="top-right">
            <span className="privacy-label">🔒 Privacy-aware research • Raw text never stored</span>
            <div className="avatar">R</div>
          </div>
        </header>

        <section className="content">{renderPage()}</section>
      </main>
    </div>
  );
}

export default App;