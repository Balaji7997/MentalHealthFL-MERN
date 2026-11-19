import { useEffect, useState } from "react";
import {
  fetchAggregation,
  fetchClients,
  fetchEvaluation,
  fetchOverview,
  fetchPersonalization,
  fetchReliability,
  fmt,
  pct,
} from "../api";

function useApiData(fetcher) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    fetcher()
      .then((res) => {
        if (active) setData(res);
      })
      .catch((err) => {
        if (active) setError(err.message || "Failed to load data from backend.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [fetcher]);

  return { data, loading, error };
}

function StatusBlock({ loading, error, title }) {
  if (loading) {
    return <div className="loading-box">Loading real experimental data for {title}...</div>;
  }
  if (error) {
    return (
      <div className="notice danger">
        <strong>Backend Connection Error</strong>
        <p>
          {error}. Make sure FastAPI (port 8000) and Express Backend (port 5000) are both running.
        </p>
      </div>
    );
  }
  return null;
}

function DeltaText({ value, asPercent = true }) {
  const num = Number(value || 0);
  const cls = num > 0 ? "delta-pos" : num < 0 ? "delta-neg" : "delta-zero";
  const formatted = asPercent
    ? `${num >= 0 ? "+" : ""}${(num * 100).toFixed(2)}%`
    : `${num >= 0 ? "+" : ""}${num.toFixed(4)}`;
  return <span className={cls}>{formatted}</span>;
}

/* ============================================================================
   2. FL ARCHITECTURE
   ============================================================================ */
export function FLArchitectureView() {
  const { data, loading, error } = useApiData(fetchOverview);
  if (loading || error) return <StatusBlock loading={loading} error={error} title="FL Architecture" />;

  const arch = data.base_architecture;
  const comm = data.communication_efficiency;
  const policy = data.human_review_policy;

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">SYSTEM DESIGN</div>
          <h1>FL Architecture</h1>
          <p>
            End-to-end modular architecture combining RoBERTa + LoRA, Non-IID FedAvg,
            Ditto-LoRA personalization, and Temperature-Scaled reliability gating.
          </p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          3-Tier Modular Stack
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-top"><span>Base Backbone</span><span className="stat-icon">◇</span></div>
          <h2>RoBERTa</h2>
          <p>{arch.base_model} ({(arch.total_parameters / 1e6).toFixed(1)}M total)</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>LoRA Trainable</span><span className="stat-icon">⚙</span></div>
          <h2>{arch.trainable_percentage}%</h2>
          <p>{arch.trainable_parameters.toLocaleString()} params (r={arch.lora_r}, alpha={arch.lora_alpha})</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Upload / Client</span><span className="stat-icon">⇄</span></div>
          <h2>{comm.upload_size_per_client_mb} MB</h2>
          <p>{comm.shared_tensors_per_round} trainable tensors per FedAvg round</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Review Gate</span><span className="stat-icon">◈</span></div>
          <h2>T = {policy.optimal_temperature}</h2>
          <p>Flag human review below {pct(policy.default_confidence_threshold, 0)} conf</p>
        </div>
      </div>

      <div className="bottom-grid">
        <div className="panel">
          <h3>Research Methodology Pipeline</h3>
          <p className="panel-subtitle">Exact execution flow across training and inference</p>
          <div className="architecture">
            <div className="architecture-box">1. Dreaddit Corpus ({data.dataset.train_rows} Train / {data.dataset.validation_rows} Val / {data.dataset.test_rows} Test)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">2. Domain Non-IID Partition ({data.dataset.num_clients} Private Clients — Raw Text Stays Local)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">3. Local Fine-Tuning: Frozen RoBERTa-base + LoRA ({arch.lora_target_modules.join(", ")})</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">4. Server Aggregation: Sample-Weighted FedAvg ({comm.upload_size_per_client_mb} MB/client)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box highlight-box">5. Client Personalization: Ditto-LoRA Proximal Regularization (μ = 0.1)</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box highlight-box">6. Reliability Calibration: Temperature Scaling + Normalized Entropy</div>
            <div className="down-arrow">↓</div>
            <div className="architecture-box">7. Binary Output (Stress / Non-Stress) or Flag for Human Review</div>
          </div>
        </div>

        <div className="panel">
          <h3>MERN + FastAPI Service Architecture</h3>
          <p className="panel-subtitle">Decoupled application tiers and voice pipeline</p>
          <div className="config-list">
            <div><span>Frontend Dashboard</span><strong>React + Vite + Tailwind (Port 5173)</strong></div>
            <div><span>API Gateway & Audit</span><strong>Node.js + Express.js (Port 5000)</strong></div>
            <div><span>AI/ML Inference Engine</span><strong>Python FastAPI + PyTorch + PEFT (Port 8000)</strong></div>
            <div><span>Multi-Adapter Serving</span><strong>1 RoBERTa Backbone + 7 Hot-Swappable LoRA Adapters</strong></div>
            <div><span>Voice Pipeline</span><strong>Mic → Web Speech STT → Same Text Model → TTS</strong></div>
            <div><span>Database Status</span><strong>MongoDB Optional (Privacy-Safe Metadata Only)</strong></div>
            <div><span>Raw Text Persistence</span><strong>Disabled by Design (Never Stored)</strong></div>
          </div>
        </div>
      </div>

      <div className="notice">
        <strong>Privacy & Research Scope Clarification</strong>
        {data.privacy_and_scope_notes.map((note, idx) => (
          <p key={idx}>• {note}</p>
        ))}
      </div>
    </>
  );
}

/* ============================================================================
   3. FEDERATED CLIENTS
   ============================================================================ */
export function FederatedClientsView() {
  const { data, loading, error } = useApiData(fetchClients);
  if (loading || error) return <StatusBlock loading={loading} error={error} title="Federated Clients" />;

  const spread = data.train_stress_ratio_range;

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">NON-IID DATA DISTRIBUTION</div>
          <h1>Federated Clients</h1>
          <p>{data.partition_strategy} across {data.num_clients} isolated mental-health communities.</p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          Stress Skew Spread: {pct(spread.spread)}
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-top"><span>Total Clients</span><span className="stat-icon">⚙</span></div>
          <h2>0{data.num_clients}</h2>
          <p>Isolated Dreaddit domains</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Lowest Stress Client</span><span className="stat-icon">↓</span></div>
          <h2>{pct(spread.min)}</h2>
          <p>client_3 (Financial & Housing)</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Highest Stress Client</span><span className="stat-icon">↑</span></div>
          <h2>{pct(spread.max)}</h2>
          <p>client_2 (Anxiety & Stress)</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Label-Prevalence Skew</span><span className="stat-icon">⇄</span></div>
          <h2>{pct(spread.spread)}</h2>
          <p>Realistic Non-IID heterogeneity</p>
        </div>
      </div>

      <div className="panel">
        <h3>Client-Wise Non-IID Stress Prevalence (Training Split)</h3>
        <p className="panel-subtitle">Each client experiences a distinct community vocabulary and stress base rate</p>
        {data.clients_partition.map((c) => (
          <div className="bar-row" key={c.client_id}>
            <div className="bar-meta">
              <strong>{c.client_id} — {c.display_name}</strong>
              <span>
                Train: {c.train.rows} | Stress: {c.train.stress} ({pct(c.train.stress_ratio)}) | Non-Stress: {c.train.non_stress}
              </span>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${c.train.stress_ratio > 0.6 ? "red" : c.train.stress_ratio < 0.45 ? "navy" : ""}`}
                style={{ width: `${(c.train.stress_ratio * 100).toFixed(1)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="panel">
        <h3>Client Partition & Test Performance Breakdown</h3>
        <p className="panel-subtitle">Exact sample counts, source subreddits, and test metrics per federated client</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Client</th>
                <th>Domain</th>
                <th>Subreddits</th>
                <th>Train / Val / Test</th>
                <th>FedAvg F1</th>
                <th>Ditto-LoRA F1</th>
                <th>Gain</th>
                <th>Flagged %</th>
              </tr>
            </thead>
            <tbody>
              {data.client_comparison.map((c) => {
                const part = data.clients_partition.find((p) => p.client_id === c.client_id);
                return (
                  <tr key={c.client_id}>
                    <td><strong>{c.client_id}</strong></td>
                    <td><span className="badge badge-teal">{c.domain}</span></td>
                    <td>{c.subreddits.join(", ")}</td>
                    <td>{part?.train.rows} / {part?.validation.rows} / {part?.test.rows}</td>
                    <td>{pct(c.fedavg_f1)}</td>
                    <td><strong>{pct(c.ditto_f1)}</strong></td>
                    <td><DeltaText value={c.ditto_vs_fedavg_f1_delta} /></td>
                    <td>{pct(c.ditto_human_review_rate, 1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ============================================================================
   4. AGGREGATION (FEDAVG)
   ============================================================================ */
export function AggregationView() {
  const { data, loading, error } = useApiData(fetchAggregation);
  if (loading || error) return <StatusBlock loading={loading} error={error} title="Aggregation" />;

  const comm = data.communication_efficiency;
  const hp = data.hyperparameters;

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">FEDERATED AVERAGING</div>
          <h1>Aggregation (FedAvg)</h1>
          <p>{data.aggregation_algorithm}</p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          Best Global Round: {data.best_round} / {hp.num_rounds}
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-top"><span>Completed Rounds</span><span className="stat-icon">⇄</span></div>
          <h2>0{hp.num_rounds}</h2>
          <p>{hp.local_epochs} local epoch/round, lr={hp.learning_rate}</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Payload / Client</span><span className="stat-icon">◇</span></div>
          <h2>{comm.upload_size_per_client_mb} MB</h2>
          <p>vs ~480 MB full RoBERTa (~99% saved)</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Global Test Acc</span><span className="stat-icon">▣</span></div>
          <h2>{pct(data.final_global_test_metrics.accuracy)}</h2>
          <p>Untouched Dreaddit test set (n=715)</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Global Test F1</span><span className="stat-icon">◈</span></div>
          <h2>{pct(data.final_global_test_metrics.f1)}</h2>
          <p>Precision {pct(data.final_global_test_metrics.precision)} | Recall {pct(data.final_global_test_metrics.recall)}</p>
        </div>
      </div>

      <div className="panel">
        <h3>Round-by-Round Global Convergence (Validation Split)</h3>
        <p className="panel-subtitle">Aggregated FedAvg global model F1-score and loss after each communication round</p>
        {data.rounds_history.map((r) => (
          <div className="bar-row" key={r.round}>
            <div className="bar-meta">
              <strong>
                Round {r.round} {r.round === data.best_round ? " (Best Global Checkpoint)" : ""}
              </strong>
              <span>
                Val Loss: {fmt(r.global_val_metrics.loss)} | Val Acc: {pct(r.global_val_metrics.accuracy)} |{" "}
                <strong>Val F1: {pct(r.global_val_metrics.f1)}</strong> ({r.round_seconds}s)
              </span>
            </div>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${(r.global_val_metrics.f1 * 100).toFixed(1)}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="panel">
        <h3>Per-Round Client Participation & Sample Weights</h3>
        <p className="panel-subtitle">Weighted FedAvg formula: w_global = Σ (n_k / N) · w_k</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Round</th>
                <th>client_1 (abuse)</th>
                <th>client_2 (anxiety)</th>
                <th>client_3 (financial)</th>
                <th>client_4 (ptsd)</th>
                <th>client_5 (social)</th>
                <th>Global Val F1</th>
                <th>Global Val Loss</th>
              </tr>
            </thead>
            <tbody>
              {data.rounds_history.map((r) => (
                <tr key={r.round} className={r.round === data.best_round ? "row-highlight" : ""}>
                  <td><strong>Round {r.round}</strong></td>
                  {r.clients.map((c) => (
                    <td key={c.client_id}>
                      F1 {pct(c.local_val_f1, 1)}
                      <br />
                      <span style={{ color: "#7f8d9e", fontSize: "10px" }}>
                        Loss {fmt(c.local_train_loss, 3)} (w={r.aggregation_weights[c.client_id]})
                      </span>
                    </td>
                  ))}
                  <td><strong>{pct(r.global_val_metrics.f1)}</strong></td>
                  <td>{fmt(r.global_val_metrics.loss)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

/* ============================================================================
   5. PERSONALIZATION (DITTO-LORA vs FEDPER-LORA vs FEDAVG)
   ============================================================================ */
export function PersonalizationView() {
  const { data, loading, error } = useApiData(fetchPersonalization);
  if (loading || error) return <StatusBlock loading={loading} error={error} title="Personalization" />;

  const ditto = data.ditto_lora;
  const fedper = data.fedper_lora;

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">CLIENT-SPECIFIC ADAPTATION</div>
          <h1>Personalized Federated Learning</h1>
          <p>
            Empirical comparison of Ditto-LoRA (Proximal Regularized Adaptation) and FedPer-LoRA
            (Shared LoRA + Private Classifier Heads) against the FedAvg Global model.
          </p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          Recommended: {data.recommended_method}
        </div>
      </div>

      <div className="two-col-grid">
        <div className="panel">
          <h3>Ditto-LoRA (Selected Primary Method)</h3>
          <p className="panel-subtitle">{ditto.method_name}</p>
          <div className="config-list">
            <div><span>Initialization</span><strong>Converged FedAvg Global LoRA</strong></div>
            <div><span>Regularization</span><strong>Proximal L2 Penalty (μ = {ditto.hyperparameters.proximal_mu})</strong></div>
            <div><span>Local Adaptation</span><strong>Up to {ditto.hyperparameters.adapt_epochs} epochs (lr = {ditto.hyperparameters.adapt_lr}), Val-Selected</strong></div>
            <div><span>Overall Test Accuracy</span><strong>{pct(ditto.overall_personalized_test_metrics.accuracy)}</strong></div>
            <div><span>Overall Test F1-Score</span><strong>{pct(ditto.overall_personalized_test_metrics.f1)}</strong></div>
            <div><span>Mean Client Test F1</span><strong>{pct(ditto.mean_client_test_metrics.f1)}</strong></div>
          </div>
        </div>

        <div className="panel">
          <h3>FedPer-LoRA (Bandwidth-Efficient Ablation)</h3>
          <p className="panel-subtitle">{fedper.method_name}</p>
          <div className="config-list">
            <div><span>Shared (Aggregated)</span><strong>{fedper.parameter_split.shared_parameters.toLocaleString()} LoRA params ({fedper.parameter_split.shared_upload_mb_per_client_round} MB)</strong></div>
            <div><span>Private (Local Only)</span><strong>{fedper.parameter_split.personalized_parameters.toLocaleString()} Head params ({fedper.parameter_split.personalized_local_mb_per_client} MB)</strong></div>
            <div><span>Upload Savings vs FedAvg</span><strong>{fedper.parameter_split.communication_reduction_vs_fedavg_percent}% Bandwidth Reduction</strong></div>
            <div><span>Overall Test Accuracy</span><strong>{pct(fedper.overall_personalized_test_metrics.accuracy)}</strong></div>
            <div><span>Overall Test F1-Score</span><strong>{pct(fedper.overall_personalized_test_metrics.f1)}</strong></div>
            <div><span>Mean Client Test F1</span><strong>{pct(fedper.mean_client_test_metrics.f1)}</strong></div>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Client-Wise Test Comparison: FedAvg Global vs. FedPer-LoRA vs. Ditto-LoRA</h3>
        <p className="panel-subtitle">Evaluated strictly on each client's untouched local Dreaddit test set</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Client</th>
                <th>Domain</th>
                <th>Test n</th>
                <th>Ditto Best Ep</th>
                <th>FedAvg Acc / F1</th>
                <th>FedPer Acc / F1</th>
                <th>Ditto Acc / F1</th>
                <th>Ditto Acc Δ</th>
                <th>Ditto F1 Δ</th>
              </tr>
            </thead>
            <tbody>
              {data.client_comparison.map((c) => {
                const dittoRow = ditto.comparison_vs_fedavg_global.find((x) => x.client_id === c.client_id);
                return (
                  <tr key={c.client_id} className={c.ditto_vs_fedavg_f1_delta > 0 ? "row-highlight" : ""}>
                    <td><strong>{c.client_id}</strong></td>
                    <td><span className="badge badge-teal">{c.domain}</span></td>
                    <td>{c.test_samples}</td>
                    <td>Epoch {dittoRow?.best_adapt_epoch ?? 0}</td>
                    <td>{pct(c.fedavg_accuracy)} / {pct(c.fedavg_f1)}</td>
                    <td>{pct(c.fedper_accuracy)} / {pct(c.fedper_f1)}</td>
                    <td><strong>{pct(c.ditto_accuracy)} / {pct(c.ditto_f1)}</strong></td>
                    <td><DeltaText value={c.ditto_vs_fedavg_acc_delta} /></td>
                    <td><DeltaText value={c.ditto_vs_fedavg_f1_delta} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div className="notice">
        <strong>Research Finding (Ablation Insight)</strong>
        <p>
          Ditto-LoRA outperforms both FedAvg Global and FedPer-LoRA overall because starting from the converged
          FedAvg model with proximal regularization prevents overfitting on small (~435 sample) NLP clients while
          still allowing strong local adaptation (e.g., +2.83% F1 on abuse/client_1). Meanwhile, FedPer-LoRA cuts
          upload bandwidth by {fedper.parameter_split.communication_reduction_vs_fedavg_percent}% and excels on
          high-stress domains (anxiety and PTSD).
        </p>
      </div>
    </>
  );
}

/* ============================================================================
   6. RELIABILITY ANALYSIS
   ============================================================================ */
export function ReliabilityAnalysisView() {
  const { data, loading, error } = useApiData(fetchReliability);
  const [selectedModel, setSelectedModel] = useState("personalized_ditto");

  if (loading || error) return <StatusBlock loading={loading} error={error} title="Reliability Analysis" />;

  const modelKeys = Object.keys(data.models);
  const m = data.models[selectedModel] || data.models.personalized_ditto;
  const sep = m.uncertainty_separation;
  const defPol = m.default_human_review_policy;

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">CALIBRATION & UNCERTAINTY</div>
          <h1>Reliability Analysis</h1>
          <p>{data.calibration_method} paired with {data.uncertainty_metric}.</p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          Optimal T = {m.optimal_temperature}
        </div>
      </div>

      <div className="btn-row" style={{ marginBottom: "18px", marginTop: 0 }}>
        {modelKeys.map((k) => (
          <button
            key={k}
            className={`btn-secondary ${selectedModel === k ? "active-tab" : ""}`}
            onClick={() => setSelectedModel(k)}
          >
            {data.models[k].display_name}
          </button>
        ))}
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-top"><span>Calibrated ECE</span><span className="stat-icon">◈</span></div>
          <h2>{fmt(m.calibrated.ece)}</h2>
          <p>Uncalibrated: {fmt(m.uncalibrated.ece)} ({m.calibration_improvement.ece_delta})</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Calibrated Brier</span><span className="stat-icon">▣</span></div>
          <h2>{fmt(m.calibrated.brier_score)}</h2>
          <p>Uncalibrated: {fmt(m.uncalibrated.brier_score)} ({m.calibration_improvement.brier_delta})</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Auto Accuracy (τ=0.75)</span><span className="stat-icon">✓</span></div>
          <h2>{pct(defPol.automated_accuracy)}</h2>
          <p>Coverage: {pct(defPol.coverage)} ({defPol.automated_count} auto / {defPol.human_review_count} flagged)</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Flagged Error Rate</span><span className="stat-icon">!</span></div>
          <h2>{pct(defPol.flagged_subset_error_rate)}</h2>
          <p>vs {pct(defPol.automated_error_rate)} error on automated cases</p>
        </div>
      </div>

      <div className="two-col-grid">
        <div className="panel">
          <h3>Uncertainty Separation (Correct vs. Incorrect)</h3>
          <p className="panel-subtitle">Misclassified predictions carry substantially higher predictive entropy</p>
          <div className="bar-row">
            <div className="bar-meta">
              <span>Mean Entropy — Correct Predictions</span>
              <strong>{fmt(sep.mean_entropy_correct)} (Conf {pct(sep.mean_confidence_correct)})</strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(sep.mean_entropy_correct * 100).toFixed(1)}%` }} />
            </div>
          </div>
          <div className="bar-row">
            <div className="bar-meta">
              <span>Mean Entropy — Incorrect Predictions</span>
              <strong>{fmt(sep.mean_entropy_incorrect)} (Conf {pct(sep.mean_confidence_incorrect)})</strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill red" style={{ width: `${(sep.mean_entropy_incorrect * 100).toFixed(1)}%` }} />
            </div>
          </div>
          <div className="config-list" style={{ marginTop: "14px" }}>
            <div><span>Uncalibrated NLL</span><strong>{fmt(m.uncalibrated.nll)}</strong></div>
            <div><span>Calibrated NLL</span><strong>{fmt(m.calibrated.nll)} ({m.calibration_improvement.nll_delta})</strong></div>
            <div><span>Default Confidence Threshold</span><strong>{pct(data.default_confidence_threshold, 0)}</strong></div>
          </div>
        </div>

        <div className="panel">
          <h3>Calibrated Reliability Bins (Non-Empty)</h3>
          <p className="panel-subtitle">Empirical accuracy vs. mean predicted confidence after Temperature Scaling</p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Conf Bin</th>
                  <th>Count</th>
                  <th>Avg Conf</th>
                  <th>Empirical Acc</th>
                  <th>Gap</th>
                </tr>
              </thead>
              <tbody>
                {m.calibrated.reliability_bins
                  .filter((b) => b.count > 0)
                  .map((b) => (
                    <tr key={b.bin_index}>
                      <td>{b.range}</td>
                      <td>{b.count}</td>
                      <td>{pct(b.avg_confidence)}</td>
                      <td><strong>{pct(b.empirical_accuracy)}</strong></td>
                      <td>{fmt(b.calibration_gap)}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div className="panel">
        <h3>Human-Review Coverage vs. Error Trade-Off ({m.display_name})</h3>
        <p className="panel-subtitle">Higher confidence thresholds route uncertain cases to human review and sharply reduce automated errors</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Confidence Threshold</th>
                <th>Automated Coverage</th>
                <th>Flagged for Review</th>
                <th>Automated Accuracy</th>
                <th>Automated Error Rate</th>
                <th>Flagged Subset Error Rate</th>
              </tr>
            </thead>
            <tbody>
              {m.threshold_tradeoff_curve.map((pt) => (
                <tr
                  key={pt.confidence_threshold}
                  className={pt.confidence_threshold === data.default_confidence_threshold ? "row-highlight" : ""}
                >
                  <td>
                    <strong>{pt.confidence_threshold.toFixed(2)}</strong>
                    {pt.confidence_threshold === data.default_confidence_threshold && (
                      <span className="badge badge-teal" style={{ marginLeft: "8px" }}>Default</span>
                    )}
                  </td>
                  <td>{pct(pt.coverage)} ({pt.automated_count})</td>
                  <td>{pct(pt.human_review_rate)} ({pt.human_review_count})</td>
                  <td><strong>{pct(pt.automated_accuracy)}</strong></td>
                  <td>{pct(pt.automated_error_rate)}</td>
                  <td>{pct(pt.flagged_subset_error_rate)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="notice warning">
        <strong>Reliability Disclaimer</strong>
        <p>{data.disclaimer}</p>
      </div>
    </>
  );
}

/* ============================================================================
   7. MODEL EVALUATION
   ============================================================================ */
export function ModelEvaluationView() {
  const { data, loading, error } = useApiData(fetchEvaluation);
  if (loading || error) return <StatusBlock loading={loading} error={error} title="Model Evaluation" />;

  const summary = data.summary;

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">EXPERIMENTAL BENCHMARK</div>
          <h1>Model Evaluation</h1>
          <p>
            Verified test results on the untouched Dreaddit test set (n={summary.dataset.test_rows})
            comparing Centralized, FedAvg Global, FedPer-LoRA, and Ditto-LoRA models.
          </p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          100% Real Experimental Metrics
        </div>
      </div>

      <div className="panel">
        <h3>Unified 4-Model Benchmark Comparison</h3>
        <p className="panel-subtitle">Classification metrics, probability calibration (ECE / Brier), and automated accuracy at τ = 0.75</p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Paradigm</th>
                <th>Data Pooled?</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Cal. ECE</th>
                <th>Cal. Brier</th>
                <th>Auto Acc (τ=0.75)</th>
              </tr>
            </thead>
            <tbody>
              {summary.model_comparison.map((m) => (
                <tr key={m.model_key} className={m.model_key === "personalized_ditto" ? "row-highlight" : ""}>
                  <td><strong>{m.name}</strong></td>
                  <td>{m.paradigm}</td>
                  <td>
                    <span className={`badge ${m.data_centralized ? "badge-amber" : "badge-teal"}`}>
                      {m.data_centralized ? "Yes" : "No (Local)"}
                    </span>
                  </td>
                  <td>{pct(m.accuracy)}</td>
                  <td>{pct(m.precision)}</td>
                  <td>{pct(m.recall)}</td>
                  <td><strong>{pct(m.f1)}</strong></td>
                  <td><strong>{fmt(m.calibrated_ece)}</strong></td>
                  <td>{fmt(m.calibrated_brier)}</td>
                  <td>{pct(m.default_policy_auto_acc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="two-col-grid">
        {summary.model_comparison.map((m) => {
          const [[tn, fp], [fn, tp]] = m.confusion_matrix;
          return (
            <div className="panel" key={m.model_key}>
              <h3>{m.name} — Confusion Matrix</h3>
              <p className="panel-subtitle">
                Accuracy {pct(m.accuracy)} | F1 {pct(m.f1)} | Calibrated ECE {fmt(m.calibrated_ece)}
              </p>
              <div className="cm-grid">
                <div className="cm-cell good">
                  <strong>{tn}</strong>
                  <span>True Non-Stress (TN)</span>
                </div>
                <div className="cm-cell err">
                  <strong>{fp}</strong>
                  <span>False Stress (FP)</span>
                </div>
                <div className="cm-cell err">
                  <strong>{fn}</strong>
                  <span>Missed Stress (FN)</span>
                </div>
                <div className="cm-cell good">
                  <strong>{tp}</strong>
                  <span>True Stress (TP)</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <div className="panel">
        <h3>Dataset Citation & Reproducibility Record</h3>
        <p className="panel-subtitle">Verified dataset provenance for academic reporting</p>
        <div className="config-list">
          <div><span>Dataset Source</span><strong>{summary.dataset.source}</strong></div>
          <div><span>Official Citation</span><strong>{summary.dataset.citation}</strong></div>
          <div><span>Splits (Train / Val / Test)</span><strong>{summary.dataset.train_rows} / {summary.dataset.validation_rows} / {summary.dataset.test_rows}</strong></div>
          <div><span>Base Language Model</span><strong>{summary.base_architecture.base_model}</strong></div>
        </div>
      </div>
    </>
  );
}