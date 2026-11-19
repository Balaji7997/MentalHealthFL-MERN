import { useEffect, useRef, useState } from "react";
import { fetchSessions, fmt, pct, predictStress } from "../api";

const CLIENT_OPTIONS = [
  { id: "client_1", label: "client_1 — Abuse & Domestic Violence" },
  { id: "client_2", label: "client_2 — Anxiety & General Stress" },
  { id: "client_3", label: "client_3 — Financial & Housing Distress" },
  { id: "client_4", label: "client_4 — PTSD & Trauma Support" },
  { id: "client_5", label: "client_5 — Social & Relationships" },
];

const SAMPLE_TEXTS = [
  {
    label: "Clear Stress Example",
    text: "I have barely slept all week, my chest feels tight every morning, and I feel completely overwhelmed by constant panic and deadlines.",
  },
  {
    label: "Clear Non-Stress Example",
    text: "Spent a calm Sunday morning cooking breakfast, taking a walk outside, and catching up with an old college friend over coffee.",
  },
  {
    label: "Borderline / Uncertain Example",
    text: "Work has been unusually busy lately and I have a lot on my plate, though I think I might figure things out next week.",
  },
];

function PredictionResultCard({ result }) {
  if (!result) return null;

  const isStress = result.prediction === "Stress";
  const probStress = result.probabilities?.stress ?? 0;
  const probNonStress = result.probabilities?.non_stress ?? 0;

  return (
    <div className="panel">
      <h3>Prediction & Reliability Assessment</h3>
      <p className="panel-subtitle">
        Model: {result.model_used} | Calibration Temperature T = {result.calibration_temperature} |
        Threshold τ = {result.confidence_threshold.toFixed(2)}
      </p>

      <div className={`result-banner ${isStress ? "stress" : "non-stress"}`}>
        <div>
          <span className="eyebrow">CALIBRATED CLASSIFICATION</span>
          <h2>{result.prediction}</h2>
        </div>
        <div style={{ textAlign: "right" }}>
          <span
            className={`badge ${
              result.needs_human_review
                ? "badge-amber"
                : isStress
                ? "badge-red"
                : "badge-teal"
            }`}
          >
            {result.needs_human_review ? "HUMAN REVIEW REQUIRED" : `Reliability: ${result.reliability_level}`}
          </span>
          <div style={{ fontSize: "12px", color: "#55657a", marginTop: "6px" }}>
            Calibrated Confidence: <strong>{pct(result.confidence)}</strong>
          </div>
        </div>
      </div>

      {result.needs_human_review && (
        <div className="notice warning" style={{ marginTop: 0, marginBottom: "16px" }}>
          <strong>Human-Review Warning (Uncertain Prediction)</strong>
          <p>{result.warning}</p>
        </div>
      )}

      <div className="two-col-grid">
        <div>
          <div className="bar-row">
            <div className="bar-meta">
              <span>P(Stress)</span>
              <strong>{pct(probStress)}</strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill red" style={{ width: `${(probStress * 100).toFixed(1)}%` }} />
            </div>
          </div>

          <div className="bar-row">
            <div className="bar-meta">
              <span>P(Non-Stress)</span>
              <strong>{pct(probNonStress)}</strong>
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(probNonStress * 100).toFixed(1)}%` }} />
            </div>
          </div>
        </div>

        <div>
          <div className="bar-row">
            <div className="bar-meta">
              <span>Reliability Score (1 − Entropy)</span>
              <strong>{pct(result.reliability_score)}</strong>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${result.needs_human_review ? "amber" : ""}`}
                style={{ width: `${(result.reliability_score * 100).toFixed(1)}%` }}
              />
            </div>
          </div>

          <div className="bar-row">
            <div className="bar-meta">
              <span>Normalized Predictive Entropy (0=Certain, 1=Max)</span>
              <strong>{fmt(result.normalized_entropy)}</strong>
            </div>
            <div className="bar-track">
              <div
                className={`bar-fill ${result.normalized_entropy > 0.7 ? "amber" : "navy"}`}
                style={{ width: `${(result.normalized_entropy * 100).toFixed(1)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="notice" style={{ marginTop: "8px" }}>
        <strong>Research Scope Notice</strong>
        <p>{result.disclaimer}</p>
      </div>
    </div>
  );
}

/* ============================================================================
   8. LIVE TEXT STRESS DETECTION
   ============================================================================ */
export function StressDetectionView() {
  const [text, setText] = useState("");
  const [modelType, setModelType] = useState("personalized_ditto");
  const [clientId, setClientId] = useState("client_2");
  const [threshold, setThreshold] = useState(0.75);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [sessions, setSessions] = useState(null);

  const loadSessions = () => {
    fetchSessions()
      .then((res) => setSessions(res))
      .catch(() => {});
  };

  useEffect(() => {
    loadSessions();
  }, []);

  async function handlePredict(e) {
    e.preventDefault();
    if (!text.trim()) {
      setError("Please enter some text before running stress detection.");
      return;
    }

    setLoading(true);
    setError("");
    try {
      const res = await predictStress({
        text: text.trim(),
        model_type: modelType,
        client_id: clientId,
        confidence_threshold: threshold,
        source: "text",
        language: "en",
      });
      setResult(res);
      loadSessions();
    } catch (err) {
      setError(err.message || "Prediction request failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">LIVE CALIBRATED INFERENCE</div>
          <h1>Stress Detection</h1>
          <p>
            Run real-time binary Stress / Non-Stress inference with RoBERTa + LoRA,
            Temperature Scaling, entropy estimation, and human-review flagging.
          </p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          Raw Text Never Stored
        </div>
      </div>

      <form className="panel" onSubmit={handlePredict}>
        <h3>Inference Configuration & Input Text</h3>
        <p className="panel-subtitle">Select model paradigm, personalized client adapter, and confidence threshold</p>

        <div className="form-grid">
          <div className="field">
            <label>Model Variant</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="personalized_ditto">Ditto-LoRA Personalized (Recommended)</option>
              <option value="federated_global">FedAvg Global Model</option>
              <option value="centralized_lora">Centralized Baseline (Upper Bound)</option>
            </select>
          </div>

          <div className="field">
            <label>Personalized Client Adapter</label>
            <select
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              disabled={modelType !== "personalized_ditto"}
            >
              {CLIENT_OPTIONS.map((c) => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>Human-Review Threshold (τ = {Number(threshold).toFixed(2)})</label>
            <input
              type="range"
              min="0.50"
              max="0.95"
              step="0.05"
              value={threshold}
              onChange={(e) => setThreshold(Number(e.target.value))}
            />
          </div>
        </div>

        <div className="field">
          <label>Text Input</label>
          <textarea
            placeholder="Enter a statement or reflection to evaluate for text-based stress indicators..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </div>

        <div className="btn-row">
          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? "Running RoBERTa + LoRA..." : "Analyze Stress & Reliability"}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              setText("");
              setResult(null);
              setError("");
            }}
          >
            Clear
          </button>
          {SAMPLE_TEXTS.map((s) => (
            <button
              type="button"
              key={s.label}
              className="btn-secondary"
              onClick={() => setText(s.text)}
            >
              Load: {s.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="notice danger">
            <strong>Error</strong>
            <p>{error}</p>
          </div>
        )}
      </form>

      <PredictionResultCard result={result} />

      {sessions && (
        <div className="panel">
          <h3>Privacy-Safe Session Audit Metadata</h3>
          <p className="panel-subtitle">{sessions.privacy_note}</p>
          <div className="stats-grid" style={{ marginBottom: "16px" }}>
            <div className="stat-card">
              <div className="stat-top"><span>Session Predictions</span></div>
              <h2>{sessions.totals.predictions}</h2>
              <p>{sessions.totals.voice_requests} via Voice Assistant</p>
            </div>
            <div className="stat-card">
              <div className="stat-top"><span>Stress Predicted</span></div>
              <h2>{sessions.totals.stress_predictions}</h2>
              <p>Metadata counter only</p>
            </div>
            <div className="stat-card">
              <div className="stat-top"><span>Non-Stress Predicted</span></div>
              <h2>{sessions.totals.non_stress_predictions}</h2>
              <p>Metadata counter only</p>
            </div>
            <div className="stat-card">
              <div className="stat-top"><span>Flagged for Review</span></div>
              <h2>{sessions.totals.human_review_flagged}</h2>
              <p>Low-confidence / high-entropy</p>
            </div>
          </div>

          {sessions.recent.length > 0 && (
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Channel</th>
                    <th>Model</th>
                    <th>Words</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Entropy</th>
                    <th>Human Review</th>
                  </tr>
                </thead>
                <tbody>
                  {sessions.recent.slice(0, 8).map((row) => (
                    <tr key={row.id}>
                      <td>{new Date(row.timestamp).toLocaleTimeString()}</td>
                      <td><span className="badge badge-slate">{row.input_channel} ({row.language})</span></td>
                      <td>{row.model_used}</td>
                      <td>{row.input_word_count}</td>
                      <td><strong>{row.prediction}</strong></td>
                      <td>{pct(row.confidence)}</td>
                      <td>{fmt(row.normalized_entropy)}</td>
                      <td>
                        <span className={`badge ${row.needs_human_review ? "badge-amber" : "badge-teal"}`}>
                          {row.needs_human_review ? "Flagged" : "Automated"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </>
  );
}

/* ============================================================================
   9. VOICE ASSISTANT (ENGLISH + TELUGU -> SAME TEXT STRESS MODEL -> TTS)
   ============================================================================ */
export function VoiceAssistantView() {
  const [language, setLanguage] = useState("en-IN");
  const [modelType, setModelType] = useState("personalized_ditto");
  const [clientId, setClientId] = useState("client_2");
  const [threshold, setThreshold] = useState(0.75);
  const [autoSpeak, setAutoSpeak] = useState(true);

  const [transcript, setTranscript] = useState("");
  const [interimText, setInterimText] = useState("");
  const [isListening, setIsListening] = useState(false);
  const [micStatus, setMicStatus] = useState("Idle — click Start Microphone to speak.");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const [voices, setVoices] = useState([]);
  const [ttsStatus, setTtsStatus] = useState("");
  const recognitionRef = useRef(null);

  const SpeechRecognitionAPI =
    typeof window !== "undefined"
      ? window.SpeechRecognition || window.webkitSpeechRecognition
      : null;

  const sttSupported = Boolean(SpeechRecognitionAPI);
  const ttsSupported = typeof window !== "undefined" && "speechSynthesis" in window;

  useEffect(() => {
    if (!ttsSupported) return;
    const loadVoices = () => {
      const list = window.speechSynthesis.getVoices() || [];
      setVoices(list);
    };
    loadVoices();
    window.speechSynthesis.onvoiceschanged = loadVoices;
    return () => {
      window.speechSynthesis.onvoiceschanged = null;
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // ignore cleanup errors
        }
      }
    };
  }, [ttsSupported]);

  const hasEnglishVoice = voices.some((v) => v.lang?.toLowerCase().startsWith("en"));
  const hasTeluguVoice = voices.some((v) => v.lang?.toLowerCase().startsWith("te"));

  function startListening() {
    if (!SpeechRecognitionAPI) {
      setError(
        "Web Speech Recognition is not supported in this browser. Please test in Google Chrome or Microsoft Edge on Windows."
      );
      return;
    }

    setError("");
    const recognition = new SpeechRecognitionAPI();
    recognition.lang = language;
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      setIsListening(true);
      setMicStatus(`Listening in ${language === "te-IN" ? "Telugu (te-IN)" : "English (en-IN)"}...`);
    };

    recognition.onresult = (event) => {
      let finalChunk = "";
      let interimChunk = "";
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const piece = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          finalChunk += `${piece} `;
        } else {
          interimChunk += piece;
        }
      }
      if (finalChunk.trim()) {
        setTranscript((prev) => `${prev ? `${prev.trim()} ` : ""}${finalChunk.trim()}`);
      }
      setInterimText(interimChunk);
    };

    recognition.onerror = (event) => {
      setIsListening(false);
      setInterimText("");
      if (event.error === "not-allowed") {
        setError("Microphone permission was blocked. Allow microphone access in your browser address bar.");
      } else if (event.error === "no-speech") {
        setMicStatus("No speech detected. Click Start Microphone and try again.");
      } else {
        setError(`Speech recognition error: ${event.error}`);
      }
    };

    recognition.onend = () => {
      setIsListening(false);
      setInterimText("");
      setMicStatus("Microphone stopped. Review or edit the recognized text, then click Analyze.");
    };

    recognitionRef.current = recognition;
    recognition.start();
  }

  function stopListening() {
    if (recognitionRef.current) {
      recognitionRef.current.stop();
    }
    setIsListening(false);
    setInterimText("");
  }

  function buildSpokenMessage(res, langCode) {
    const confPct = Math.round((res.confidence || 0) * 100);
    const isTelugu = langCode === "te-IN";

    if (isTelugu) {
      const labelTe = res.prediction === "Stress" ? "ఒత్తిడి (Stress)" : "ఒత్తిడి లేదు (Non-Stress)";
      if (res.needs_human_review) {
        return `అంచనా ఫలితం ${labelTe}, నమ్మకం ${confPct} శాతం. ఈ ఫలితంలో అనిశ్చితి ఉంది, కాబట్టి మానవ సమీక్ష అవసరం.`;
      }
      return `అంచనా ఫలితం ${labelTe}. నమ్మకం ${confPct} శాతం. విశ్వసనీయత స్థాయి ${res.reliability_level}.`;
    }

    if (res.needs_human_review) {
      return `The text model predicts ${res.prediction} with ${confPct} percent calibrated confidence. Because uncertainty is above the reliability threshold, this result is flagged for human review.`;
    }
    return `The text model predicts ${res.prediction} with ${confPct} percent calibrated confidence and ${res.reliability_level} reliability.`;
  }

  function speakResult(resToSpeak = result) {
    if (!resToSpeak) return;
    if (!ttsSupported) {
      setTtsStatus("Browser Text-to-Speech (speechSynthesis) is not supported.");
      return;
    }

    window.speechSynthesis.cancel();
    const isTelugu = language === "te-IN";
    const textToRead = buildSpokenMessage(resToSpeak, language);
    const utterance = new SpeechSynthesisUtterance(textToRead);
    utterance.lang = language;

    const matchingVoice = voices.find((v) =>
      v.lang?.toLowerCase().startsWith(isTelugu ? "te" : "en")
    );

    if (matchingVoice) {
      utterance.voice = matchingVoice;
      setTtsStatus(`Speaking using browser voice: ${matchingVoice.name} (${matchingVoice.lang})`);
    } else if (isTelugu) {
      setTtsStatus(
        "No Telugu (te-IN) TTS voice is installed on this browser/OS. Browser will attempt fallback or remain silent; install Windows Telugu speech pack or test in Edge/Chrome."
      );
    } else {
      setTtsStatus("Speaking with default browser voice.");
    }

    window.speechSynthesis.speak(utterance);
  }

  async function handleAnalyzeTranscript() {
    const combined = `${transcript} ${interimText}`.trim();
    if (!combined) {
      setError("No recognized text available. Speak into the microphone or type text to test.");
      return;
    }

    if (isListening) stopListening();
    setLoading(true);
    setError("");

    try {
      const res = await predictStress({
        text: combined,
        model_type: modelType,
        client_id: clientId,
        confidence_threshold: threshold,
        source: "voice",
        language: language === "te-IN" ? "te" : "en",
      });
      setResult(res);
      if (autoSpeak) {
        speakResult(res);
      }
    } catch (err) {
      setError(err.message || "Voice text inference failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <div className="welcome">
        <div>
          <div className="eyebrow">ENGLISH & TELUGU SPEECH INTERFACE</div>
          <h1>Voice Assistant</h1>
          <p>
            Pipeline: Voice Input → Speech-to-Text → Existing RoBERTa + LoRA Text Model →
            Reliability Check → Result & Human-Review Warning → Text-to-Speech.
          </p>
        </div>
        <div className="project-tag">
          <span className="status-dot"></span>
          Same Text Stress Model (No Separate Voice Model)
        </div>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-top"><span>Web Speech STT</span><span className="stat-icon">♫</span></div>
          <h2>{sttSupported ? "Supported" : "Unavailable"}</h2>
          <p>Chrome / Edge recommended</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Web Speech TTS</span><span className="stat-icon">🔊</span></div>
          <h2>{ttsSupported ? "Supported" : "Unavailable"}</h2>
          <p>{voices.length} system voices detected</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>English TTS Voice</span><span className="stat-icon">EN</span></div>
          <h2>{hasEnglishVoice ? "Available" : "Default"}</h2>
          <p>en-IN / en-US speech output</p>
        </div>
        <div className="stat-card">
          <div className="stat-top"><span>Telugu TTS Voice</span><span className="stat-icon">తె</span></div>
          <h2>{hasTeluguVoice ? "Available" : "Not Found"}</h2>
          <p>{hasTeluguVoice ? "te-IN voice ready" : "Depends on OS/Edge voice pack"}</p>
        </div>
      </div>

      <div className="panel">
        <h3>Microphone Controls & Language Selection</h3>
        <p className="panel-subtitle">
          Test English (en-IN) and Telugu (te-IN) speech recognition and audio output independently
        </p>

        <div className="form-grid">
          <div className="field">
            <label>Voice Language (STT & TTS)</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              disabled={isListening}
            >
              <option value="en-IN">English (en-IN)</option>
              <option value="te-IN">Telugu — తెలుగు (te-IN)</option>
            </select>
          </div>

          <div className="field">
            <label>Text Stress-Detection Model</label>
            <select value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="personalized_ditto">Ditto-LoRA Personalized</option>
              <option value="federated_global">FedAvg Global Model</option>
              <option value="centralized_lora">Centralized Baseline</option>
            </select>
          </div>

          <div className="field">
            <label>Personalized Client (τ = {threshold.toFixed(2)})</label>
            <select
              value={clientId}
              onChange={(e) => setClientId(e.target.value)}
              disabled={modelType !== "personalized_ditto"}
            >
              {CLIENT_OPTIONS.map((c) => (
                <option key={c.id} value={c.id}>{c.label}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="btn-row" style={{ marginBottom: "14px" }}>
          <button
            type="button"
            className="btn-primary"
            onClick={startListening}
            disabled={isListening || !sttSupported}
          >
            {isListening ? "● Recording..." : "Start Microphone"}
          </button>
          <button
            type="button"
            className="btn-danger"
            onClick={stopListening}
            disabled={!isListening}
          >
            Stop Microphone
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => {
              setTranscript("");
              setInterimText("");
              setResult(null);
              setError("");
              setTtsStatus("");
            }}
          >
            Clear Transcript
          </button>
          <label style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "#4a5a6e" }}>
            <input
              type="checkbox"
              checked={autoSpeak}
              onChange={(e) => setAutoSpeak(e.target.checked)}
            />
            Read result aloud automatically
          </label>
        </div>

        <div style={{ fontSize: "12px", color: "#52657a", marginBottom: "10px" }}>
          <strong>Status:</strong> {micStatus}
        </div>

        <div className="field">
          <label>Recognized Text (Confirm or Edit Before Sending to Stress Model)</label>
          <textarea
            value={interimText ? `${transcript} ${interimText}`.trim() : transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder={
              language === "te-IN"
                ? "తెలుగులో మాట్లాడండి లేదా ఇక్కడ టెక్స్ట్ మార్చండి..."
                : "Speak into your microphone or edit the recognized English text here..."
            }
          />
        </div>

        <div className="btn-row">
          <button
            type="button"
            className="btn-primary"
            onClick={handleAnalyzeTranscript}
            disabled={loading}
          >
            {loading ? "Analyzing Recognized Text..." : "Send to Text Stress Model"}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => speakResult(result)}
            disabled={!result || !ttsSupported}
          >
            Read Result Aloud ({language === "te-IN" ? "Telugu" : "English"})
          </button>
        </div>

        {ttsStatus && (
          <div style={{ fontSize: "12px", color: "#287d75", marginTop: "10px" }}>
            <strong>TTS Output:</strong> {ttsStatus}
          </div>
        )}

        {error && (
          <div className="notice danger">
            <strong>Voice / Inference Error</strong>
            <p>{error}</p>
          </div>
        )}
      </div>

      <PredictionResultCard result={result} />

      <div className="notice">
        <strong>Browser Compatibility & Language Scope Notes</strong>
        <p>
          • <strong>Speech-to-Text (Web Speech API):</strong> Works best in Google Chrome and Microsoft Edge on
          Windows with an active internet connection. Firefox and Safari have limited or no support for
          SpeechRecognition.
        </p>
        <p>
          • <strong>Telugu TTS Output:</strong> Telugu audio readout depends on whether a <code>te-IN</code> voice
          is available in your browser/Windows speech pack (Microsoft Edge online voices typically include Telugu).
        </p>
        <p>
          • <strong>Cross-Lingual Model Limitation:</strong> Because the underlying RoBERTa + LoRA classifier was
          trained strictly on the English Dreaddit dataset, raw Telugu-script input may yield lower confidence and
          trigger the Human-Review flag—demonstrating why uncertainty-aware flagging is essential for out-of-distribution input.
        </p>
      </div>
    </>
  );
}