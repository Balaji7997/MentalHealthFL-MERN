const express = require("express");
const cors = require("cors");

const app = express();
const PORT = process.env.PORT || 5000;
const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

app.use(
  cors({
    origin: ["http://localhost:5173", "http://127.0.0.1:5173"],
    credentials: true,
  })
);
app.use(express.json({ limit: "1mb" }));

// Privacy-by-design metadata store (in-memory until MongoDB is connected).
// IMPORTANT: Raw user mental-health text is NEVER stored here.
const sessionMetadataLog = [];
const MAX_LOG_ITEMS = 100;

async function proxyGet(aiPath, res) {
  try {
    const response = await fetch(`${AI_SERVICE_URL}${aiPath}`);
    const data = await response.json();
    if (!response.ok) {
      return res.status(response.status).json(data);
    }
    return res.json(data);
  } catch (err) {
    return res.status(502).json({
      error: "Unable to reach Python FastAPI AI Service",
      ai_service_url: AI_SERVICE_URL,
      details: err.message,
    });
  }
}

app.get("/", (req, res) => {
  res.json({
    service: "MentalHealthFL Express Backend",
    status: "running",
    ai_service_url: AI_SERVICE_URL,
    mongodb_status: "not_connected (optional - using privacy-safe in-memory metadata log)",
  });
});

app.get("/api/health", async (req, res) => {
  let aiHealth = null;
  let aiConnected = false;

  try {
    const response = await fetch(`${AI_SERVICE_URL}/health`);
    if (response.ok) {
      aiHealth = await response.json();
      aiConnected = true;
    }
  } catch (err) {
    aiHealth = { error: err.message };
  }

  res.json({
    backend_status: "healthy",
    port: PORT,
    mongodb_status: "not_connected (optional)",
    ai_service_connected: aiConnected,
    ai_service: aiHealth,
    privacy_policy: "Raw user mental-health text is never persisted.",
  });
});

// Research & Evaluation proxy endpoints (live real experimental artifacts from FastAPI)
app.get("/api/research", (req, res) => proxyGet("/api/research", res));
app.get("/api/overview", (req, res) => proxyGet("/api/overview", res));
app.get("/api/clients", (req, res) => proxyGet("/api/clients", res));
app.get("/api/aggregation", (req, res) => proxyGet("/api/aggregation", res));
app.get("/api/personalization", (req, res) => proxyGet("/api/personalization", res));
app.get("/api/reliability", (req, res) => proxyGet("/api/reliability", res));
app.get("/api/evaluation", (req, res) => proxyGet("/api/evaluation", res));

// Live Stress Detection & Voice Assistant inference endpoint
app.post("/api/predict", async (req, res) => {
  try {
    const {
      text,
      model_type = "personalized_ditto",
      client_id = "client_2",
      confidence_threshold = 0.75,
      source = "text",
      language = "en",
    } = req.body || {};

    if (!text || typeof text !== "string" || !text.trim()) {
      return res.status(400).json({ error: "Input text is required and cannot be empty." });
    }

    const aiResponse = await fetch(`${AI_SERVICE_URL}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text.trim(),
        model_type,
        client_id,
        confidence_threshold: Number(confidence_threshold),
        source,
        language,
      }),
    });

    const result = await aiResponse.json();
    if (!aiResponse.ok) {
      return res.status(aiResponse.status).json(result);
    }

    // Store ONLY non-sensitive metadata (no raw text stored)
    const metadataEntry = {
      id: `pred_${Date.now()}`,
      timestamp: new Date().toISOString(),
      input_channel: source,
      language,
      input_word_count: result.input_word_count,
      model_used: result.model_used,
      prediction: result.prediction,
      confidence: result.confidence,
      normalized_entropy: result.normalized_entropy,
      needs_human_review: result.needs_human_review,
    };

    sessionMetadataLog.unshift(metadataEntry);
    if (sessionMetadataLog.length > MAX_LOG_ITEMS) {
      sessionMetadataLog.pop();
    }

    return res.json(result);
  } catch (err) {
    return res.status(502).json({
      error: "Prediction request failed while communicating with Python AI Service",
      details: err.message,
    });
  }
});

// Session audit summary (non-sensitive metadata only)
app.get("/api/sessions", (req, res) => {
  const total = sessionMetadataLog.length;
  const stressCount = sessionMetadataLog.filter((x) => x.prediction === "Stress").length;
  const reviewCount = sessionMetadataLog.filter((x) => x.needs_human_review).length;
  const voiceCount = sessionMetadataLog.filter((x) => x.input_channel === "voice").length;

  res.json({
    storage_mode: "in-memory metadata (MongoDB optional)",
    privacy_note: "Only non-sensitive prediction metadata is logged; raw user text is never stored.",
    totals: {
      predictions: total,
      stress_predictions: stressCount,
      non_stress_predictions: total - stressCount,
      human_review_flagged: reviewCount,
      voice_requests: voiceCount,
    },
    recent: sessionMetadataLog.slice(0, 20),
  });
});

app.listen(PORT, "127.0.0.1", () => {
  console.log(`MentalHealthFL Express Backend running at http://127.0.0.1:${PORT}`);
  console.log(`Connected AI Service target: ${AI_SERVICE_URL}`);
  console.log("Privacy mode active: raw mental-health text is NOT persisted.");
});