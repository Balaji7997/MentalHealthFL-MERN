require("dotenv").config();
const express = require("express");
const cors = require("cors");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const mongoose = require("mongoose");

const app = express();
const PORT = process.env.PORT || 5000;
const AI_SERVICE_URL = (process.env.AI_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const JWT_SECRET = process.env.JWT_SECRET || "mentalhealthfl_local_dev_secret_change_in_prod";
const MONGO_URI = process.env.MONGO_URI || "";

// Allow localhost for development + Vercel URL(s) from FRONTEND_URL env var in production
const envOrigins = (process.env.FRONTEND_URL || "")
  .split(",")
  .map((u) => u.trim().replace(/\/+$/, ""))
  .filter(Boolean);

const allowedOrigins = new Set([
  "http://localhost:5173",
  "http://127.0.0.1:5173",
  ...envOrigins,
]);

app.use(
  cors({
    origin(origin, callback) {
      if (!origin) return callback(null, true);
      const cleanOrigin = origin.replace(/\/+$/, "");
      if (allowedOrigins.has(cleanOrigin) || cleanOrigin.endsWith(".vercel.app")) {
        return callback(null, true);
      }
      return callback(new Error(`CORS blocked origin: ${origin}`));
    },
    credentials: true,
  })
);
app.use(express.json({ limit: "1mb" }));

// ============================================================================
// Database Setup (MongoDB Atlas in cloud, clean in-memory fallback locally)
// ============================================================================
let mongoConnected = false;

const userSchema = new mongoose.Schema({
  name: { type: String, required: true, trim: true },
  email: { type: String, required: true, unique: true, lowercase: true, trim: true },
  passwordHash: { type: String, required: true },
  role: { type: String, default: "Researcher" },
  createdAt: { type: Date, default: Date.now },
});

// Privacy-by-Design: Raw user mental-health text is NEVER included in this schema.
const sessionMetaSchema = new mongoose.Schema({
  userId: { type: String, required: true, index: true },
  timestamp: { type: Date, default: Date.now },
  input_channel: String,
  language: String,
  input_word_count: Number,
  model_used: String,
  prediction: String,
  confidence: Number,
  normalized_entropy: Number,
  needs_human_review: Boolean,
});

const UserModel = mongoose.model("User", userSchema);
const SessionMetaModel = mongoose.model("SessionMetadata", sessionMetaSchema);

// In-memory fallbacks when MONGO_URI is not configured yet
const memoryUsers = [];
const memorySessionLog = [];
const MAX_MEMORY_LOG = 100;

async function connectDatabase() {
  if (!MONGO_URI) {
    console.log("MONGO_URI not set — running with local in-memory user & metadata store.");
    return;
  }
  try {
    await mongoose.connect(MONGO_URI);
    mongoConnected = true;
    console.log("Connected to MongoDB Atlas successfully.");
  } catch (err) {
    mongoConnected = false;
    console.error("MongoDB connection failed (falling back to in-memory store):", err.message);
  }
}

// ============================================================================
// JWT Auth Helpers & Middleware
// ============================================================================
function signToken(user) {
  return jwt.sign(
    {
      id: user.id,
      name: user.name,
      email: user.email,
      role: user.role,
    },
    JWT_SECRET,
    { expiresIn: "7d" }
  );
}

function requireAuth(req, res, next) {
  const authHeader = req.headers.authorization || "";
  if (!authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "Authentication required. Please log in." });
  }
  const token = authHeader.slice(7).trim();
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    return next();
  } catch {
    return res.status(401).json({ error: "Session expired or invalid token. Please log in again." });
  }
}

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

// ============================================================================
// Public Health & Status Endpoints
// ============================================================================
app.get("/", (req, res) => {
  res.json({
    service: "MentalHealthFL Express Backend",
    status: "running",
    ai_service_url: AI_SERVICE_URL,
    mongodb_status: mongoConnected ? "connected" : "in-memory fallback (set MONGO_URI for persistence)",
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
    mongodb_status: mongoConnected ? "connected" : "in-memory fallback",
    ai_service_connected: aiConnected,
    ai_service: aiHealth,
    privacy_policy: "Raw user mental-health text is never persisted.",
  });
});

// ============================================================================
// Authentication Endpoints (Register / Login / Current User)
// ============================================================================
app.post("/api/auth/register", async (req, res) => {
  try {
    const name = String(req.body?.name || "").trim();
    const email = String(req.body?.email || "").trim().toLowerCase();
    const password = String(req.body?.password || "");

    if (!name || name.length < 2) {
      return res.status(400).json({ error: "Please enter your full name (at least 2 characters)." });
    }
    if (!email || !email.includes("@")) {
      return res.status(400).json({ error: "Please enter a valid email address." });
    }
    if (password.length < 6) {
      return res.status(400).json({ error: "Password must be at least 6 characters long." });
    }

    const passwordHash = await bcrypt.hash(password, 10);
    let userPayload;

    if (mongoConnected) {
      const existing = await UserModel.findOne({ email });
      if (existing) {
        return res.status(409).json({ error: "An account with this email already exists. Please log in." });
      }
      const created = await UserModel.create({ name, email, passwordHash });
      userPayload = {
        id: created._id.toString(),
        name: created.name,
        email: created.email,
        role: created.role,
      };
    } else {
      const existing = memoryUsers.find((u) => u.email === email);
      if (existing) {
        return res.status(409).json({ error: "An account with this email already exists. Please log in." });
      }
      const created = {
        id: `usr_${Date.now()}`,
        name,
        email,
        passwordHash,
        role: "Researcher",
      };
      memoryUsers.push(created);
      userPayload = {
        id: created.id,
        name: created.name,
        email: created.email,
        role: created.role,
      };
    }

    const token = signToken(userPayload);
    return res.status(201).json({ token, user: userPayload });
  } catch (err) {
    return res.status(500).json({ error: "Registration failed.", details: err.message });
  }
});

app.post("/api/auth/login", async (req, res) => {
  try {
    const email = String(req.body?.email || "").trim().toLowerCase();
    const password = String(req.body?.password || "");

    if (!email || !password) {
      return res.status(400).json({ error: "Email and password are required." });
    }

    let foundUser = null;
    if (mongoConnected) {
      const doc = await UserModel.findOne({ email });
      if (doc) {
        foundUser = {
          id: doc._id.toString(),
          name: doc.name,
          email: doc.email,
          passwordHash: doc.passwordHash,
          role: doc.role,
        };
      }
    } else {
      foundUser = memoryUsers.find((u) => u.email === email) || null;
    }

    if (!foundUser) {
      return res.status(401).json({ error: "Invalid email or password." });
    }

    const validPassword = await bcrypt.compare(password, foundUser.passwordHash);
    if (!validPassword) {
      return res.status(401).json({ error: "Invalid email or password." });
    }

    const userPayload = {
      id: foundUser.id,
      name: foundUser.name,
      email: foundUser.email,
      role: foundUser.role,
    };

    const token = signToken(userPayload);
    return res.json({ token, user: userPayload });
  } catch (err) {
    return res.status(500).json({ error: "Login failed.", details: err.message });
  }
});

app.get("/api/auth/me", requireAuth, (req, res) => {
  res.json({ user: req.user });
});

// ============================================================================
// Protected Research & Inference Endpoints
// ============================================================================
app.get("/api/research", requireAuth, (req, res) => proxyGet("/api/research", res));
app.get("/api/overview", requireAuth, (req, res) => proxyGet("/api/overview", res));
app.get("/api/clients", requireAuth, (req, res) => proxyGet("/api/clients", res));
app.get("/api/aggregation", requireAuth, (req, res) => proxyGet("/api/aggregation", res));
app.get("/api/personalization", requireAuth, (req, res) => proxyGet("/api/personalization", res));
app.get("/api/reliability", requireAuth, (req, res) => proxyGet("/api/reliability", res));
app.get("/api/evaluation", requireAuth, (req, res) => proxyGet("/api/evaluation", res));

app.post("/api/predict", requireAuth, async (req, res) => {
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

    // Store ONLY non-sensitive metadata tied to the logged-in user (no raw text)
    const metadataEntry = {
      userId: req.user.id,
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

    if (mongoConnected) {
      await SessionMetaModel.create(metadataEntry);
    } else {
      memorySessionLog.unshift({ id: `pred_${Date.now()}`, ...metadataEntry });
      if (memorySessionLog.length > MAX_MEMORY_LOG) {
        memorySessionLog.pop();
      }
    }

    return res.json(result);
  } catch (err) {
    return res.status(502).json({
      error: "Prediction request failed while communicating with Python AI Service",
      details: err.message,
    });
  }
});

app.get("/api/sessions", requireAuth, async (req, res) => {
  try {
    let userLogs = [];
    if (mongoConnected) {
      const docs = await SessionMetaModel.find({ userId: req.user.id })
        .sort({ timestamp: -1 })
        .limit(100)
        .lean();
      userLogs = docs.map((d) => ({
        id: d._id.toString(),
        timestamp: d.timestamp,
        input_channel: d.input_channel,
        language: d.language,
        input_word_count: d.input_word_count,
        model_used: d.model_used,
        prediction: d.prediction,
        confidence: d.confidence,
        normalized_entropy: d.normalized_entropy,
        needs_human_review: d.needs_human_review,
      }));
    } else {
      userLogs = memorySessionLog.filter((x) => x.userId === req.user.id);
    }

    const total = userLogs.length;
    const stressCount = userLogs.filter((x) => x.prediction === "Stress").length;
    const reviewCount = userLogs.filter((x) => x.needs_human_review).length;
    const voiceCount = userLogs.filter((x) => x.input_channel === "voice").length;

    return res.json({
      storage_mode: mongoConnected ? "MongoDB Atlas (metadata only)" : "in-memory metadata",
      privacy_note: "Only non-sensitive prediction metadata is logged; raw user text is never stored.",
      totals: {
        predictions: total,
        stress_predictions: stressCount,
        non_stress_predictions: total - stressCount,
        human_review_flagged: reviewCount,
        voice_requests: voiceCount,
      },
      recent: userLogs.slice(0, 20),
    });
  } catch (err) {
    return res.status(500).json({ error: "Failed to load session metadata.", details: err.message });
  }
});

connectDatabase().finally(() => {
  app.listen(PORT, "0.0.0.0", () => {
    console.log(`MentalHealthFL Express Backend running on port ${PORT}`);
    console.log(`Connected AI Service target: ${AI_SERVICE_URL}`);
    console.log("Privacy mode active: raw mental-health text is NOT persisted.");
  });
});