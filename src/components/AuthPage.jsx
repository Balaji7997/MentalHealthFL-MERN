import { useState } from "react";
import { loginUser, registerUser } from "../api";

export default function AuthPage({ onAuthSuccess }) {
  const [mode, setMode] = useState("login"); // "login" or "register"
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const data =
        mode === "register"
          ? await registerUser({ name, email, password })
          : await loginUser({ email, password });

      onAuthSuccess(data.user);
    } catch (err) {
      setError(err.message || "Authentication failed. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <div className="auth-header">
          <div className="brand-icon">M</div>
          <div>
            <h2>MentalHealthFL</h2>
            <span>RELIABILITY-AWARE PERSONALIZED FL</span>
          </div>
        </div>

        <div className="auth-tabs">
          <button
            type="button"
            className={`auth-tab ${mode === "login" ? "active" : ""}`}
            onClick={() => {
              setMode("login");
              setError("");
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab ${mode === "register" ? "active" : ""}`}
            onClick={() => {
              setMode("register");
              setError("");
            }}
          >
            Create Account
          </button>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          {mode === "register" && (
            <div className="field">
              <label>Full Name</label>
              <input
                type="text"
                placeholder="e.g., Anil Dattu"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                minLength={2}
              />
            </div>
          )}

          <div className="field">
            <label>Email Address</label>
            <input
              type="email"
              placeholder="you@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>

          <div className="field">
            <label>Password</label>
            <input
              type="password"
              placeholder={mode === "register" ? "Minimum 6 characters" : "Enter your password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
            />
          </div>

          {error && (
            <div className="notice danger" style={{ marginTop: 0 }}>
              <strong>Authentication Error</strong>
              <p>{error}</p>
            </div>
          )}

          <button type="submit" className="btn-primary auth-submit" disabled={loading}>
            {loading
              ? "Connecting to server..."
              : mode === "register"
              ? "Create Account & Enter Dashboard"
              : "Sign In to Dashboard"}
          </button>
        </form>

        <div className="notice" style={{ marginTop: "18px" }}>
          <strong>Privacy Guarantee</strong>
          <p>
            Accounts secure dashboard access and personal session statistics. Raw mental-health
            text and voice transcripts are never stored in any database.
          </p>
        </div>
      </div>
    </div>
  );
}