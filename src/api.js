export const API_BASE_URL = (
  import.meta.env.VITE_API_URL || "http://127.0.0.1:5000"
).replace(/\/+$/, "");

const TOKEN_KEY = "mentalhealthfl_token";

export function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function setToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  }
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(extra = {}) {
  const token = getToken();
  return {
    ...extra,
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (response.status === 401) {
    clearToken();
  }
  if (!response.ok) {
    const message =
      data.detail || data.error || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return data;
}

export async function registerUser({ name, email, password }) {
  const res = await fetch(`${API_BASE_URL}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, email, password }),
  });
  const data = await handleResponse(res);
  setToken(data.token);
  return data;
}

export async function loginUser({ email, password }) {
  const res = await fetch(`${API_BASE_URL}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  const data = await handleResponse(res);
  setToken(data.token);
  return data;
}

export async function fetchCurrentUser() {
  const res = await fetch(`${API_BASE_URL}/api/auth/me`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  return handleResponse(res);
}

export async function fetchOverview() {
  const res = await fetch(`${API_BASE_URL}/api/overview`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchClients() {
  const res = await fetch(`${API_BASE_URL}/api/clients`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchAggregation() {
  const res = await fetch(`${API_BASE_URL}/api/aggregation`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchPersonalization() {
  const res = await fetch(`${API_BASE_URL}/api/personalization`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchReliability() {
  const res = await fetch(`${API_BASE_URL}/api/reliability`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchEvaluation() {
  const res = await fetch(`${API_BASE_URL}/api/evaluation`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function fetchSessions() {
  const res = await fetch(`${API_BASE_URL}/api/sessions`, {
    headers: authHeaders(),
  });
  return handleResponse(res);
}

export async function predictStress({
  text,
  model_type = "personalized_ditto",
  client_id = "client_2",
  confidence_threshold = 0.75,
  source = "text",
  language = "en",
}) {
  const res = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({
      text,
      model_type,
      client_id,
      confidence_threshold: Number(confidence_threshold),
      source,
      language,
    }),
  });
  return handleResponse(res);
}

export function pct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

export function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(digits);
}