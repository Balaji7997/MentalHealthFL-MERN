export const API_BASE_URL = "http://127.0.0.1:5000";

async function handleResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message =
      data.detail || data.error || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return data;
}

export async function fetchHealth() {
  const res = await fetch(`${API_BASE_URL}/api/health`);
  return handleResponse(res);
}

export async function fetchOverview() {
  const res = await fetch(`${API_BASE_URL}/api/overview`);
  return handleResponse(res);
}

export async function fetchClients() {
  const res = await fetch(`${API_BASE_URL}/api/clients`);
  return handleResponse(res);
}

export async function fetchAggregation() {
  const res = await fetch(`${API_BASE_URL}/api/aggregation`);
  return handleResponse(res);
}

export async function fetchPersonalization() {
  const res = await fetch(`${API_BASE_URL}/api/personalization`);
  return handleResponse(res);
}

export async function fetchReliability() {
  const res = await fetch(`${API_BASE_URL}/api/reliability`);
  return handleResponse(res);
}

export async function fetchEvaluation() {
  const res = await fetch(`${API_BASE_URL}/api/evaluation`);
  return handleResponse(res);
}

export async function fetchSessions() {
  const res = await fetch(`${API_BASE_URL}/api/sessions`);
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
    headers: { "Content-Type": "application/json" },
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