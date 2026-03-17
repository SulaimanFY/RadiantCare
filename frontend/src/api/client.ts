import type {
  ClinicalContext,
  FullReportResponse,
  HealthResponse,
  PredictResponse,
  ChatRequest,
  ChatResponse,
} from "./types";

const BASE = "/api";

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? res.statusText;
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/health`);
  return handleResponse<HealthResponse>(res);
}

export async function fetchPredict(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/predict`, { method: "POST", body: form });
  return handleResponse<PredictResponse>(res);
}

export async function fetchFullReport(
  file: File,
  ctx?: ClinicalContext,
): Promise<FullReportResponse> {
  const form = new FormData();
  form.append("file", file);
  if (ctx) {
    form.append("clinical_context", JSON.stringify(ctx));
  }
  const res = await fetch(`${BASE}/full-report`, {
    method: "POST",
    body: form,
  });
  return handleResponse<FullReportResponse>(res);
}

export async function fetchChat(payload: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse<ChatResponse>(res);
}
