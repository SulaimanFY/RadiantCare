export interface Prediction {
  label: string;
  probability: number;
  positive: boolean;
}

export interface PredictResponse {
  model_version: string;
  predictions: Prediction[];
  inference_time_ms: number | null;
  grad_cam_image: string | null;
  grad_cam_label: string | null;
}

export interface ClinicalContext {
  free_text?: string;
  age?: number;
  sex?: string;
  other_info?: string;
}

export interface FullReportResponse {
  predictions: Prediction[];
  report: string;
  disclaimer: string;
  inference_time_ms: number | null;
  grad_cam_image: string | null;
  grad_cam_label: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  message: string;
  predictions?: Prediction[];
  report?: string;
  clinical_context?: ClinicalContext;
  history?: ChatMessage[];
}

export interface ChatResponse {
  answer: string;
  disclaimer: string;
}

export interface HealthResponse {
  model_loaded: boolean;
  model_version: string;
  device: string;
  num_labels: number;
  error: string | null;
  rag_ready: boolean;
  rag_documents: number;
  rag_chunks: number;
  rag_error: string | null;
}
