import { useEffect, useState } from "react";
import { Loader2, Sparkles, Stethoscope, Timer } from "lucide-react";
import Layout from "./components/Layout";
import ImageUpload from "./components/ImageUpload";
import ClinicalContextForm from "./components/ClinicalContextForm";
import PredictionsTable from "./components/PredictionsTable";
import ReportDisplay from "./components/ReportDisplay";
import ChatPanel from "./components/ChatPanel";
import GradCamViewer from "./components/GradCamViewer";
import { fetchHealth, fetchFullReport, fetchPredict } from "./api/client";
import type {
  ClinicalContext,
  HealthResponse,
  Prediction,
} from "./api/types";

type AnalysisMode = "predict" | "full-report";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [ctx, setCtx] = useState<ClinicalContext>({});
  const [mode, setMode] = useState<AnalysisMode>("full-report");

  const [predictions, setPredictions] = useState<Prediction[] | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [disclaimer, setDisclaimer] = useState<string | null>(null);
  const [gradCamImage, setGradCamImage] = useState<string | null>(null);
  const [gradCamLabel, setGradCamLabel] = useState<string | null>(null);
  const [inferenceMs, setInferenceMs] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  const analyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setPredictions(null);
    setReport(null);
    setDisclaimer(null);
    setGradCamImage(null);
    setGradCamLabel(null);
    setInferenceMs(null);

    try {
      if (mode === "full-report") {
        const res = await fetchFullReport(file, ctx);
        setPredictions(res.predictions);
        setReport(res.report);
        setDisclaimer(res.disclaimer);
        setGradCamImage(res.grad_cam_image);
        setGradCamLabel(res.grad_cam_label);
        setInferenceMs(res.inference_time_ms);
      } else {
        const res = await fetchPredict(file);
        setPredictions(res.predictions);
        setGradCamImage(res.grad_cam_image);
        setGradCamLabel(res.grad_cam_label);
        setInferenceMs(res.inference_time_ms);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPredictions(null);
    setReport(null);
    setDisclaimer(null);
    setGradCamImage(null);
    setGradCamLabel(null);
    setInferenceMs(null);
    setError(null);
    setCtx({});
  };

  const hasResults = predictions !== null;

  return (
    <Layout health={health}>
      {/* Hero */}
      {!hasResults && !loading && (
        <div className="mb-8 text-center">
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-gray-900">
            Chest X-Ray Analysis
          </h1>
          <p className="mx-auto max-w-lg text-sm text-gray-500">
            Upload a chest X-ray image, optionally add clinical context, and
            get AI-powered predictions with an LLM-generated report.
          </p>
        </div>
      )}

      {/* Input section */}
      {!hasResults && (
        <div className="mx-auto max-w-xl space-y-4">
          <ImageUpload file={file} onFileChange={setFile} disabled={loading} />

          <ClinicalContextForm ctx={ctx} onChange={setCtx} disabled={loading} />

          {/* Mode toggle */}
          <div className="flex items-center justify-center gap-3">
            <button
              onClick={() => setMode("predict")}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                mode === "predict"
                  ? "bg-primary-100 text-primary-700"
                  : "text-gray-500 hover:bg-gray-100"
              }`}
            >
              <Stethoscope className="mr-1 inline h-3.5 w-3.5" />
              Predictions only
            </button>
            <button
              onClick={() => setMode("full-report")}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                mode === "full-report"
                  ? "bg-primary-100 text-primary-700"
                  : "text-gray-500 hover:bg-gray-100"
              }`}
            >
              <Sparkles className="mr-1 inline h-3.5 w-3.5" />
              Full report
            </button>
          </div>

          {/* Submit */}
          <button
            onClick={analyze}
            disabled={!file || loading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-primary-600 py-3 text-sm font-semibold text-white shadow-lg shadow-primary-200 transition hover:bg-primary-700 disabled:opacity-40 disabled:shadow-none"
          >
            {loading ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Analyze X-Ray"
            )}
          </button>

          {error && (
            <div className="rounded-lg border border-danger-200 bg-danger-50 px-4 py-3 text-sm text-danger-700">
              {error}
            </div>
          )}
        </div>
      )}

      {/* Loading overlay */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="mb-4 h-10 w-10 animate-spin text-primary-500" />
          <p className="text-sm font-medium text-gray-500">
            {mode === "full-report"
              ? "Running model & generating report..."
              : "Running model predictions..."}
          </p>
        </div>
      )}

      {/* Results */}
      {hasResults && !loading && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h2 className="text-xl font-bold text-gray-900">Results</h2>
              {inferenceMs !== null && (
                <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-medium text-gray-600">
                  <Timer className="h-3 w-3" />
                  {inferenceMs.toFixed(0)} ms inference
                </span>
              )}
            </div>
            <button
              onClick={reset}
              className="rounded-lg border border-gray-300 px-4 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-100"
            >
              New analysis
            </button>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* Left column: image / Grad-CAM + predictions */}
            <div className="space-y-4">
              {file && gradCamImage && gradCamLabel ? (
                <GradCamViewer
                  originalSrc={URL.createObjectURL(file)}
                  gradCamSrc={gradCamImage}
                  gradCamLabel={gradCamLabel}
                />
              ) : file ? (
                <div className="overflow-hidden rounded-xl border border-gray-200 bg-white p-3">
                  <img
                    src={URL.createObjectURL(file)}
                    alt="Uploaded X-ray"
                    className="mx-auto max-h-72 rounded-lg object-contain"
                  />
                </div>
              ) : null}
              <PredictionsTable predictions={predictions} />
            </div>

            {/* Right column: report + chat */}
            <div className="space-y-4">
              {report && disclaimer && (
                <ReportDisplay report={report} disclaimer={disclaimer} />
              )}
              <ChatPanel
                predictions={predictions}
                report={report}
                clinicalContext={ctx}
              />
            </div>
          </div>
        </div>
      )}
    </Layout>
  );
}
