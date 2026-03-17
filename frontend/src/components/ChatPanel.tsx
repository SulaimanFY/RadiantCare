import { useState, useRef, useEffect } from "react";
import { Send, MessageCircle, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { fetchChat } from "../api/client";
import type { ChatMessage, ClinicalContext, Prediction } from "../api/types";

interface Props {
  predictions: Prediction[] | null;
  report: string | null;
  clinicalContext: ClinicalContext;
}

export default function ChatPanel({
  predictions,
  report,
  clinicalContext,
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetchChat({
        message: text,
        predictions: predictions ?? undefined,
        report: report ?? undefined,
        clinical_context: clinicalContext,
        history: messages,
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: res.answer },
      ]);
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errMsg}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col rounded-xl border border-gray-200 bg-white">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-gray-100 px-4 py-3">
        <MessageCircle className="h-4 w-4 text-primary-600" />
        <h3 className="text-sm font-semibold text-gray-700">
          Ask the assistant
        </h3>
      </div>

      {/* Messages */}
      <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4" style={{ maxHeight: 400 }}>
        {messages.length === 0 && (
          <p className="py-8 text-center text-sm text-gray-400">
            Ask follow-up questions about the predictions or report.
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm ${
                m.role === "user"
                  ? "bg-primary-600 text-white"
                  : "bg-gray-100 text-gray-700"
              }`}
            >
              {m.role === "assistant" ? (
                <div className="prose prose-sm max-w-none prose-headings:mt-3 prose-headings:mb-1.5 prose-headings:font-semibold prose-p:my-2 first:prose-p:mt-0 last:prose-p:mb-0 prose-ul:my-2 prose-li:my-0.5 prose-ul:list-disc prose-ol:list-decimal">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
                </div>
              ) : (
                m.content
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="flex items-center gap-2 rounded-2xl bg-gray-100 px-4 py-2.5 text-sm text-gray-500">
              <Loader2 className="h-4 w-4 animate-spin" />
              Thinking…
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="flex items-center gap-2 border-t border-gray-100 px-4 py-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          disabled={loading}
          placeholder="Type a question…"
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-400 focus:ring-1 focus:ring-primary-400 focus:outline-none disabled:opacity-50"
        />
        <button
          onClick={send}
          disabled={loading || !input.trim()}
          className="rounded-lg bg-primary-600 p-2 text-white transition hover:bg-primary-700 disabled:opacity-40"
        >
          <Send className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
