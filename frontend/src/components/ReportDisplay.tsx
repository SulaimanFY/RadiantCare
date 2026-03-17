import ReactMarkdown from "react-markdown";
import { FileText } from "lucide-react";

interface Props {
  report: string;
  disclaimer: string;
}

export default function ReportDisplay({ report, disclaimer }: Props) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white">
      <div className="flex items-center gap-2 border-b border-gray-100 px-4 py-3">
        <FileText className="h-4 w-4 text-primary-600" />
        <h3 className="text-sm font-semibold text-gray-700">AI Report</h3>
      </div>

      <div className="prose prose-sm max-w-none px-4 py-4 text-gray-700">
        <ReactMarkdown>{report}</ReactMarkdown>
      </div>

      <div className="border-t border-amber-100 bg-amber-50 px-4 py-2.5">
        <p className="text-xs italic text-amber-700">{disclaimer}</p>
      </div>
    </div>
  );
}
