import type { ClinicalContext } from "../api/types";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";

interface Props {
  ctx: ClinicalContext;
  onChange: (ctx: ClinicalContext) => void;
  disabled?: boolean;
}

export default function ClinicalContextForm({
  ctx,
  onChange,
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);

  const set = <K extends keyof ClinicalContext>(
    key: K,
    val: ClinicalContext[K],
  ) => onChange({ ...ctx, [key]: val });

  return (
    <div className="rounded-xl border border-gray-200 bg-white">
      <button
        type="button"
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold text-gray-700"
        onClick={() => setOpen(!open)}
      >
        Clinical context (optional)
        {open ? (
          <ChevronUp className="h-4 w-4 text-gray-400" />
        ) : (
          <ChevronDown className="h-4 w-4 text-gray-400" />
        )}
      </button>

      {open && (
        <div className="grid gap-4 border-t border-gray-100 px-4 py-4 sm:grid-cols-2">
          <div>
            <label className="mb-1 block text-xs font-medium text-gray-500">
              Age
            </label>
            <input
              type="number"
              min={0}
              max={150}
              value={ctx.age ?? ""}
              onChange={(e) =>
                set("age", e.target.value ? Number(e.target.value) : undefined)
              }
              disabled={disabled}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-400 focus:ring-1 focus:ring-primary-400 focus:outline-none disabled:opacity-50"
              placeholder="e.g. 58"
            />
          </div>

          <div>
            <label className="mb-1 block text-xs font-medium text-gray-500">
              Sex
            </label>
            <select
              value={ctx.sex ?? ""}
              onChange={(e) => set("sex", e.target.value || undefined)}
              disabled={disabled}
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-400 focus:ring-1 focus:ring-primary-400 focus:outline-none disabled:opacity-50"
            >
              <option value="">Select</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div className="sm:col-span-2">
            <label className="mb-1 block text-xs font-medium text-gray-500">
              Additional notes
            </label>
            <textarea
              rows={2}
              value={ctx.free_text ?? ""}
              onChange={(e) => set("free_text", e.target.value || undefined)}
              disabled={disabled}
              className="w-full resize-none rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-400 focus:ring-1 focus:ring-primary-400 focus:outline-none disabled:opacity-50"
              placeholder="e.g. Persistent cough for 3 weeks, smoker..."
            />
          </div>
        </div>
      )}
    </div>
  );
}
