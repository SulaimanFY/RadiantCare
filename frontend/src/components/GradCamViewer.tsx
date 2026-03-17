import { useState } from "react";
import { Eye, EyeOff, Flame } from "lucide-react";

interface Props {
  originalSrc: string;
  gradCamSrc: string;
  gradCamLabel: string;
}

export default function GradCamViewer({
  originalSrc,
  gradCamSrc,
  gradCamLabel,
}: Props) {
  const [showCam, setShowCam] = useState(true);

  return (
    <div className="overflow-hidden rounded-xl border border-gray-200 bg-white">
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-2.5">
        <div className="flex items-center gap-2 text-sm font-semibold text-gray-700">
          <Flame className="h-4 w-4 text-orange-500" />
          {showCam ? "Grad-CAM" : "Original"}
          {showCam && (
            <span className="rounded-full bg-orange-100 px-2 py-0.5 text-[10px] font-bold text-orange-600">
              {gradCamLabel}
            </span>
          )}
        </div>

        <button
          onClick={() => setShowCam(!showCam)}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-medium text-gray-500 transition hover:bg-gray-100"
        >
          {showCam ? (
            <>
              <EyeOff className="h-3.5 w-3.5" /> Show original
            </>
          ) : (
            <>
              <Eye className="h-3.5 w-3.5" /> Show heatmap
            </>
          )}
        </button>
      </div>

      <div className="relative p-3">
        <img
          src={showCam ? gradCamSrc : originalSrc}
          alt={showCam ? "Grad-CAM heatmap overlay" : "Original X-ray"}
          className="mx-auto max-h-80 rounded-lg object-contain transition-opacity duration-300"
        />
      </div>
    </div>
  );
}
