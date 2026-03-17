import { useCallback, useRef, useState } from "react";
import { ImagePlus, X } from "lucide-react";

interface Props {
  file: File | null;
  onFileChange: (file: File | null) => void;
  disabled?: boolean;
}

export default function ImageUpload({ file, onFileChange, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragActive, setDragActive] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const accept = (f: File) => {
    onFileChange(f);
    setPreview(URL.createObjectURL(f));
  };

  const clear = () => {
    onFileChange(null);
    setPreview(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      if (disabled) return;
      const f = e.dataTransfer.files?.[0];
      if (f && f.type.startsWith("image/")) accept(f);
    },
    [disabled],
  );

  return (
    <div className="flex flex-col items-center gap-3">
      <div
        className={`relative flex w-full cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition-colors ${
          dragActive
            ? "border-primary-400 bg-primary-50"
            : "border-gray-300 bg-white hover:border-primary-300 hover:bg-primary-50/40"
        } ${disabled ? "pointer-events-none opacity-50" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        {preview ? (
          <>
            <img
              src={preview}
              alt="X-ray preview"
              className="max-h-64 rounded-lg object-contain"
            />
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                clear();
              }}
              className="absolute right-2 top-2 rounded-full bg-gray-800/60 p-1 text-white transition hover:bg-gray-800"
            >
              <X className="h-4 w-4" />
            </button>
          </>
        ) : (
          <>
            <ImagePlus className="mb-2 h-10 w-10 text-gray-400" />
            <p className="text-sm font-medium text-gray-600">
              Drop a chest X-ray here or{" "}
              <span className="text-primary-600 underline">browse</span>
            </p>
            <p className="mt-1 text-xs text-gray-400">
              PNG, JPG or DICOM
            </p>
          </>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) accept(f);
          }}
        />
      </div>
      {file && (
        <p className="text-xs text-gray-500">
          {file.name} ({(file.size / 1024).toFixed(0)} KB)
        </p>
      )}
    </div>
  );
}
