import { useState, useRef, useCallback } from 'react';
import { Upload, ImageIcon, X, CheckCircle } from 'lucide-react';

interface ImageUploaderProps {
  onFileSelect: (file: File) => void;
  selectedFile?: File | null;
  onClear?: () => void;
}

const ALLOWED = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];

export function ImageUploader({ onFileSelect, selectedFile, onClear }: ImageUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (!ALLOWED.includes(file.type)) {
      alert('Please upload a JPG, PNG, WebP, or BMP image.');
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onFileSelect(file);
  }, [onFileSelect]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleClear = () => {
    setPreviewUrl(null);
    if (inputRef.current) inputRef.current.value = '';
    onClear?.();
  };

  if (previewUrl && selectedFile) {
    return (
      <div className="uploader">
        <div className="uploader__preview">
          <img src={previewUrl} alt="Preview" />
          <div className="uploader__preview-overlay">
            <span className="badge badge-success">
              <CheckCircle size={11} /> {selectedFile.name}
            </span>
            <button
              className="btn btn-ghost btn-sm"
              style={{ marginLeft: 'auto' }}
              onClick={handleClear}
            >
              <X size={14} /> Clear
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="uploader">
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={e => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />
      <div
        className={`uploader__zone${dragOver ? ' drag-over' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <div className="uploader__icon">
          {dragOver ? <ImageIcon size={28} /> : <Upload size={28} />}
        </div>
        <div>
          <p className="uploader__title">
            {dragOver ? 'Drop your leaf image here' : 'Upload a leaf image'}
          </p>
          <p className="uploader__subtitle">
            Drag & drop or click to browse
          </p>
        </div>
        <div className="uploader__formats">
          {['JPG', 'PNG', 'WebP', 'BMP'].map(fmt => (
            <span key={fmt} className="badge badge-info">{fmt}</span>
          ))}
        </div>
      </div>
    </div>
  );
}
