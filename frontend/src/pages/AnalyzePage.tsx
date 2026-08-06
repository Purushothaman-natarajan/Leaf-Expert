import { useState } from 'react';
import { FlaskConical, Settings2, AlertCircle, Loader2 } from 'lucide-react';
import { ImageUploader } from '../components/ImageUploader';
import { ResultCard } from '../components/ResultCard';
import { ExplanationPanel } from '../components/ExplanationPanel';
import { explainImage, type ExplainResponse } from '../api/client';

export function AnalyzePage() {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [modelPath, setModelPath] = useState('');
  const [numSamples, setNumSamples] = useState(100);
  const [numFeatures, setNumFeatures] = useState(30);
  const [segAlg, setSegAlg] = useState('quickshift');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!imageFile) { setError('Please select an image first.'); return; }
    if (!modelPath) { setError('Please enter the model path.'); return; }
    setError(null);
    setResult(null);
    setLoading(true);
    try {
      const res = await explainImage(imageFile, modelPath, numSamples, numFeatures, segAlg);
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Analysis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <div className="page__header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <FlaskConical size={22} color="var(--accent)" />
            <div>
              <h1 className="page__title" style={{ fontSize: '1.8rem' }}>Leaf Analysis</h1>
              <p className="page__subtitle">Upload a leaf image and get an AI diagnosis with visual explanations</p>
            </div>
          </div>
        </div>

        <div className="analyze-grid">
          {/* Left: upload + settings */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ marginBottom: 20, fontSize: '1rem' }}>Image Input</h3>
              <ImageUploader
                onFileSelect={setImageFile}
                selectedFile={imageFile}
                onClear={() => { setImageFile(null); setResult(null); }}
              />
            </div>

            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>Model Configuration</h3>
              <div className="form-group" style={{ marginBottom: 16 }}>
                <label style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Model Path (.pth file)
                </label>
                <input
                  className="input"
                  value={modelPath}
                  onChange={e => setModelPath(e.target.value)}
                  placeholder="/path/to/efficientnet_v2_s_best.pth"
                />
              </div>

              <button
                className="btn btn-ghost btn-sm"
                style={{ marginBottom: showAdvanced ? 16 : 0 }}
                onClick={() => setShowAdvanced(!showAdvanced)}
              >
                <Settings2 size={13} />
                {showAdvanced ? 'Hide' : 'Show'} XAI Options
              </button>

              {showAdvanced && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                  <div className="form-group">
                    <label>LIME Samples ({numSamples})</label>
                    <input type="range" min={20} max={500} value={numSamples}
                      onChange={e => setNumSamples(Number(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent)' }} />
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Higher = more accurate but slower
                    </span>
                  </div>
                  <div className="form-group">
                    <label>LIME Features ({numFeatures})</label>
                    <input type="range" min={5} max={100} value={numFeatures}
                      onChange={e => setNumFeatures(Number(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent)' }} />
                  </div>
                  <div className="form-group">
                    <label>Segmentation Algorithm</label>
                    <select className="input" value={segAlg} onChange={e => setSegAlg(e.target.value)}
                      style={{ cursor: 'pointer' }}>
                      <option value="quickshift" style={{ background: 'var(--bg-surface)' }}>Quickshift</option>
                      <option value="slic" style={{ background: 'var(--bg-surface)' }}>SLIC</option>
                    </select>
                  </div>
                </div>
              )}
            </div>

            {error && (
              <div style={{
                display: 'flex', alignItems: 'flex-start', gap: 10, padding: '12px 16px',
                background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)',
                borderRadius: 'var(--radius-md)'
              }}>
                <AlertCircle size={15} color="var(--danger)" style={{ flexShrink: 0, marginTop: 1 }} />
                <span style={{ fontSize: '0.875rem', color: 'var(--danger)', lineHeight: 1.5 }}>{error}</span>
              </div>
            )}

            <button
              className="btn btn-primary btn-lg"
              onClick={handleAnalyze}
              disabled={loading || !imageFile}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {loading ? (
                <><Loader2 size={17} className="spin" style={{ animation: 'spin 0.8s linear infinite' }} /> Analyzing…</>
              ) : (
                <><FlaskConical size={17} /> Analyze Leaf</>
              )}
            </button>
          </div>

          {/* Right: results */}
          <div className="analyze-results">
            {loading && (
              <div className="glass-card" style={{
                padding: '60px 24px', display: 'flex', flexDirection: 'column',
                alignItems: 'center', gap: 16, textAlign: 'center'
              }}>
                <div className="spinner" style={{ width: 40, height: 40, borderWidth: 3 }} />
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  Running inference + generating explanations…<br />
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
                    LIME may take 10–30 seconds depending on sample count.
                  </span>
                </p>
              </div>
            )}
            {result && !loading && (
              <>
                <ResultCard result={result} />
                {(result.gradcam_available || result.lime_available) && (
                  <ExplanationPanel result={result} />
                )}
              </>
            )}
            {!result && !loading && (
              <div className="glass-card" style={{
                padding: '60px 24px', display: 'flex', flexDirection: 'column',
                alignItems: 'center', gap: 16, textAlign: 'center', opacity: 0.6
              }}>
                <FlaskConical size={48} strokeWidth={1} color="var(--text-muted)" />
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                  Upload an image and click Analyze to see results here.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
