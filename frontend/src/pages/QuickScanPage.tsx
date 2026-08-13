import { useState, useCallback } from 'react';
import { Zap, AlertCircle, Database, ChevronRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ImageUploader } from '../components/ImageUploader';
import { ProviderPicker } from '../components/ProviderPicker';
import { InsightCard } from '../components/InsightCard';
import { DataBankDrawer } from '../components/DataBankDrawer';
import { scanLeaf, getDataBankStats, type ScanResponse } from '../api/client';

export function QuickScanPage() {
  // Upload state
  const [imageFile, setImageFile] = useState<File | null>(null);

  // Provider state
  const [provider, setProvider] = useState('gemini');
  const [apiKey, setApiKey] = useState('');
  const [ollamaHost, setOllamaHost] = useState('http://localhost:11434');

  // Scan state
  const [scanResult, setScanResult] = useState<ScanResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // DataBank state
  const [showDrawer, setShowDrawer] = useState(false);
  const [savedCount, setSavedCount] = useState(0);
  const [bankStats, setBankStats] = useState<{ total: number; can_train: boolean } | null>(null);

  const fetchBankStats = useCallback(async () => {
    try {
      const s = await getDataBankStats();
      setBankStats({ total: s.total_points, can_train: s.can_train });
    } catch {/* backend might not be up */}
  }, []);

  useState(() => { fetchBankStats(); });

  const handleScan = async () => {
    if (!imageFile) { setError('Please upload a leaf image first.'); return; }
    if (provider !== 'ollama' && !apiKey) {
      setError(`Please enter your ${provider.charAt(0).toUpperCase() + provider.slice(1)} API key.`);
      return;
    }
    setError(null);
    setScanResult(null);
    setShowDrawer(false);
    setLoading(true);

    try {
      const res = await scanLeaf(
        imageFile,
        provider,
        provider === 'ollama' ? null : apiKey,
        undefined,
        ollamaHost,
      );
      setScanResult(res);
    } catch (e: any) {
      const detail = e.response?.data?.detail || e.message;
      setError(
        detail?.includes('API key') || detail?.includes('401')
          ? 'Invalid API key. Please check and try again.'
          : detail || 'Scan failed. Check your connection and API key.'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSaved = (_label: string) => {
    setSavedCount(c => c + 1);
    fetchBankStats();
  };

  return (
    <div className="page">
      <div className="container">
        <div className="page__header">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <Zap size={22} color="var(--accent)" />
              <div>
                <h1 className="page__title" style={{ fontSize: '1.8rem' }}>Quick Scan</h1>
                <p className="page__subtitle">
                  Zero-shot plant disease analysis — no training data needed
                </p>
              </div>
            </div>
            {bankStats && bankStats.total > 0 && (
              <Link to="/databank" className="btn btn-secondary btn-sm" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Database size={13} />
                {bankStats.total} in DataBank
                {bankStats.can_train && <span className="badge badge-success" style={{ marginLeft: 4 }}>Ready to train</span>}
                <ChevronRight size={13} />
              </Link>
            )}
          </div>
        </div>

        {/* How it works banner */}
        <div style={{
          display: 'flex', gap: 0, marginBottom: 32,
          background: 'var(--bg-surface)', borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--border)', overflow: 'hidden',
        }}>
          {[
            { step: '1', text: 'Upload leaf photo', icon: '📸' },
            { step: '2', text: 'VLM analyses instantly', icon: '🤖' },
            { step: '3', text: 'Accept or correct label', icon: '✅' },
            { step: '4', text: 'Collect → train your model', icon: '🚀' },
          ].map((s, i) => (
            <div key={s.step} style={{
              flex: 1, padding: '14px 16px', textAlign: 'center',
              borderRight: i < 3 ? '1px solid var(--border)' : 'none',
            }}>
              <div style={{ fontSize: '1.1rem', marginBottom: 4 }}>{s.icon}</div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Step {s.step}</div>
              <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 2 }}>{s.text}</div>
            </div>
          ))}
        </div>

        {/* Main two-column layout */}
        <div style={{ display: 'grid', gridTemplateColumns: '380px 1fr', gap: 24, alignItems: 'start' }}>
          {/* Left: Config */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Image upload */}
            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>Leaf Image</h3>
              <ImageUploader
                onFileSelect={setImageFile}
                selectedFile={imageFile}
                onClear={() => { setImageFile(null); setScanResult(null); setShowDrawer(false); }}
              />
            </div>

            {/* Provider picker */}
            <div className="glass-card" style={{ padding: 24 }}>
              <h3 style={{ marginBottom: 16, fontSize: '1rem' }}>AI Provider</h3>
              <ProviderPicker
                selectedProvider={provider}
                onProviderChange={p => { setProvider(p); setScanResult(null); }}
                apiKey={apiKey}
                onApiKeyChange={setApiKey}
                ollamaHost={ollamaHost}
                onOllamaHostChange={setOllamaHost}
              />
            </div>

            {/* Error */}
            {error && (
              <div style={{
                display: 'flex', alignItems: 'flex-start', gap: 10, padding: '12px 16px',
                background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.25)',
                borderRadius: 'var(--radius-md)',
              }}>
                <AlertCircle size={15} color="var(--danger)" style={{ flexShrink: 0, marginTop: 1 }} />
                <span style={{ fontSize: '0.875rem', color: 'var(--danger)', lineHeight: 1.5 }}>{error}</span>
              </div>
            )}

            {/* Scan button */}
            <button
              className="btn btn-primary btn-lg"
              onClick={handleScan}
              disabled={loading || !imageFile}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {loading ? (
                <><div className="spinner" style={{ width: 17, height: 17 }} /> Scanning…</>
              ) : (
                <><Zap size={17} /> Scan Now</>
              )}
            </button>
          </div>

          {/* Right: Results */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {loading && (
              <div className="glass-card" style={{
                padding: '60px 24px', display: 'flex', flexDirection: 'column',
                alignItems: 'center', gap: 20, textAlign: 'center',
              }}>
                <div style={{ position: 'relative', width: 64, height: 64 }}>
                  <div className="spinner" style={{ width: 64, height: 64, borderWidth: 4 }} />
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.5rem' }}>🌿</div>
                </div>
                <div>
                  <h3 style={{ marginBottom: 8 }}>Analysing leaf…</h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                    The VLM is examining symptoms, severity, and treatment options.
                    <br />This typically takes 3–10 seconds.
                  </p>
                </div>
              </div>
            )}

            {scanResult && !loading && (
              <>
                <InsightCard
                  result={scanResult.result}
                  processingMs={scanResult.processing_time_ms}
                  provider={scanResult.provider_used}
                  model={scanResult.model_used}
                />

                {/* DataBank CTA */}
                {!showDrawer && (
                  <div className="glass-card" style={{
                    padding: 20,
                    background: 'linear-gradient(135deg, rgba(6,182,212,0.05), rgba(15,23,42,0.8))',
                    borderColor: 'rgba(6,182,212,0.2)',
                    display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
                  }}>
                    <div style={{ flex: 1 }}>
                      <h4 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: 4 }}>
                        💾 Save this scan to DataBank
                      </h4>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        Accept or correct the label, then save it as a training data point.
                        Collect 30+ per class to unlock your own private model.
                      </p>
                    </div>
                    <button
                      className="btn btn-primary"
                      onClick={() => setShowDrawer(true)}
                    >
                      <Database size={14} /> Save to DataBank
                    </button>
                  </div>
                )}

                {showDrawer && (
                  <DataBankDrawer
                    scanResult={scanResult}
                    onSaved={handleSaved}
                    onDismiss={() => setShowDrawer(false)}
                    totalSaved={savedCount + (bankStats?.total ?? 0)}
                  />
                )}
              </>
            )}

            {!scanResult && !loading && (
              <div className="glass-card" style={{
                padding: '60px 24px', display: 'flex', flexDirection: 'column',
                alignItems: 'center', gap: 16, textAlign: 'center', opacity: 0.6,
              }}>
                <Zap size={48} strokeWidth={1} color="var(--text-muted)" />
                <div>
                  <h3 style={{ marginBottom: 8, opacity: 0.8 }}>Ready to scan</h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                    Upload a leaf image and choose your AI provider to get an instant diagnosis.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
