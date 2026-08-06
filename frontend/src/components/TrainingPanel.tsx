import { useState, useEffect, useRef } from 'react';
import { Play, RefreshCw, AlertCircle } from 'lucide-react';
import { startTraining, getTrainStatus, type TrainStatusResponse } from '../api/client';

const BACKBONES = [
  { id: 'efficientnet_v2_s', name: 'EfficientNetV2-S', tag: 'Recommended' },
  { id: 'efficientnet_v2_m', name: 'EfficientNetV2-M', tag: 'High accuracy' },
  { id: 'resnet50', name: 'ResNet-50', tag: 'Classic' },
  { id: 'resnet101', name: 'ResNet-101', tag: 'Deeper' },
  { id: 'densenet121', name: 'DenseNet-121', tag: 'Dense connections' },
  { id: 'mobilenet_v3_large', name: 'MobileNetV3-L', tag: 'Fast / edge' },
  { id: 'vgg16', name: 'VGG-16', tag: 'Legacy' },
  { id: 'vit_base_patch16_224', name: 'ViT-B/16', tag: 'Transformer' },
];

const OPTIMIZERS = ['adamw', 'adam', 'sgd'];

export function TrainingPanel() {
  const [dataPath, setDataPath] = useState('');
  const [modelDir, setModelDir] = useState('./models');
  const [logDir, setLogDir] = useState('./logs');
  const [selectedBackbones, setSelectedBackbones] = useState<string[]>(['efficientnet_v2_s']);
  const [epochs, setEpochs] = useState(30);
  const [batchSize, setBatchSize] = useState(32);
  const [lr, setLr] = useState(0.001);
  const [optimizer, setOptimizer] = useState('adamw');
  const [patience, setPatience] = useState(7);
  const [imageSize, setImageSize] = useState(224);

  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<TrainStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const toggleBackbone = (id: string) => {
    setSelectedBackbones(prev =>
      prev.includes(id) ? prev.filter(b => b !== id) : [...prev, id]
    );
  };

  const startPoll = (id: string) => {
    pollRef.current = setInterval(async () => {
      const s = await getTrainStatus(id);
      setStatus(s);
      if (s.status === 'completed' || s.status === 'failed') {
        clearInterval(pollRef.current!);
        pollRef.current = null;
      }
    }, 2000);
  };

  const handleStart = async () => {
    if (!dataPath) { setError('Dataset path is required.'); return; }
    if (selectedBackbones.length === 0) { setError('Select at least one backbone.'); return; }
    setError(null);
    setLoading(true);
    try {
      const res = await startTraining({
        data_path: dataPath,
        model_dir: modelDir,
        log_dir: logDir,
        backbones: selectedBackbones,
        epochs,
        batch_size: batchSize,
        learning_rate: lr,
        optimizer,
        image_size: imageSize,
        patience,
      });
      setJobId(res.job_id);
      startPoll(res.job_id);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Failed to start training');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    clearInterval(pollRef.current!);
    pollRef.current = null;
    setJobId(null);
    setStatus(null);
    setError(null);
  };

  useEffect(() => () => { clearInterval(pollRef.current!); }, []);

  const isRunning = status?.status === 'running' || status?.status === 'queued';
  const progress = status?.current_epoch && status?.total_epochs
    ? (status.current_epoch / status.total_epochs) * 100 : 0;

  return (
    <div className="glass-card training-panel">
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ marginBottom: 6 }}>No-Code Model Training</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>
          Select backbones, configure hyperparameters, and train — no code required.
        </p>
      </div>

      {/* Backbone selection */}
      <div style={{ marginBottom: 24 }}>
        <label className="form-group" style={{ marginBottom: 12 }}>
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Model Architectures
          </span>
        </label>
        <div className="training-panel__grid">
          {BACKBONES.map(b => (
            <div
              key={b.id}
              className={`backbone-chip${selectedBackbones.includes(b.id) ? ' selected' : ''}`}
              onClick={() => toggleBackbone(b.id)}
            >
              <div style={{
                width: 18, height: 18, borderRadius: 4,
                border: `2px solid ${selectedBackbones.includes(b.id) ? 'var(--accent)' : 'var(--border)'}`,
                background: selectedBackbones.includes(b.id) ? 'var(--accent)' : 'transparent',
                display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                transition: 'all 0.15s'
              }}>
                {selectedBackbones.includes(b.id) && (
                  <svg width="10" height="10" viewBox="0 0 10 10">
                    <path d="M1.5 5L4 7.5 8.5 2.5" stroke="var(--text-inverse)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                  </svg>
                )}
              </div>
              <div>
                <div className="backbone-chip__name">{b.name}</div>
                <div className="backbone-chip__tag">{b.tag}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Paths */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 24 }}>
        {[
          { label: 'Dataset Path (train/val/test subdirs)', value: dataPath, set: setDataPath, placeholder: '/path/to/dataset' },
          { label: 'Model Save Directory', value: modelDir, set: setModelDir, placeholder: './models' },
          { label: 'Log Directory', value: logDir, set: setLogDir, placeholder: './logs' },
        ].map(({ label, value, set, placeholder }) => (
          <div key={label} className="form-group">
            <label>{label}</label>
            <input className="input" value={value} onChange={e => set(e.target.value)} placeholder={placeholder} />
          </div>
        ))}
      </div>

      {/* Hyperparameters */}
      <div className="training-panel__hyperparams">
        <div className="form-group">
          <label>Epochs</label>
          <input className="input" type="number" value={epochs} min={1} max={500}
            onChange={e => setEpochs(Number(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Batch Size</label>
          <input className="input" type="number" value={batchSize} min={1} max={256}
            onChange={e => setBatchSize(Number(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Learning Rate</label>
          <input className="input" type="number" value={lr} step={0.0001} min={0.00001}
            onChange={e => setLr(Number(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Early Stop Patience</label>
          <input className="input" type="number" value={patience} min={1} max={100}
            onChange={e => setPatience(Number(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Image Size (px)</label>
          <input className="input" type="number" value={imageSize} min={32} max={512}
            onChange={e => setImageSize(Number(e.target.value))} />
        </div>
        <div className="form-group">
          <label>Optimizer</label>
          <select className="input" value={optimizer} onChange={e => setOptimizer(e.target.value)}
            style={{ cursor: 'pointer' }}>
            {OPTIMIZERS.map(o => (
              <option key={o} value={o} style={{ background: 'var(--bg-surface)' }}>{o.toUpperCase()}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '10px 14px',
          background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)',
          borderRadius: 'var(--radius-md)', marginBottom: 16
        }}>
          <AlertCircle size={14} color="var(--danger)" />
          <span style={{ fontSize: '0.85rem', color: 'var(--danger)' }}>{error}</span>
        </div>
      )}

      {/* Status */}
      {status && (
        <div style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`badge badge-${
                status.status === 'completed' ? 'success' :
                status.status === 'failed' ? 'danger' :
                status.status === 'running' ? 'info' : 'warning'
              }`}>
                {status.status.toUpperCase()}
              </span>
              {status.job_id && (
                <span style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                  job:{status.job_id}
                </span>
              )}
            </div>
            {status.current_epoch != null && (
              <span style={{ fontSize: '0.8rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                Epoch {status.current_epoch} / {status.total_epochs}
              </span>
            )}
          </div>
          <div className="progress-bar" style={{ marginBottom: 12 }}>
            <div className="progress-fill" style={{ width: `${progress}%` }} />
          </div>
          <div className="training-panel__log">
            {status.train_loss != null && (
              <div className={`log-entry ${status.status}`}>
                › Epoch {status.current_epoch} — loss: {status.train_loss?.toFixed(4)} | acc: {((status.train_acc ?? 0)*100).toFixed(1)}% | val_acc: {((status.val_acc ?? 0)*100).toFixed(1)}% | best: {((status.best_val_acc ?? 0)*100).toFixed(1)}%
              </div>
            )}
            {status.status === 'completed' && (
              <div className="log-entry completed">
                ✓ Training complete. Best val accuracy: {((status.best_val_acc ?? 0)*100).toFixed(2)}%
              </div>
            )}
            {status.message && status.status !== 'completed' && (
              <div className={`log-entry ${status.status}`}>› {status.message}</div>
            )}
          </div>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: 12, justifyContent: 'flex-end' }}>
        {jobId && (
          <button className="btn btn-ghost" onClick={handleReset} disabled={isRunning}>
            <RefreshCw size={15} /> Reset
          </button>
        )}
        <button
          className="btn btn-primary btn-lg"
          onClick={handleStart}
          disabled={loading || isRunning}
        >
          {loading || isRunning ? (
            <><div className="spinner" style={{ width: 16, height: 16 }} /> Training…</>
          ) : (
            <><Play size={16} /> Start Training</>
          )}
        </button>
      </div>
    </div>
  );
}
