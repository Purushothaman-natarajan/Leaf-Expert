import { Cpu } from 'lucide-react';
import { TrainingPanel } from '../components/TrainingPanel';

export function TrainPage() {
  return (
    <div className="page">
      <div className="container">
        <div className="page__header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Cpu size={22} color="var(--accent)" />
            <div>
              <h1 className="page__title" style={{ fontSize: '1.8rem' }}>Model Training</h1>
              <p className="page__subtitle">
                No-code PyTorch transfer learning — select a backbone, point to your dataset, and train.
              </p>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gap: 24, gridTemplateColumns: '1fr 320px', alignItems: 'start' }}>
          <TrainingPanel />

          {/* Tips sidebar */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div className="glass-card" style={{ padding: 20 }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: 14 }}>💡 Training Tips</h3>
              {[
                { t: 'Dataset structure', d: 'train/, val/, test/ — each containing class subdirectories.' },
                { t: 'EfficientNetV2-S', d: 'Best accuracy/speed tradeoff for most leaf datasets.' },
                { t: 'Image size', d: '224px works for most backbones. ViT-B/16 requires exactly 224px.' },
                { t: 'Epochs', d: 'Early stopping fires if val accuracy doesn\'t improve for patience epochs.' },
                { t: 'GPU', d: 'Mixed precision (FP16) is auto-enabled on CUDA. ~3x faster than CPU.' },
              ].map(tip => (
                <div key={tip.t} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 3 }}>
                    {tip.t}
                  </div>
                  <div style={{ fontSize: '0.775rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{tip.d}</div>
                </div>
              ))}
            </div>

            <div className="glass-card" style={{ padding: 20 }}>
              <h3 style={{ fontSize: '0.9rem', marginBottom: 14 }}>📁 Expected Dataset</h3>
              <pre style={{
                fontSize: '0.72rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)',
                lineHeight: 1.8, background: 'var(--bg-base)', padding: 12,
                borderRadius: 'var(--radius-md)', overflowX: 'auto'
              }}>
{`dataset/
├── train/
│   ├── healthy/
│   │   └── *.jpg
│   └── rust/
│       └── *.jpg
├── val/
│   ├── healthy/
│   └── rust/
└── test/
    ├── healthy/
    └── rust/`}
              </pre>
              <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)', marginTop: 10 }}>
                Use the <strong style={{ color: 'var(--accent)' }}>Data Prep API</strong> endpoint to auto-split a raw dataset.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
