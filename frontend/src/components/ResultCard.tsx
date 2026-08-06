import { AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react';
import type { PredictionResponse } from '../api/client';

interface ResultCardProps {
  result: PredictionResponse;
}

function ConfidenceGauge({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = pct >= 80 ? 'var(--accent)' : pct >= 50 ? 'var(--warning)' : 'var(--danger)';
  return (
    <div className="result-card__confidence-ring" title={`Confidence: ${pct}%`}>
      <svg width="72" height="72" viewBox="0 0 72 72" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="36" cy="36" r="28" fill="none" stroke="var(--border)" strokeWidth="6" />
        <circle
          cx="36" cy="36" r="28"
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={`${2 * Math.PI * 28}`}
          strokeDashoffset={`${2 * Math.PI * 28 * (1 - value)}`}
          style={{ transition: 'stroke-dashoffset 1s ease, stroke 0.3s' }}
        />
      </svg>
      <div style={{ position: 'absolute', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <span className="result-card__confidence-value" style={{ color }}>{pct}%</span>
      </div>
    </div>
  );
}

export function ResultCard({ result }: ResultCardProps) {
  const { label, confidence, all_class_probs } = result;
  const sorted = Object.entries(all_class_probs)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8);

  const maxProb = sorted[0]?.[1] ?? 1;

  const barColors = [
    'linear-gradient(90deg, var(--green-400), var(--accent))',
    'linear-gradient(90deg, var(--info), #93c5fd)',
    'linear-gradient(90deg, var(--warning), #fde68a)',
    'linear-gradient(90deg, var(--danger), #fca5a5)',
  ];

  return (
    <div className="glass-card result-card animate-fadeInUp">
      <div className="result-card__header">
        <div>
          <div style={{ marginBottom: 6 }}>
            <CheckCircle2 size={16} color="var(--accent)" style={{ marginRight: 6 }} />
            <span className="text-muted" style={{ fontSize: '0.78rem', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Diagnosis</span>
          </div>
          <h2 className="result-card__label">{label.replace(/_/g, ' ')}</h2>
        </div>
        <div style={{ position: 'relative', width: 72, height: 72 }}>
          <ConfidenceGauge value={confidence} />
          <div style={{
            position: 'absolute', inset: 0, display: 'flex',
            flexDirection: 'column', alignItems: 'center', justifyContent: 'center'
          }}>
            <span className="result-card__confidence-value" style={{
              color: confidence >= 0.8 ? 'var(--accent)' : confidence >= 0.5 ? 'var(--warning)' : 'var(--danger)'
            }}>
              {Math.round(confidence * 100)}%
            </span>
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
        <TrendingUp size={14} color="var(--text-muted)" />
        <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
          Class Probabilities
        </span>
      </div>

      <div className="prob-list">
        {sorted.map(([cls, prob], idx) => (
          <div key={cls} className="prob-item">
            <div className="prob-item__label">
              <span className="prob-item__name">{cls.replace(/_/g, ' ')}</span>
              <span className="prob-item__pct">{(prob * 100).toFixed(1)}%</span>
            </div>
            <div className="prob-item__bar">
              <div
                className="prob-item__fill"
                style={{
                  width: `${(prob / maxProb) * 100}%`,
                  background: barColors[idx % barColors.length],
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {confidence < 0.6 && (
        <div style={{
          marginTop: 20, padding: '10px 14px',
          background: 'rgba(251,191,36,0.08)', border: '1px solid rgba(251,191,36,0.2)',
          borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'center', gap: 8
        }}>
          <AlertCircle size={14} color="var(--warning)" />
          <span style={{ fontSize: '0.8rem', color: 'var(--warning)' }}>
            Low confidence — consider retraining with more data.
          </span>
        </div>
      )}
    </div>
  );
}
