import { useState } from 'react';
import {
  AlertTriangle, CheckCircle2, Leaf, Microscope,
  ChevronDown, ChevronUp, Clock, Shield, Activity
} from 'lucide-react';
import type { LeafScanResult } from '../api/client';

interface InsightCardProps {
  result: LeafScanResult;
  processingMs?: number;
  provider?: string;
  model?: string;
}

const SEVERITY_CONFIG = {
  none:     { label: 'None',     pct: 0,   color: 'var(--accent)',   bg: 'rgba(6,182,212,0.1)' },
  low:      { label: 'Low',      pct: 25,  color: 'var(--info)',     bg: 'rgba(96,165,250,0.1)' },
  moderate: { label: 'Moderate', pct: 50,  color: 'var(--warning)',  bg: 'rgba(251,191,36,0.1)' },
  high:     { label: 'High',     pct: 75,  color: '#fb923c',         bg: 'rgba(251,146,60,0.1)' },
  critical: { label: 'Critical', pct: 100, color: 'var(--danger)',   bg: 'rgba(248,113,113,0.1)' },
};

const URGENCY_STRIPE = {
  none:   'transparent',
  low:    'var(--info)',
  medium: 'var(--warning)',
  high:   'var(--danger)',
};

function SeverityMeter({ severity }: { severity: string }) {
  const cfg = SEVERITY_CONFIG[severity as keyof typeof SEVERITY_CONFIG] ?? SEVERITY_CONFIG.none;
  return (
    <div style={{ flex: 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Severity</span>
        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: cfg.color }}>{cfg.label}</span>
      </div>
      <div style={{ height: 8, background: 'var(--border)', borderRadius: 'var(--radius-full)', overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${cfg.pct}%`,
          background: cfg.color,
          borderRadius: 'var(--radius-full)',
          transition: 'width 1s ease',
          boxShadow: cfg.pct > 0 ? `0 0 8px ${cfg.color}55` : 'none',
        }} />
      </div>
    </div>
  );
}

function AffectedArc({ pct }: { pct: number }) {
  const R = 26;
  const circ = 2 * Math.PI * R;
  const dash = (pct / 100) * circ;
  const color = pct < 25 ? 'var(--accent)' : pct < 50 ? 'var(--warning)' : 'var(--danger)';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
      <svg width="64" height="64" viewBox="0 0 64 64" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="32" cy="32" r={R} fill="none" stroke="var(--border)" strokeWidth="7" />
        <circle cx="32" cy="32" r={R} fill="none" stroke={color} strokeWidth="7"
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`}
          style={{ transition: 'stroke-dasharray 1s ease, stroke 0.3s' }} />
      </svg>
      <div style={{ position: 'absolute', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: 800, color, fontFamily: 'var(--font-mono)' }}>{pct}%</span>
      </div>
      <span style={{ fontSize: '0.65rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Affected</span>
    </div>
  );
}

export function InsightCard({ result, processingMs, provider, model }: InsightCardProps) {
  const [expanded, setExpanded] = useState(false);
  const urgencyColor = URGENCY_STRIPE[result.urgency as keyof typeof URGENCY_STRIPE] ?? 'transparent';

  return (
    <div className="glass-card animate-fadeInUp" style={{ overflow: 'hidden' }}>
      {/* Urgency stripe */}
      {result.urgency !== 'none' && (
        <div style={{
          height: 4, background: urgencyColor,
          boxShadow: `0 0 12px ${urgencyColor}88`,
        }} />
      )}

      <div style={{ padding: '24px 24px 0' }}>
        {/* Header */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              {result.is_diseased ? (
                <AlertTriangle size={16} color="var(--warning)" />
              ) : (
                <CheckCircle2 size={16} color="var(--accent)" />
              )}
              <span style={{ fontSize: '0.7rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                {result.crop_type}
              </span>
            </div>
            <h2 style={{
              fontSize: 'clamp(1.2rem, 3vw, 1.6rem)', fontWeight: 800,
              fontFamily: 'var(--font-heading)',
              background: result.is_diseased
                ? 'linear-gradient(135deg, var(--warning), var(--danger))'
                : 'linear-gradient(135deg, var(--text-primary), var(--accent))',
              WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              lineHeight: 1.2, marginBottom: 4,
            }}>
              {result.disease_name}
            </h2>
            {result.scientific_name && (
              <p style={{ fontSize: '0.8rem', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                {result.scientific_name}
              </p>
            )}
          </div>

          {/* Affected area arc */}
          {result.is_diseased && (
            <div style={{ position: 'relative', width: 64, height: 64, flexShrink: 0 }}>
              <AffectedArc pct={result.affected_area_percent} />
            </div>
          )}
        </div>

        {/* Confidence + Severity */}
        <div style={{ display: 'flex', gap: 16, marginBottom: 20, alignItems: 'center', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Activity size={14} color="var(--text-muted)" />
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Confidence</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 700, color: 'var(--accent)' }}>
              {Math.round(result.confidence * 100)}%
            </span>
          </div>
          <div style={{ flex: 1 }}>
            <SeverityMeter severity={result.severity} />
          </div>
        </div>

        {/* Consume safety badge */}
        {result.is_safe_to_consume != null && (
          <div style={{
            display: 'inline-flex', alignItems: 'center', gap: 6,
            padding: '4px 12px', borderRadius: 'var(--radius-full)',
            marginBottom: 16,
            background: result.is_safe_to_consume ? 'rgba(6,182,212,0.08)' : 'rgba(248,113,113,0.08)',
            border: `1px solid ${result.is_safe_to_consume ? 'rgba(6,182,212,0.2)' : 'rgba(248,113,113,0.2)'}`,
          }}>
            <Shield size={12} color={result.is_safe_to_consume ? 'var(--accent)' : 'var(--danger)'} />
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: result.is_safe_to_consume ? 'var(--accent)' : 'var(--danger)' }}>
              {result.is_safe_to_consume ? 'Safe to consume' : 'Not safe to consume'}
            </span>
          </div>
        )}
      </div>

      {/* Symptoms */}
      {result.symptoms.length > 0 && (
        <div style={{ padding: '0 24px 20px' }}>
          <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
            <Microscope size={13} style={{ marginRight: 6, verticalAlign: 'middle' }} />
            Observed Symptoms
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {result.symptoms.map((s, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'flex-start', gap: 8,
                animation: `fadeInUp 0.3s ease ${i * 0.06}s both`
              }}>
                <span style={{
                  minWidth: 20, height: 20, borderRadius: 'var(--radius-full)',
                  background: 'rgba(251,191,36,0.1)', border: '1px solid rgba(251,191,36,0.2)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.65rem', fontWeight: 700, color: 'var(--warning)', flexShrink: 0,
                }}>{i + 1}</span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{s}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Treatment */}
      {result.treatment.length > 0 && (
        <div style={{
          margin: '0 24px 20px', padding: '16px',
          background: 'rgba(6,182,212,0.04)', borderRadius: 'var(--radius-md)',
          border: '1px solid rgba(6,182,212,0.1)'
        }}>
          <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--accent)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
            💊 Treatment Steps
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {result.treatment.map((t, i) => (
              <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <span style={{
                  minWidth: 22, height: 22, borderRadius: 'var(--radius-full)',
                  background: 'linear-gradient(135deg, var(--info), var(--accent))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.65rem', fontWeight: 800, color: 'var(--text-inverse)', flexShrink: 0,
                }}>{i + 1}</span>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.5 }}>{t}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Expandable: Explanation + Prevention */}
      <div style={{ padding: '0 24px 20px' }}>
        <button
          onClick={() => setExpanded(!expanded)}
          style={{
            display: 'flex', alignItems: 'center', gap: 6,
            background: 'none', border: 'none', cursor: 'pointer',
            color: 'var(--text-muted)', fontSize: '0.8rem', fontWeight: 600,
            padding: '6px 0', fontFamily: 'var(--font-body)',
          }}
        >
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {expanded ? 'Hide' : 'Show'} full explanation & prevention
        </button>

        {expanded && (
          <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', gap: 16, animation: 'fadeInUp 0.3s ease' }}>
            {result.explanation && (
              <div>
                <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                  🔬 Explanation
                </h4>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.7, fontStyle: 'italic' }}>
                  "{result.explanation}"
                </p>
              </div>
            )}
            {result.prevention.length > 0 && (
              <div>
                <h4 style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>
                  🛡 Prevention
                </h4>
                <ul style={{ display: 'flex', flexDirection: 'column', gap: 5, listStyle: 'none' }}>
                  {result.prevention.map((p, i) => (
                    <li key={i} style={{ display: 'flex', gap: 8, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      <Leaf size={13} color="var(--info)" style={{ flexShrink: 0, marginTop: 3 }} />
                      {p}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      {(processingMs || provider) && (
        <div style={{
          borderTop: '1px solid var(--border)', padding: '10px 24px',
          display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap'
        }}>
          {provider && (
            <span style={{ fontSize: '0.72rem', fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
              via {provider} / {model}
            </span>
          )}
          {processingMs && (
            <span style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: 'auto' }}>
              <Clock size={11} /> {(processingMs / 1000).toFixed(1)}s
            </span>
          )}
        </div>
      )}
    </div>
  );
}
