import { useState } from 'react';
import { Save, CheckCircle2, Edit3, X, Database, AlertCircle } from 'lucide-react';
import type { ScanResponse } from '../api/client';
import { saveDataPoint } from '../api/client';

interface DataBankDrawerProps {
  scanResult: ScanResponse;
  onSaved: (label: string) => void;
  onDismiss: () => void;
  totalSaved: number;
}

export function DataBankDrawer({ scanResult, onSaved, onDismiss, totalSaved }: DataBankDrawerProps) {
  const { result, scan_id, provider_used, model_used } = scanResult;

  const [label, setLabel] = useState(result.disease_name);
  const [isEditing, setIsEditing] = useState(false);
  const [notes, setNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  const handleSave = async (confirmed: boolean) => {
    setSaving(true);
    setError(null);
    try {
      const res = await saveDataPoint({
        scan_id,
        accepted_label: label.trim().replace(/\s+/g, '_'),
        confirmed,
        notes: notes || undefined,
        vlm_provider: provider_used,
        vlm_model: model_used,
        vlm_prediction: result as unknown as object,
      });
      if (res.status === 'duplicate') {
        setError('This exact image is already in your DataBank.');
        return;
      }
      setSaved(true);
      onSaved(label);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  if (saved) {
    return (
      <div className="glass-card animate-fadeInUp" style={{ padding: 24, borderColor: 'rgba(6,182,212,0.3)' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, textAlign: 'center' }}>
          <div style={{
            width: 48, height: 48, borderRadius: '50%',
            background: 'rgba(6,182,212,0.15)', border: '2px solid var(--accent)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            <CheckCircle2 size={22} color="var(--accent)" />
          </div>
          <div>
            <h3 style={{ fontSize: '1rem', fontWeight: 700, marginBottom: 4 }}>Saved to DataBank!</h3>
            <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
              Label: <strong style={{ color: 'var(--accent)' }}>{label}</strong>
            </p>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: 4 }}>
              {totalSaved + 1} data point{totalSaved + 1 !== 1 ? 's' : ''} collected
            </p>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onDismiss}>
            <X size={13} /> Dismiss
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card animate-fadeInUp" style={{
      padding: 24,
      borderColor: 'rgba(6,182,212,0.2)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 18 }}>
        <div style={{
          width: 36, height: 36, borderRadius: 'var(--radius-md)',
          background: 'linear-gradient(135deg, rgba(59,130,246,0.2), rgba(6,182,212,0.2))',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <Database size={17} color="var(--accent)" />
        </div>
        <div>
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Save to DataBank</h3>
          <p style={{ fontSize: '0.73rem', color: 'var(--text-muted)' }}>
            Build your training dataset
          </p>
        </div>
        <button
          onClick={onDismiss}
          style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}
        >
          <X size={16} />
        </button>
      </div>

      {/* VLM Prediction */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          VLM Predicted Label
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {isEditing ? (
            <input
              className="input"
              value={label}
              onChange={e => setLabel(e.target.value)}
              style={{ flex: 1 }}
              autoFocus
              onBlur={() => setIsEditing(false)}
              onKeyDown={e => { if (e.key === 'Enter') setIsEditing(false); }}
            />
          ) : (
            <>
              <div style={{
                flex: 1, padding: '8px 12px',
                background: 'var(--bg-surface)', border: '1px solid var(--border)',
                borderRadius: 'var(--radius-md)', fontSize: '0.875rem',
                fontFamily: 'var(--font-mono)', color: 'var(--text-primary)',
              }}>
                {label}
              </div>
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setIsEditing(true)}
                title="Correct label"
              >
                <Edit3 size={13} /> Correct
              </button>
            </>
          )}
        </div>
        {label !== result.disease_name && (
          <div style={{
            marginTop: 6, display: 'flex', alignItems: 'center', gap: 6,
            fontSize: '0.72rem', color: 'var(--accent)',
          }}>
            <CheckCircle2 size={11} /> Label corrected from "{result.disease_name}"
          </div>
        )}
      </div>

      {/* Notes */}
      <div className="form-group" style={{ marginBottom: 16 }}>
        <label style={{ fontSize: '0.75rem' }}>Notes (optional)</label>
        <textarea
          className="input"
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Growing conditions, location, additional observations…"
          rows={2}
          style={{ resize: 'vertical', minHeight: 56 }}
        />
      </div>

      {/* Error */}
      {error && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, padding: '8px 12px',
          background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.2)',
          borderRadius: 'var(--radius-md)', marginBottom: 12,
        }}>
          <AlertCircle size={13} color="var(--danger)" />
          <span style={{ fontSize: '0.8rem', color: 'var(--danger)' }}>{error}</span>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', gap: 10 }}>
        <button
          className="btn btn-primary"
          style={{ flex: 1, justifyContent: 'center' }}
          onClick={() => handleSave(true)}
          disabled={saving || !label.trim()}
        >
          {saving ? (
            <><div className="spinner" style={{ width: 14, height: 14 }} /> Saving…</>
          ) : (
            <><Save size={14} /> Save</>
          )}
        </button>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => handleSave(false)}
          disabled={saving}
          title="Save but mark as unconfirmed (uncertain label)"
        >
          Save (unsure)
        </button>
      </div>

      {/* DataBank count */}
      {totalSaved > 0 && (
        <p style={{ marginTop: 12, fontSize: '0.73rem', color: 'var(--text-muted)', textAlign: 'center' }}>
          📊 {totalSaved} data point{totalSaved !== 1 ? 's' : ''} in your DataBank
        </p>
      )}
    </div>
  );
}
