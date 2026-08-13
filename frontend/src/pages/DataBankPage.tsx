import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Database, Trash2, FolderOutput, RefreshCw, AlertCircle, Sparkles, CheckCircle2 } from 'lucide-react';
import {
  listDataPoints, getDataBankStats, deleteDataPoint, exportDataset,
  type DataPointRecord, type DataBankStats
} from '../api/client';

export function DataBankPage() {
  const navigate = useNavigate();
  const [points, setPoints] = useState<DataPointRecord[]>([]);
  const [stats, setStats] = useState<DataBankStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [filterLabel, setFilterLabel] = useState<string>('');
  const [confirmedOnly, setConfirmedOnly] = useState(true);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [pts, st] = await Promise.all([
        listDataPoints({ label: filterLabel || undefined, confirmed_only: confirmedOnly, limit: 500 }),
        getDataBankStats(),
      ]);
      setPoints(pts);
      setStats(st);
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message || 'Failed to load DataBank');
    } finally {
      setLoading(false);
    }
  }, [filterLabel, confirmedOnly]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this data point permanently?')) return;
    try {
      await deleteDataPoint(id);
      fetchData();
    } catch (e: any) {
      alert(e.response?.data?.detail || 'Failed to delete');
    }
  };

  const handleExportAndTrain = async () => {
    setExporting(true);
    try {
      const res = await exportDataset({
        target_dir: 'data/from-databank',
        confirmed_only: true,
      });
      // Redirect to train page, optionally passing the dataset path via state
      navigate('/train', { state: { datasetPath: res.target_dir, message: res.message } });
    } catch (e: any) {
      setError(e.response?.data?.detail || 'Failed to export dataset');
      setExporting(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        {/* Header */}
        <div className="page__header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 'var(--radius-lg)',
              background: 'linear-gradient(135deg, rgba(59,130,246,0.2), rgba(6,182,212,0.2))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 8px 16px rgba(6,182,212,0.2)',
            }}>
              <Database size={24} color="var(--accent)" />
            </div>
            <div>
              <h1 className="page__title">DataBank</h1>
              <p className="page__subtitle">Your private, curated training dataset</p>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            <button className="btn btn-secondary" onClick={fetchData} disabled={loading}>
              <RefreshCw size={14} className={loading ? 'spin' : ''} /> Refresh
            </button>
            <button
              className="btn btn-primary"
              onClick={handleExportAndTrain}
              disabled={exporting || !stats?.can_train}
              style={{
                background: stats?.can_train
                  ? 'linear-gradient(135deg, var(--info), var(--accent))'
                  : 'var(--bg-surface)',
                color: stats?.can_train ? 'var(--text-inverse)' : 'var(--text-muted)',
              }}
            >
              {exporting ? (
                <><div className="spinner" style={{ width: 14, height: 14 }} /> Exporting…</>
              ) : (
                <><Sparkles size={14} /> Export & Train Model</>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div style={{
            padding: 16, background: 'rgba(248,113,113,0.1)', border: '1px solid var(--danger)',
            borderRadius: 'var(--radius-md)', color: 'var(--danger)', marginBottom: 24,
            display: 'flex', alignItems: 'center', gap: 10
          }}>
            <AlertCircle size={16} /> {error}
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: 24, alignItems: 'start' }}>
          {/* Main List */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {/* Filters */}
            <div className="glass-card" style={{ padding: '16px 20px', display: 'flex', gap: 20, alignItems: 'center' }}>
              <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 10 }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>FILTER CLASS:</span>
                <select
                  className="input"
                  style={{ flex: 1, height: 36 }}
                  value={filterLabel}
                  onChange={e => setFilterLabel(e.target.value)}
                >
                  <option value="">All Classes</option>
                  {stats && Object.keys(stats.class_distribution).map(lbl => (
                    <option key={lbl} value={lbl}>{lbl} ({stats.class_distribution[lbl]})</option>
                  ))}
                </select>
              </div>
              <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '0.85rem', cursor: 'pointer' }}>
                <input
                  type="checkbox"
                  checked={confirmedOnly}
                  onChange={e => setConfirmedOnly(e.target.checked)}
                  style={{ accentColor: 'var(--accent)' }}
                />
                Confirmed labels only
              </label>
            </div>

            {/* Gallery */}
            {loading ? (
              <div style={{ padding: 60, textAlign: 'center', color: 'var(--text-muted)' }}>
                <div className="spinner" style={{ margin: '0 auto 16px' }} /> Loading DataBank...
              </div>
            ) : points.length === 0 ? (
              <div className="glass-card" style={{ padding: 60, textAlign: 'center' }}>
                <Database size={32} color="var(--text-muted)" style={{ margin: '0 auto 16px', opacity: 0.5 }} />
                <h3>No data points found</h3>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginTop: 8 }}>
                  Go to Quick Scan to start collecting training data.
                </p>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 16 }}>
                {points.map(pt => (
                  <div key={pt.id} className="glass-card animate-fadeInUp" style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
                    <div style={{ position: 'relative', height: 160, background: '#111' }}>
                      <img
                        src={`${import.meta.env.VITE_API_URL}${pt.image_url}`}
                        alt={pt.user_label}
                        style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                        loading="lazy"
                      />
                      {pt.used_for_training && (
                        <div style={{
                          position: 'absolute', top: 8, left: 8,
                          background: 'rgba(6,182,212,0.9)', color: '#000',
                          padding: '2px 8px', borderRadius: 4, fontSize: '0.65rem', fontWeight: 800,
                          textTransform: 'uppercase', letterSpacing: '0.05em'
                        }}>
                          Trained
                        </div>
                      )}
                      {!pt.confirmed && (
                        <div style={{
                          position: 'absolute', top: 8, right: 8,
                          background: 'rgba(251,191,36,0.9)', color: '#000',
                          padding: '2px 8px', borderRadius: 4, fontSize: '0.65rem', fontWeight: 800,
                        }}>
                          Unsure
                        </div>
                      )}
                    </div>
                    <div style={{ padding: '12px 16px', flex: 1, display: 'flex', flexDirection: 'column' }}>
                      <div style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--accent)', marginBottom: 4 }}>
                        {pt.user_label}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 12, display: 'flex', gap: 6 }}>
                        <span>via {pt.vlm_provider}</span>
                        <span>•</span>
                        <span>{new Date(pt.collected_at).toLocaleDateString()}</span>
                      </div>
                      <div style={{ marginTop: 'auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.7rem', color: pt.disease_name !== pt.user_label ? 'var(--warning)' : 'var(--text-muted)' }} title={`Original prediction: ${pt.disease_name}`}>
                          Orig: {pt.disease_name.substring(0, 15)}...
                        </span>
                        <button
                          className="btn btn-ghost btn-sm"
                          style={{ padding: 4, height: 28, width: 28, color: 'var(--danger)' }}
                          onClick={() => handleDelete(pt.id)}
                          title="Delete"
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Sidebar: Stats */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            {stats && (
              <div className="glass-card" style={{ padding: 24 }}>
                <h3 style={{ fontSize: '1.1rem', marginBottom: 20, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <FolderOutput size={18} color="var(--accent)" />
                  Training Readiness
                </h3>

                <div style={{
                  display: 'flex', gap: 20, marginBottom: 24, paddingBottom: 20,
                  borderBottom: '1px solid var(--border)'
                }}>
                  <div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--text-primary)', lineHeight: 1 }}>{stats.confirmed_points}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 4 }}>Valid points</div>
                  </div>
                  <div>
                    <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--accent)', lineHeight: 1 }}>{stats.classes_ready.length}</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: 4 }}>Ready classes</div>
                  </div>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 12 }}>
                    Class Progress (Min {stats.training_threshold})
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                    {Object.entries(stats.class_distribution).map(([lbl, count]) => {
                      const pct = Math.min(100, (count / stats.training_threshold) * 100);
                      const isReady = count >= stats.training_threshold;
                      return (
                        <div key={lbl}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 6 }}>
                            <span style={{ color: isReady ? 'var(--accent)' : 'var(--text-primary)', fontWeight: isReady ? 600 : 400 }}>{lbl}</span>
                            <span style={{ fontFamily: 'var(--font-mono)' }}>{count} / {stats.training_threshold}</span>
                          </div>
                          <div style={{ height: 6, background: 'var(--border)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{
                              height: '100%', width: `${pct}%`,
                              background: isReady ? 'var(--accent)' : 'var(--warning)',
                              borderRadius: 3
                            }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {stats.can_train ? (
                  <div style={{
                    padding: 12, background: 'rgba(6,182,212,0.1)', border: '1px solid var(--accent)',
                    borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'flex-start', gap: 10
                  }}>
                    <CheckCircle2 size={16} color="var(--accent)" style={{ flexShrink: 0, marginTop: 2 }} />
                    <span style={{ fontSize: '0.85rem', color: 'var(--accent)', lineHeight: 1.4 }}>
                      You have enough data in at least two classes to train a model!
                    </span>
                  </div>
                ) : (
                  <div style={{
                    padding: 12, background: 'rgba(251,191,36,0.1)', border: '1px solid var(--warning)',
                    borderRadius: 'var(--radius-md)', display: 'flex', alignItems: 'flex-start', gap: 10
                  }}>
                    <AlertCircle size={16} color="var(--warning)" style={{ flexShrink: 0, marginTop: 2 }} />
                    <span style={{ fontSize: '0.85rem', color: 'var(--warning)', lineHeight: 1.4 }}>
                      Need at least 2 classes with {stats.training_threshold}+ images each to unlock training.
                    </span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
