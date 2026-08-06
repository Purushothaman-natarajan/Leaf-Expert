import { useState } from 'react';
import { ScanLine, Grid3x3, Info } from 'lucide-react';
import type { ExplainResponse } from '../api/client';

interface ExplanationPanelProps {
  result: ExplainResponse;
}

type Tab = 'gradcam' | 'lime' | 'original';

export function ExplanationPanel({ result }: ExplanationPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>('gradcam');

  const tabs: { key: Tab; label: string; icon: React.ReactNode; available: boolean }[] = [
    {
      key: 'gradcam',
      label: 'Grad-CAM++',
      icon: <ScanLine size={13} />,
      available: result.gradcam_available,
    },
    {
      key: 'lime',
      label: 'LIME',
      icon: <Grid3x3 size={13} />,
      available: result.lime_available,
    },
  ];

  const descriptions: Record<Tab, string> = {
    gradcam: 'Grad-CAM++ highlights regions the model focused on. Warmer colours indicate higher influence on the prediction.',
    lime: 'LIME identifies superpixel regions that positively contributed to the predicted class (highlighted in orange).',
    original: 'Original uploaded image.',
  };

  const imgSrc: Record<Tab, string> = {
    gradcam: result.gradcam_b64 ? `data:image/png;base64,${result.gradcam_b64}` : '',
    lime: result.lime_b64 ? `data:image/png;base64,${result.lime_b64}` : '',
    original: '',
  };

  return (
    <div className="glass-card explanation-panel animate-fadeInUp" style={{ animationDelay: '0.1s' }}>
      <div style={{ marginBottom: 16 }}>
        <h3 style={{ fontSize: '1rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: 4 }}>
          XAI Explanations
        </h3>
        <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          Visual explanations for the model's decision
        </p>
      </div>

      <div className="explanation-panel__tabs">
        {tabs.map(tab => (
          <button
            key={tab.key}
            className={`explanation-tab${activeTab === tab.key ? ' active' : ''}`}
            onClick={() => setActiveTab(tab.key)}
            disabled={!tab.available}
            style={{ opacity: tab.available ? 1 : 0.4, cursor: tab.available ? 'pointer' : 'not-allowed' }}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              {tab.icon} {tab.label}
            </span>
          </button>
        ))}
      </div>

      <div className="explanation-panel__image">
        {imgSrc[activeTab] ? (
          <img src={imgSrc[activeTab]} alt={`${activeTab} explanation`} />
        ) : (
          <div style={{
            height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexDirection: 'column', gap: 12, color: 'var(--text-muted)', fontSize: '0.85rem'
          }}>
            <ScanLine size={32} opacity={0.4} />
            <span>Explanation not available</span>
          </div>
        )}
      </div>

      <div className="explanation-panel__badge">
        <Info size={12} color="var(--text-muted)" />
        <p style={{ fontSize: '0.775rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>
          {descriptions[activeTab]}
        </p>
      </div>
    </div>
  );
}
