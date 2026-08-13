import { useState, useEffect } from 'react';
import { Key, Eye, EyeOff, Server } from 'lucide-react';
import { getProviders, type ProviderInfo } from '../api/client';

const COST_COLORS = {
  free:   { bg: 'rgba(6,182,212,0.1)',  text: 'var(--accent)',   label: 'Free' },
  low:    { bg: 'rgba(96,165,250,0.1)',  text: 'var(--info)',     label: 'Low cost' },
  medium: { bg: 'rgba(251,191,36,0.1)', text: 'var(--warning)',  label: 'Pay-per-use' },
  high:   { bg: 'rgba(248,113,113,0.1)', text: 'var(--danger)', label: 'Higher cost' },
};

const PROVIDER_ICONS: Record<string, string> = {
  gemini: '✦',
  openai: '⬡',
  claude: '◆',
  ollama: '🦙',
};

interface ProviderPickerProps {
  selectedProvider: string;
  onProviderChange: (id: string) => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  ollamaHost: string;
  onOllamaHostChange: (host: string) => void;
}

const LS_KEY = (provider: string) => `leaf_expert_apikey_${provider}`;

export function ProviderPicker({
  selectedProvider,
  onProviderChange,
  apiKey,
  onApiKeyChange,
  ollamaHost,
  onOllamaHostChange,
}: ProviderPickerProps) {
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    getProviders().then(setProviders).catch(() => {});
  }, []);

  // Load saved key when provider changes
  useEffect(() => {
    const saved = localStorage.getItem(LS_KEY(selectedProvider)) || '';
    onApiKeyChange(saved);
  }, [selectedProvider]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleKeyChange = (val: string) => {
    onApiKeyChange(val);
    localStorage.setItem(LS_KEY(selectedProvider), val);
  };

  const currentInfo = providers.find(p => p.id === selectedProvider);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Provider grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {(providers.length > 0 ? providers : [
          { id: 'gemini', name: 'Gemini 2.0 Flash', cost_tier: 'low', requires_api_key: true, description: 'Recommended', default_model: 'gemini-2.0-flash' },
          { id: 'openai', name: 'OpenAI GPT-4o', cost_tier: 'medium', requires_api_key: true, description: 'Most capable', default_model: 'gpt-4o' },
          { id: 'claude', name: 'Claude 3.5 Sonnet', cost_tier: 'medium', requires_api_key: true, description: 'Excellent reasoning', default_model: 'claude-3-5-sonnet-20241022' },
          { id: 'ollama', name: 'Ollama (Local)', cost_tier: 'free', requires_api_key: false, description: 'Fully offline', default_model: 'llava:latest' },
        ] as ProviderInfo[]).map(p => {
          const cost = COST_COLORS[p.cost_tier as keyof typeof COST_COLORS] ?? COST_COLORS.low;
          const isSelected = selectedProvider === p.id;
          return (
            <button
              key={p.id}
              onClick={() => onProviderChange(p.id)}
              style={{
                background: isSelected ? 'var(--accent-glow)' : 'var(--bg-surface)',
                border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                borderRadius: 'var(--radius-md)',
                padding: '12px 14px',
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 0.15s',
                display: 'flex',
                flexDirection: 'column',
                gap: 6,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '1.1rem' }}>{PROVIDER_ICONS[p.id]}</span>
                <span style={{
                  fontSize: '0.68rem', fontWeight: 700, padding: '1px 7px',
                  borderRadius: 'var(--radius-full)',
                  background: cost.bg, color: cost.text
                }}>{cost.label}</span>
              </div>
              <div style={{ fontSize: '0.8rem', fontWeight: 700, color: isSelected ? 'var(--accent)' : 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                {p.name.length > 18 ? p.name.slice(0, 17) + '…' : p.name}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{p.description}</div>
            </button>
          );
        })}
      </div>

      {/* API Key or Ollama host */}
      {selectedProvider === 'ollama' ? (
        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Server size={13} /> Ollama Host
          </label>
          <input
            className="input"
            value={ollamaHost}
            onChange={e => onOllamaHostChange(e.target.value)}
            placeholder="http://localhost:11434"
          />
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            No API key needed. Run Ollama locally with a vision model (llava, qwen2-vl).
          </span>
        </div>
      ) : (
        <div className="form-group">
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <Key size={13} /> API Key
            <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginLeft: 4 }}>
              (saved in browser — never sent to server storage)
            </span>
          </label>
          <div style={{ position: 'relative' }}>
            <input
              className="input"
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={e => handleKeyChange(e.target.value)}
              placeholder={
                selectedProvider === 'gemini' ? 'AIza…' :
                selectedProvider === 'openai' ? 'sk-…' : 'sk-ant-…'
              }
              style={{ paddingRight: 44 }}
            />
            <button
              onClick={() => setShowKey(!showKey)}
              style={{
                position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
                background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)',
              }}
            >
              {showKey ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
          </div>
          {currentInfo && (
            <a
              href={
                selectedProvider === 'gemini' ? 'https://aistudio.google.com/apikey' :
                selectedProvider === 'openai' ? 'https://platform.openai.com/api-keys' :
                'https://console.anthropic.com/account/keys'
              }
              target="_blank" rel="noopener noreferrer"
              style={{ fontSize: '0.72rem', color: 'var(--accent)', marginTop: 2 }}
            >
              Get {currentInfo.name} API key →
            </a>
          )}
        </div>
      )}
    </div>
  );
}
