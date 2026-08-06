import { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { Leaf, FlaskConical, Cpu } from 'lucide-react';
import { getHealth, type HealthResponse } from '../api/client';

export function Navbar() {
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  return (
    <nav className="navbar">
      <div className="navbar__inner">
        <NavLink to="/" className="navbar__logo">
          <div className="navbar__logo-icon">
            <Leaf size={18} strokeWidth={2.5} />
          </div>
          <span className="navbar__logo-text">Leaf-Expert</span>
        </NavLink>

        <div className="navbar__links">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `navbar__link${isActive ? ' active' : ''}`}
          >
            Home
          </NavLink>
          <NavLink
            to="/analyze"
            className={({ isActive }) => `navbar__link${isActive ? ' active' : ''}`}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <FlaskConical size={14} /> Analyze
            </span>
          </NavLink>
          <NavLink
            to="/train"
            className={({ isActive }) => `navbar__link${isActive ? ' active' : ''}`}
          >
            <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <Cpu size={14} /> Train
            </span>
          </NavLink>
        </div>

        <div className="navbar__status">
          <div className={`status-dot${health ? ' online' : ''}`} />
          {health ? (
            <>
              <span>API v{health.version}</span>
              <span>·</span>
              <span style={{ color: health.cuda_available ? 'var(--accent)' : 'var(--text-muted)' }}>
                {health.cuda_available ? '⚡ GPU' : '🖥 CPU'}
              </span>
            </>
          ) : (
            <span>Connecting…</span>
          )}
        </div>
      </div>
    </nav>
  );
}
