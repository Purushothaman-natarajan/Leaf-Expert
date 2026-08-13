import { Link } from 'react-router-dom';
import { Leaf, FlaskConical, Cpu, ScanEye, Zap, ChevronRight } from 'lucide-react';

const features = [
  {
    icon: '🌿',
    color: 'var(--info)',
    title: 'Transfer Learning',
    desc: 'Choose from EfficientNetV2, ResNet, ViT and more. State-of-the-art PyTorch backbones pre-trained on ImageNet.',
  },
  {
    icon: '🔬',
    color: 'rgba(96,165,250,0.15)',
    title: 'Grad-CAM++ Explanations',
    desc: 'Understand why the model made a prediction. Heatmaps highlight the exact leaf regions that triggered the diagnosis.',
  },
  {
    icon: '🧩',
    color: 'rgba(251,191,36,0.12)',
    title: 'LIME Superpixels',
    desc: 'Model-agnostic LIME explanations show which superpixel patches contributed most to the classification.',
  },
  {
    icon: '⚡',
    color: 'rgba(167,139,250,0.12)',
    title: 'No-Code Training',
    desc: 'Upload your dataset, pick a backbone, set hyperparameters — the REST API handles everything else.',
  },
  {
    icon: '🛡',
    color: 'rgba(248,113,113,0.12)',
    title: 'Confidence Scores',
    desc: 'Per-class probability distributions surface uncertainty, so you know when to seek expert validation.',
  },
  {
    icon: '🚀',
    color: 'rgba(6,182,212,0.1)',
    title: 'FastAPI Backend',
    desc: 'Production-ready REST API with async endpoints, Pydantic validation, and full Swagger / ReDoc docs.',
  },
];

const steps = [
  { num: '01', title: 'Prepare Data', desc: 'Upload your raw leaf images. Our API splits them into train/val/test with optional augmentation.' },
  { num: '02', title: 'Train Model', desc: 'Select a backbone and hyperparameters. Two-phase fine-tuning with mixed precision and early stopping.' },
  { num: '03', title: 'Analyze & Explain', desc: 'Upload any leaf image. Get an instant diagnosis with Grad-CAM++ and LIME explanations.' },
];

export function HomePage() {
  return (
    <div className="page">
      {/* Hero */}
      <section className="hero">
        <div className="container">
          <div className="hero__eyebrow">
            <Leaf size={13} /> AI-Powered Plant Health Analysis
          </div>
          <h1 className="hero__title">
            Diagnose Plant Diseases<br />
            <span className="hero__title-highlight">With Explainable AI</span>
          </h1>
          <p className="hero__subtitle">
            Leaf-Expert combines state-of-the-art PyTorch transfer learning with
            Grad-CAM++ and LIME explanations — so you don't just get an answer,
            you understand <em>why</em>.
          </p>
          <div className="hero__cta">
            <Link to="/analyze" className="btn btn-primary btn-lg">
              <FlaskConical size={18} /> Analyze a Leaf
            </Link>
            <Link to="/train" className="btn btn-secondary btn-lg">
              <Cpu size={18} /> Train a Model
            </Link>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section style={{ padding: '20px 0 60px' }}>
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20 }}>
            {steps.map((step, i) => (
              <div
                key={step.num}
                className="glass-card"
                style={{
                  padding: '28px 24px',
                  animationDelay: `${i * 0.1}s`,
                  display: 'flex',
                  flexDirection: 'column',
                  gap: 14,
                }}
              >
                <span style={{
                  fontFamily: 'var(--font-mono)', fontSize: '2rem', fontWeight: 800,
                  color: 'var(--info)', lineHeight: 1,
                }}>{step.num}</span>
                <h3 style={{ fontSize: '1rem', fontWeight: 700 }}>{step.title}</h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                  {step.desc}
                </p>
                {i < steps.length - 1 && (
                  <div style={{ position: 'absolute', right: -10, top: '50%', transform: 'translateY(-50%)' }}>
                    <ChevronRight size={18} color="var(--text-muted)" />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <h2>Everything You Need</h2>
            <p style={{ marginTop: 10, fontSize: '0.95rem', color: 'var(--text-muted)' }}>
              From data preparation to explainable prediction — all in one place.
            </p>
          </div>
          <div className="features__grid">
            {features.map(f => (
              <div key={f.title} className="glass-card feature-card">
                <div className="feature-card__icon" style={{ background: f.color }}>
                  <span style={{ fontSize: 22 }}>{f.icon}</span>
                </div>
                <h3 className="feature-card__title">{f.title}</h3>
                <p className="feature-card__desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Banner */}
      <section style={{ padding: '40px 0 80px' }}>
        <div className="container">
          <div className="glass-card" style={{
            padding: '48px', textAlign: 'center',
            background: 'linear-gradient(135deg, rgba(6,182,212,0.06), rgba(15,23,42,0.8))',
            borderColor: 'rgba(6,182,212,0.2)',
          }}>
            <ScanEye size={40} color="var(--accent)" style={{ marginBottom: 20 }} />
            <h2 style={{ marginBottom: 12 }}>Ready to diagnose your plants?</h2>
            <p style={{ marginBottom: 28, maxWidth: 480, margin: '0 auto 28px', color: 'var(--text-secondary)' }}>
              Upload a leaf photo and get an AI-powered diagnosis with full visual explanations in seconds.
            </p>
            <Link to="/analyze" className="btn btn-primary btn-lg">
              <Zap size={18} /> Get Started Free
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
