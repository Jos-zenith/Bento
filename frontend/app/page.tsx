const pillars = [
  {
    title: "PyTorch-first multimodal AI",
    description:
      "Native CV, audio, and vision-language pipelines for custom models and production-grade inference.",
  },
  {
    title: "Vernacular reasoning",
    description:
      "Indian-language and code-mixed support for Hinglish, regional nuance, and culturally aware mediation.",
  },
  {
    title: "Low-latency production stack",
    description:
      "FastAPI, vector retrieval, and optimized inference with a clean frontend for demos and pilots.",
  },
];

const flow = [
  "User uploads video, audio, or text",
  "PyTorch extracts emotion, posture, and tone",
  "VLM and RAG add visual and cultural context",
  "LangGraph routes reasoning and safety checks",
  "LLM returns a sensitive mediation plan",
];

export default function Home() {
  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Bento / Next.js frontend scaffold</p>
          <h1>Build a culturally aware multimodal wellness product with a sharp product surface.</h1>
          <p className="lead">
            This frontend gives Bento a modern landing experience for demos,
            investor reviews, and product iteration while the backend grows in parallel.
          </p>
          <div className="actions">
            <a className="primary" href="#architecture">View architecture</a>
            <a className="secondary" href="http://127.0.0.1:8000/health">Check backend</a>
          </div>
        </div>

        <div className="hero-panel">
          <div className="panel-card accent">
            <span className="label">Production Stack</span>
            <strong>PyTorch + Next.js + FastAPI</strong>
            <p>Prepared for CV, audio, RAG, and vernacular LLM workflows.</p>
          </div>
          <div className="panel-grid">
            <div className="panel-card">
              <span className="label">Vision</span>
              <strong>Video, emotion, posture</strong>
            </div>
            <div className="panel-card">
              <span className="label">Language</span>
              <strong>Hinglish, Tamil, Hindi, code-mix</strong>
            </div>
            <div className="panel-card">
              <span className="label">Inference</span>
              <strong>Optimized, low latency</strong>
            </div>
            <div className="panel-card">
              <span className="label">Delivery</span>
              <strong>Web demo and mobile-ready</strong>
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="architecture">
        <div className="section-heading">
          <p className="eyebrow">Architecture focus</p>
          <h2>Three pillars for a high-end product story</h2>
        </div>
        <div className="card-grid">
          {pillars.map((pillar) => (
            <article className="feature-card" key={pillar.title}>
              <h3>{pillar.title}</h3>
              <p>{pillar.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="section">
        <div className="section-heading">
          <p className="eyebrow">Proof of skill flow</p>
          <h2>From multimodal input to culturally sensitive output</h2>
        </div>
        <ol className="flow-list">
          {flow.map((step, index) => (
            <li key={step}>
              <span>{index + 1}</span>
              <p>{step}</p>
            </li>
          ))}
        </ol>
      </section>
    </main>
  );
}
