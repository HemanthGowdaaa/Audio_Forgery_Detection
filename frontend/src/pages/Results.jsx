import { Upload, AlertCircle, CheckCircle, XCircle, Layers } from 'lucide-react'

const FAKE_COLOR = '#ef4444'
const REAL_COLOR = '#10b981'

function CircularScore({ value, label, color, size = 104 }) {
  const r    = (size / 2) - 8
  const cx   = size / 2
  const cy   = size / 2
  const circ = 2 * Math.PI * r
  const dash = (value / 100) * circ

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="rgba(255,255,255,0.06)" strokeWidth="7" />
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke={color} strokeWidth="7"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          className="circle-progress"
          style={{ filter: `drop-shadow(0 0 6px ${color}88)` }} />
        <text x={cx} y={cy + 5} textAnchor="middle"
          fill="white" fontSize="14" fontWeight="700" fontFamily="JetBrains Mono">
          {value.toFixed(1)}%
        </text>
      </svg>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 500 }}>{label}</p>
    </div>
  )
}

function MetricRow({ label, value }) {
  const pct = (value * 100).toFixed(1)
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{label}</span>
        <span className="mono" style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>
          {pct}%
        </span>
      </div>
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function ModelResultCard({ name, prediction, confidence, metrics, mode }) {
  const unavailable = prediction === 'UNAVAILABLE'
  const isFake  = prediction === 'FAKE'
  const color   = unavailable ? 'var(--text-muted)' : isFake ? FAKE_COLOR : REAL_COLOR
  const bgColor = unavailable
    ? 'rgba(71,85,105,0.08)'
    : isFake ? 'rgba(239,68,68,0.08)' : 'rgba(16,185,129,0.08)'
  const border  = unavailable
    ? 'rgba(71,85,105,0.2)'
    : isFake ? 'rgba(239,68,68,0.2)' : 'rgba(16,185,129,0.2)'

  return (
    <div className="glass glass-hover fade-in" style={{ padding: '24px', borderRadius: 18 }}>
      <p style={{ fontWeight: 700, fontSize: 14, color: 'var(--text-primary)', marginBottom: 16 }}>{name}</p>

      <div style={{
        display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20,
        padding: '14px 16px', borderRadius: 12,
        background: bgColor, border: `1px solid ${border}`
      }}>
        <div style={{ flex: 1 }}>
          <p style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 3 }}>Prediction</p>
          {unavailable
            ? <p style={{ fontSize: 18, fontWeight: 800, color: 'var(--text-muted)' }}>N/A</p>
            : <p style={{ fontSize: 22, fontWeight: 900, color }}>{prediction}</p>
          }
        </div>
        {!unavailable && (
          <CircularScore value={confidence} label="Confidence" color={color} size={96} />
        )}
      </div>

      {unavailable ? (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', textAlign: 'center', padding: '8px 0' }}>
          Model not available — run the training pipeline.
        </p>
      ) : (
        <div>
          <MetricRow label="Accuracy"  value={metrics.accuracy}  />
          <MetricRow label="Precision" value={metrics.precision} />
          <MetricRow label="Recall"    value={metrics.recall}    />
          <MetricRow label="F1 Score"  value={metrics.f1_score}  />
          <MetricRow label="ROC AUC"   value={metrics.roc_auc}   />
        </div>
      )}
    </div>
  )
}

function ModeBadge({ mode }) {
  if (mode === 'ensemble')   return <span className="stat-badge stat-badge-green"><Layers size={10} /> Ensemble</span>
  if (mode === 'svm_only')   return <span className="stat-badge stat-badge-yellow"><AlertCircle size={10} /> SVM-Only</span>
  if (mode === 'resnet_only') return <span className="stat-badge stat-badge-purple"><AlertCircle size={10} /> ResNet-Only</span>
  return null
}

export default function Results({ result, navigate }) {
  if (!result) {
    return (
      <div style={{ padding: 32, maxWidth: 560, margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <div className="glass fade-in" style={{ padding: '48px 36px', borderRadius: 20, textAlign: 'center', width: '100%' }}>
          <AlertCircle size={48} style={{ color: 'var(--text-muted)', margin: '0 auto 16px' }} />
          <p style={{ fontSize: 17, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 8 }}>
            No results yet
          </p>
          <p style={{ color: 'var(--text-secondary)', fontSize: 14, marginBottom: 24 }}>
            Upload and analyze an audio file to see detection results here.
          </p>
          <button className="btn-primary" onClick={() => navigate('upload')}>
            <Upload size={16} /> Upload Audio
          </button>
        </div>
      </div>
    )
  }

  const { svm, resnet, final_decision, overall_confidence, filename, mode } = result
  const isFake     = final_decision?.includes('FAKE')
  const finalColor = isFake ? FAKE_COLOR : REAL_COLOR
  const FinalIcon  = isFake ? XCircle : CheckCircle

  return (
    <div className="page-container" style={{ padding: '32px', maxWidth: 1100, margin: '0 auto' }}>

      {/* ── Header ── */}
      <div className="fade-in" style={{ marginBottom: 28 }}>
        <p style={{ fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 6 }}>
          Detection Results
        </p>
        <h1 className="gradient-text section-title" style={{ fontSize: '1.9rem', fontWeight: 900, marginBottom: 4 }}>
          Analysis Complete
        </h1>
        {filename && (
          <p className="mono" style={{ fontSize: 12, color: 'var(--text-muted)' }}>File: {filename}</p>
        )}
      </div>

      {/* ── Final decision banner ── */}
      <div
        className="glass fade-in"
        style={{
          padding: '24px 28px', borderRadius: 18, marginBottom: 28,
          background: isFake ? 'rgba(239,68,68,0.07)' : 'rgba(16,185,129,0.07)',
          border: `1px solid ${isFake ? 'rgba(239,68,68,0.25)' : 'rgba(16,185,129,0.25)'}`,
        }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 20 }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10 }}>
              <p style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-muted)' }}>
                Final Ensemble Decision
              </p>
              {mode && <ModeBadge mode={mode} />}
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <FinalIcon size={32} style={{ color: finalColor }} />
              <p style={{ fontSize: '2rem', fontWeight: 900, color: finalColor }}>
                {final_decision}
              </p>
            </div>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 8 }}>
              {mode === 'ensemble'
                ? 'Based on weighted ensemble of SVM + ResNet++ predictions'
                : mode === 'svm_only'
                ? 'Based on SVM prediction (ResNet++ not yet trained)'
                : 'Based on available model predictions'}
            </p>
          </div>
          <CircularScore
            value={overall_confidence}
            label="Overall Confidence"
            color={finalColor}
            size={112}
          />
        </div>
      </div>

      {/* ── Per-model breakdown ── */}
      <h2 style={{ fontSize: 15, fontWeight: 700, marginBottom: 16, color: 'var(--text-primary)' }}>
        Per-Model Breakdown
      </h2>
      <div
        className="grid-responsive-2"
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 28 }}
      >
        <ModelResultCard
          name="ResNet++ (Deep Learning)"
          prediction={resnet.prediction}
          confidence={resnet.confidence}
          metrics={resnet}
          mode={mode}
        />
        <ModelResultCard
          name="SVM Baseline (ML)"
          prediction={svm.prediction}
          confidence={svm.confidence}
          metrics={svm}
          mode={mode}
        />
      </div>

      {/* ── Comparison table ── */}
      <div className="glass fade-in" style={{ borderRadius: 18, overflow: 'hidden', marginBottom: 24 }}>
        <div style={{ padding: '14px 18px', borderBottom: '1px solid rgba(99,102,241,0.1)' }}>
          <p style={{ fontWeight: 600, fontSize: 14, color: 'var(--text-primary)' }}>Model Comparison</p>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
                <th>ROC AUC</th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: 'ResNet++', data: resnet },
                { label: 'SVM',     data: svm },
              ].map(({ label, data }) => (
                <tr key={label}>
                  <td><span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{label}</span></td>
                  <td>
                    {data.prediction === 'UNAVAILABLE'
                      ? <span className="stat-badge" style={{ background: 'rgba(71,85,105,0.12)', color: 'var(--text-muted)', border: '1px solid rgba(71,85,105,0.2)' }}>N/A</span>
                      : <span className={`stat-badge ${data.prediction === 'FAKE' ? 'stat-badge-red' : 'stat-badge-green'}`}>
                          {data.prediction}
                        </span>
                    }
                  </td>
                  <td className="mono">
                    {data.prediction === 'UNAVAILABLE' ? '—' : `${data.confidence?.toFixed(1)}%`}
                  </td>
                  <td className="mono">{(data.accuracy * 100).toFixed(1)}%</td>
                  <td className="mono">{(data.f1_score * 100).toFixed(1)}%</td>
                  <td className="mono">{(data.roc_auc * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <button className="btn-secondary fade-in" onClick={() => navigate('upload')}>
        <Upload size={16} /> Analyze Another File
      </button>
    </div>
  )
}
