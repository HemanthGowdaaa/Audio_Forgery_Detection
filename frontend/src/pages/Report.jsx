import { useEffect, useState } from 'react'
import { FileText, Download, TrendingUp, Activity, Shield, Cpu } from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

function SectionTitle({ children }) {
  return (
    <p className="text-xs font-semibold uppercase tracking-widest mb-4"
      style={{ color: 'var(--text-muted)' }}>
      {children}
    </p>
  )
}

function MetricTile({ label, value, sub, color = '#6366f1' }) {
  return (
    <div className="glass glass-hover p-5 rounded-2xl text-center fade-in">
      <p className="text-2xl font-black mono" style={{ color }}>{value}</p>
      <p className="text-sm font-medium mt-1" style={{ color: 'var(--text-primary)' }}>{label}</p>
      {sub && <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>{sub}</p>}
    </div>
  )
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix) return null
  const max = Math.max(...matrix.flat())
  const colors = ['rgba(16,185,129,', 'rgba(239,68,68,', 'rgba(239,68,68,', 'rgba(16,185,129,']
  return (
    <div>
      <div className="grid grid-cols-3 gap-1 max-w-xs">
        <div />
        {labels.map(l => (
          <div key={l} className="text-center text-xs font-semibold py-1"
            style={{ color: 'var(--text-muted)' }}>Pred {l}</div>
        ))}
        {labels.map((rowL, ri) => [
          <div key={`r${ri}`} className="text-xs font-semibold flex items-center"
            style={{ color: 'var(--text-muted)' }}>True {rowL}</div>,
          ...matrix[ri].map((val, ci) => {
            const idx = ri * labels.length + ci
            const alpha = 0.15 + 0.7 * (val / max)
            return (
              <div key={`${ri}-${ci}`}
                className="rounded-lg flex flex-col items-center justify-center p-3"
                style={{ background: `${colors[idx]}${alpha})` }}>
                <span className="text-lg font-bold mono" style={{ color: 'var(--text-primary)' }}>{val}</span>
              </div>
            )
          })
        ])}
      </div>
    </div>
  )
}

function RocCurve({ fpr, tpr, auc, label, color }) {
  if (!fpr || !tpr) return null
  const W = 260, H = 200, PAD = 30

  const toX = v => PAD + v * (W - 2 * PAD)
  const toY = v => PAD + (1 - v) * (H - 2 * PAD)

  const pts = fpr.map((x, i) => `${toX(x)},${toY(tpr[i])}`).join(' ')
  const fill = fpr.map((x, i) => `${toX(x)},${toY(tpr[i])}`).join(' ') +
    ` ${toX(1)},${toY(0)} ${toX(0)},${toY(0)}`

  return (
    <div className="glass glass-hover p-5 rounded-2xl fade-in">
      <p className="font-semibold text-sm mb-3" style={{ color: 'var(--text-primary)' }}>
        ROC Curve — {label}
        <span className="mono ml-2" style={{ color }}> AUC={auc?.toFixed(4)}</span>
      </p>
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`}>
        {/* Grid */}
        {[0, 0.25, 0.5, 0.75, 1].map(v => (
          <g key={v}>
            <line x1={toX(v)} y1={PAD} x2={toX(v)} y2={H - PAD} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
            <line x1={PAD} y1={toY(v)} x2={W - PAD} y2={toY(v)} stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
          </g>
        ))}
        {/* Diagonal */}
        <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" strokeWidth="1" />
        {/* Fill */}
        <polygon points={fill} fill={`${color}20`} />
        {/* Curve */}
        <polyline points={pts} fill="none" stroke={color} strokeWidth="2.5"
          style={{ filter: `drop-shadow(0 0 4px ${color})` }} />
        {/* Axes labels */}
        <text x={W / 2} y={H - 4} fill="rgba(255,255,255,0.4)" fontSize="10" textAnchor="middle">FPR</text>
        <text x={10} y={H / 2} fill="rgba(255,255,255,0.4)" fontSize="10" textAnchor="middle"
          transform={`rotate(-90, 10, ${H / 2})`}>TPR</text>
      </svg>
    </div>
  )
}

export default function Report() {
  const [data, setData]     = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const svm    = data?.svm    || {}
  const resnet = data?.resnet || {}

  const svmCM    = svm.confusion_matrix
  const resnetCM = resnet.confusion_matrix
  const labels   = ['REAL', 'FAKE']

  return (
    <div className="p-8 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-start justify-between mb-8 fade-in">
        <div>
          <h1 className="text-3xl font-black gradient-text mb-2">Performance Report</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: 14 }}>
            In-The-Wild Audio Deepfake Detection — Full evaluation metrics
          </p>
        </div>
        <a href={`${API}/report`} target="_blank" rel="noreferrer" className="btn-secondary">
          <Download size={16} /> Export HTML
        </a>
      </div>

      {loading ? (
        <div className="space-y-4">
          {[1,2,3].map(i => <div key={i} className="glass rounded-2xl h-36 shimmer" />)}
        </div>
      ) : (
        <>
          {/* ── Overview Tiles ── */}
          <SectionTitle>Overall Performance</SectionTitle>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
            <MetricTile label="SVM Accuracy"    value={`${(svm.accuracy    * 100).toFixed(1)}%`} color="#6366f1" sub="Test set" />
            <MetricTile label="SVM F1 Score"    value={`${(svm.f1_score    * 100).toFixed(1)}%`} color="#8b5cf6" sub="Weighted" />
            <MetricTile label="ResNet Accuracy" value={`${(resnet.accuracy * 100).toFixed(1)}%`} color="#10b981" sub="Test set" />
            <MetricTile label="ResNet F1 Score" value={`${(resnet.f1_score * 100).toFixed(1)}%`} color="#f59e0b" sub="Weighted" />
          </div>

          {/* ── SVM Section ── */}
          <SectionTitle>SVM Baseline Details</SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            {[
              { l: 'Accuracy',  v: svm.accuracy,  c: '#6366f1' },
              { l: 'Precision', v: svm.precision, c: '#8b5cf6' },
              { l: 'Recall',    v: svm.recall,    c: '#a78bfa' },
              { l: 'F1 Score',  v: svm.f1_score,  c: '#10b981' },
              { l: 'ROC AUC',   v: svm.roc_auc,   c: '#f59e0b' },
              { l: 'PR AUC',    v: svm.pr_auc,    c: '#ef4444' },
            ].map(({ l, v, c }) => (
              <MetricTile key={l} label={l} value={`${(v * 100).toFixed(2)}%`} color={c} />
            ))}
          </div>

          {/* SVM Training info */}
          <div className="glass p-5 rounded-2xl mb-8 fade-in">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              {[
                ['Best Kernel',     svm.best_params?.svm__kernel || 'rbf'],
                ['Best C',          svm.best_params?.svm__C      || '—'],
                ['Train Samples',   svm.final_train_samples?.toLocaleString() || '—'],
                ['Train Time',      `${svm.training_time_sec?.toFixed(0)}s`],
              ].map(([k, v]) => (
                <div key={k}>
                  <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>{k}</p>
                  <p className="font-bold mono" style={{ color: 'var(--text-primary)' }}>{v}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ── ROC Curves ── */}
          <SectionTitle>ROC Curves</SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <RocCurve
              fpr={svm.curves?.roc_fpr}
              tpr={svm.curves?.roc_tpr}
              auc={svm.roc_auc}
              label="SVM"
              color="#6366f1"
            />
            <RocCurve
              fpr={resnet.curves?.roc_fpr}
              tpr={resnet.curves?.roc_tpr}
              auc={resnet.roc_auc}
              label="ResNet++"
              color="#10b981"
            />
          </div>

          {/* ── Confusion Matrices ── */}
          <SectionTitle>Confusion Matrices</SectionTitle>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="glass p-6 rounded-2xl fade-in">
              <p className="font-semibold text-sm mb-4" style={{ color: 'var(--text-primary)' }}>SVM Baseline</p>
              <ConfusionMatrix matrix={svmCM} labels={labels} />
            </div>
            <div className="glass p-6 rounded-2xl fade-in">
              <p className="font-semibold text-sm mb-4" style={{ color: 'var(--text-primary)' }}>ResNet++ Deep Learning</p>
              <ConfusionMatrix matrix={resnetCM} labels={labels} />
            </div>
          </div>

          {/* ── Classification Report Table ── */}
          <SectionTitle>Classification Report — SVM</SectionTitle>
          <div className="glass rounded-2xl overflow-hidden mb-8 fade-in">
            <table className="data-table w-full">
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1-Score</th>
                  <th>Support</th>
                </tr>
              </thead>
              <tbody>
                {svm.classification_report && Object.entries(svm.classification_report)
                  .filter(([k]) => !['accuracy'].includes(k))
                  .map(([cls, vals]) => (
                    <tr key={cls}>
                      <td><span className="font-semibold" style={{ color: 'var(--text-primary)' }}>{cls}</span></td>
                      <td className="mono">{(vals.precision * 100).toFixed(2)}%</td>
                      <td className="mono">{(vals.recall    * 100).toFixed(2)}%</td>
                      <td className="mono">{(vals['f1-score'] * 100).toFixed(2)}%</td>
                      <td className="mono">{vals.support}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* ── Dataset summary ── */}
          <SectionTitle>Dataset</SectionTitle>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 fade-in">
            {[
              { l: 'Total Samples', v: '31,779', icon: Activity },
              { l: 'Train Size',    v: svm.final_train_samples?.toLocaleString() || '6,000', icon: Cpu },
              { l: 'Test Size',     v: '1,000',  icon: Shield },
              { l: 'Classes',       v: '2',       icon: TrendingUp },
            ].map(({ l, v, icon: Icon }) => (
              <div key={l} className="glass glass-hover p-5 rounded-2xl flex items-center gap-4">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0"
                  style={{ background: 'rgba(99,102,241,0.12)', border: '1px solid rgba(99,102,241,0.2)' }}>
                  <Icon size={18} style={{ color: '#6366f1' }} />
                </div>
                <div>
                  <p className="text-lg font-bold mono" style={{ color: 'var(--text-primary)' }}>{v}</p>
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{l}</p>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
