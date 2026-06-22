import { useEffect, useState } from 'react'
import { GitCompare, TrendingUp, Shield, Zap } from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const CRITERIA = [
  { key: 'accuracy',  label: 'Accuracy',  icon: TrendingUp },
  { key: 'f1_score',  label: 'F1 Score',  icon: Shield },
  { key: 'roc_auc',   label: 'ROC AUC',   icon: Zap },
  { key: 'precision', label: 'Precision', icon: TrendingUp },
  { key: 'recall',    label: 'Recall',    icon: Shield },
]

function CompareBar({ label, svm, resnet }) {
  const svmPct    = (svm    * 100).toFixed(1)
  const resnetPct = (resnet * 100).toFixed(1)
  const svmWins   = svm >= resnet
  return (
    <div className="mb-6 fade-in">
      <p className="text-sm font-semibold mb-2" style={{ color: 'var(--text-primary)' }}>{label}</p>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span style={{ color: 'var(--text-muted)' }}>SVM</span>
            <span className={`mono font-semibold ${svmWins ? '' : 'opacity-60'}`}
              style={{ color: svmWins ? '#10b981' : 'var(--text-muted)' }}>{svmPct}%</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{
              width: `${svmPct}%`,
              background: svmWins ? 'linear-gradient(90deg,#10b981,#059669)' : 'rgba(99,102,241,0.4)'
            }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs mb-1">
            <span style={{ color: 'var(--text-muted)' }}>ResNet++</span>
            <span className={`mono font-semibold ${!svmWins ? '' : 'opacity-60'}`}
              style={{ color: !svmWins ? '#10b981' : 'var(--text-muted)' }}>{resnetPct}%</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{
              width: `${resnetPct}%`,
              background: !svmWins ? 'linear-gradient(90deg,#10b981,#059669)' : 'rgba(99,102,241,0.4)'
            }} />
          </div>
        </div>
      </div>
    </div>
  )
}

function RadarMetric({ label, svm, resnet }) {
  return (
    <tr>
      <td className="py-3 pr-4">
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{label}</p>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="mono text-sm" style={{ color: '#6366f1' }}>{(svm * 100).toFixed(2)}%</span>
      </td>
      <td className="py-3 px-4 text-center">
        <span className="mono text-sm" style={{ color: '#8b5cf6' }}>{(resnet * 100).toFixed(2)}%</span>
      </td>
      <td className="py-3 pl-4 text-center">
        {svm >= resnet
          ? <span className="stat-badge stat-badge-blue">SVM</span>
          : <span className="stat-badge" style={{ background: 'rgba(139,92,246,0.12)', color: '#8b5cf6', border: '1px solid rgba(139,92,246,0.2)' }}>ResNet++</span>}
      </td>
    </tr>
  )
}

export default function Comparison() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => { setMetrics(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const svm    = metrics?.svm    || {}
  const resnet = metrics?.resnet || {}
  const best   = metrics?.comparison?.best_model || '—'

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="mb-8 fade-in">
        <h1 className="text-3xl font-black gradient-text mb-2">Model Comparison</h1>
        <p style={{ color: 'var(--text-secondary)' }}>
          Head-to-head performance breakdown: SVM Baseline vs ResNet++ Deep Learning
        </p>
      </div>

      {/* Winner banner */}
      {!loading && (
        <div className="glass p-5 rounded-2xl mb-8 fade-in"
          style={{ background: 'linear-gradient(135deg,rgba(99,102,241,0.1),rgba(139,92,246,0.06))', border: '1px solid rgba(99,102,241,0.2)' }}>
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center glow-accent"
              style={{ background: 'linear-gradient(135deg,#6366f1,#8b5cf6)' }}>
              <GitCompare size={20} color="white" />
            </div>
            <div>
              <p className="text-xs uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Best Overall Model</p>
              <p className="text-xl font-bold gradient-text">{best.toUpperCase()} wins</p>
            </div>
          </div>
        </div>
      )}

      {loading ? (
        <div className="space-y-4">
          {[1,2,3,4,5].map(i => <div key={i} className="glass rounded-xl h-20 shimmer" />)}
        </div>
      ) : (
        <>
          {/* Side-by-side bars */}
          <div className="glass p-6 rounded-2xl mb-6 fade-in">
            <p className="text-sm font-bold mb-5" style={{ color: 'var(--text-primary)' }}>
              Performance Comparison (per metric)
            </p>
            {CRITERIA.map(c => (
              <CompareBar key={c.key} label={c.label}
                svm={svm[c.key] || 0} resnet={resnet[c.key] || 0} />
            ))}
          </div>

          {/* Detailed table */}
          <div className="glass rounded-2xl overflow-hidden fade-in">
            <div className="p-4 border-b" style={{ borderColor: 'rgba(99,102,241,0.1)' }}>
              <p className="font-semibold" style={{ color: 'var(--text-primary)' }}>Detailed Metrics Table</p>
            </div>
            <table className="data-table w-full">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th className="text-center">SVM</th>
                  <th className="text-center">ResNet++</th>
                  <th className="text-center">Winner</th>
                </tr>
              </thead>
              <tbody>
                {CRITERIA.map(c => (
                  <RadarMetric key={c.key} label={c.label}
                    svm={svm[c.key] || 0} resnet={resnet[c.key] || 0} />
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
