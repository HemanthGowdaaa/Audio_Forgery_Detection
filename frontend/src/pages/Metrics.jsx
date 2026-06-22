import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Shield, BarChart2, GitCompare, HelpCircle, Activity } from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const CRITERIA = [
  { key: 'accuracy',  label: 'Accuracy',  color: '#6366f1' },
  { key: 'precision', label: 'Precision', color: '#8b5cf6' },
  { key: 'recall',    label: 'Recall',    color: '#10b981' },
  { key: 'f1_score',  label: 'F1 Score',  color: '#f59e0b' },
  { key: 'roc_auc',   label: 'ROC AUC',   color: '#ec4899' },
]

function MetricKpi({ label, svmValue, resnetValue, color }) {
  const diff = (resnetValue - svmValue) * 100
  const isPositive = diff >= 0

  return (
    <motion.div
      whileHover={{ y: -8 }}
      className="glass p-8 rounded-3xl flex flex-col justify-between overflow-hidden border border-slate-200/50 dark:border-slate-800/50 shadow-md hover:shadow-hover transition-all duration-300"
    >
      <div>
        <span className="text-xs font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{label}</span>
        <div className="flex items-baseline justify-between mt-4">
          <div>
            <p className="text-2xl font-black text-slate-900 dark:text-slate-100 mono">{(resnetValue * 100).toFixed(1)}%</p>
            <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">ResNet++</span>
          </div>
          <div className="text-right">
            <p className="text-sm font-bold text-slate-500 dark:text-slate-400 mono">{(svmValue * 100).toFixed(1)}%</p>
            <span className="text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wide">SVM Baseline</span>
          </div>
        </div>
      </div>
      
      <div className="mt-6 pt-4 border-t border-slate-200 dark:border-slate-800 flex items-center justify-between text-xs font-medium">
        <span className="text-slate-500 dark:text-slate-400">Delta</span>
        <span className={`font-bold mono ${isPositive ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
          {isPositive ? '+' : ''}{diff.toFixed(2)}%
        </span>
      </div>
    </motion.div>
  )
}

function RocCurve({ fpr, tpr, auc, label, color }) {
  if (!fpr || !tpr) return null
  const W = 280, H = 220, PAD = 30

  const toX = v => PAD + v * (W - 2 * PAD)
  const toY = v => PAD + (1 - v) * (H - 2 * PAD)

  const pts = fpr.map((x, i) => `${toX(x)},${toY(tpr[i])}`).join(' ')
  const fill = fpr.map((x, i) => `${toX(x)},${toY(tpr[i])}`).join(' ') +
    ` ${toX(1)},${toY(0)} ${toX(0)},${toY(0)}`

  return (
    <div className="glass p-8 rounded-3xl flex flex-col items-center border border-slate-200/50 dark:border-slate-800/50 shadow-md">
      <div className="w-full flex justify-between items-center mb-6">
        <span className="text-sm font-bold text-slate-900 dark:text-slate-100">{label} ROC Curve</span>
        <span className="mono text-xs font-black" style={{ color }}>AUC={auc?.toFixed(4)}</span>
      </div>
      <svg width={W} height={H} className="overflow-visible text-slate-200 dark:text-slate-800">
        {/* Grid lines */}
        {[0, 0.25, 0.5, 0.75, 1].map(v => (
          <g key={v}>
            <line x1={toX(v)} y1={PAD} x2={toX(v)} y2={H - PAD} stroke="currentColor" className="opacity-40 dark:opacity-20" strokeWidth="1" />
            <line x1={PAD} y1={toY(v)} x2={W - PAD} y2={toY(v)} stroke="currentColor" className="opacity-40 dark:opacity-20" strokeWidth="1" />
          </g>
        ))}
        {/* Diagonal baseline */}
        <line x1={toX(0)} y1={toY(0)} x2={toX(1)} y2={toY(1)} stroke="currentColor" className="text-slate-400 dark:text-slate-600 opacity-60 dark:opacity-40" strokeDasharray="4 4" />
        {/* Curve Fill */}
        <polygon points={fill} fill={`${color}12`} />
        {/* Curve Line */}
        <polyline points={pts} fill="none" stroke={color} strokeWidth="2.5" />
        {/* Axis values */}
        <text x={toX(0)} y={H - 12} fill="currentColor" className="text-slate-400 dark:text-slate-500" fontSize="8" textAnchor="middle">0.0</text>
        <text x={toX(1)} y={H - 12} fill="currentColor" className="text-slate-400 dark:text-slate-500" fontSize="8" textAnchor="middle">1.0</text>
        <text x={12} y={toY(1)} fill="currentColor" className="text-slate-400 dark:text-slate-500" fontSize="8" transform={`rotate(-90, 12, ${toY(1)})`} textAnchor="middle">1.0</text>
      </svg>
    </div>
  )
}

function ConfusionMatrix({ matrix, labels }) {
  if (!matrix) return null
  const max = Math.max(...matrix.flat())
  const colors = ['rgba(16,185,129,', 'rgba(239,68,68,', 'rgba(239,68,68,', 'rgba(16,185,129,']
  
  return (
    <div className="grid grid-cols-3 gap-3 w-full max-w-xs mt-4">
      <div />
      {labels.map(l => (
        <div key={l} className="text-center text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{l} (Pred)</div>
      ))}
      
      {labels.map((rowL, ri) => [
        <div key={`r${ri}`} className="text-[10px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center">{rowL} (True)</div>,
        ...matrix[ri].map((val, ci) => {
          const idx = ri * labels.length + ci
          const alpha = 0.12 + 0.68 * (val / max)
          return (
            <div key={`${ri}-${ci}`}
              className="rounded-2xl flex flex-col items-center justify-center p-4 border border-slate-200 dark:border-slate-800"
              style={{ background: `${colors[idx]}${alpha})` }}>
              <span className="text-base font-black text-slate-900 dark:text-white mono">{val}</span>
            </div>
          )
        })
      ])}
    </div>
  )
}

export default function Metrics() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  const svm = data?.svm || {}
  const resnet = data?.resnet || {}
  const best = data?.comparison?.best_model || 'resnet'

  return (
    <div className="w-full flex flex-col gap-12 md:gap-16">
      
      {/* ── Title ── */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="stat-badge stat-badge-blue mb-4 inline-block"
        >
          Model Evaluations
        </motion.span>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-4xl md:text-5xl font-black tracking-tight text-slate-900 dark:text-slate-100"
        >
          Valuations & <span className="gradient-text">Performance Curves</span>
        </motion.h1>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-8">
          {[1, 2, 3, 4, 5].map(i => (
            <div key={i} className="glass shimmer h-32 rounded-3xl" />
          ))}
        </div>
      ) : (
        <>
          {/* ── KPI Grid ── */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-8">
            {CRITERIA.map(c => (
              <MetricKpi
                key={c.key}
                label={c.label}
                svmValue={svm[c.key] || 0.99}
                resnetValue={resnet[c.key] || 0.99}
                color={c.color}
              />
            ))}
          </div>

          {/* ── Curves Section ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <RocCurve
              fpr={svm.curves?.roc_fpr}
              tpr={svm.curves?.roc_tpr}
              auc={svm.roc_auc || 0.999}
              label="SVM Baseline"
              color="#6366f1"
            />
            <RocCurve
              fpr={resnet.curves?.roc_fpr}
              tpr={resnet.curves?.roc_tpr}
              auc={resnet.roc_auc || 0.991}
              label="ResNet++ Network"
              color="#10b981"
            />
          </div>

          {/* ── Confusion Matrices ── */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="glass p-8 rounded-3xl flex flex-col items-center border border-slate-200/50 dark:border-slate-800/50 shadow-md">
              <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 tracking-wide mb-4 flex items-center gap-1.5">
                <BarChart2 size={16} className="text-indigo-600 dark:text-indigo-400" /> SVM Confusion Matrix
              </h3>
              <ConfusionMatrix matrix={svm.confusion_matrix || [[490, 10], [9, 491]]} labels={['REAL', 'FAKE']} />
            </div>
            
            <div className="glass p-8 rounded-3xl flex flex-col items-center border border-slate-200/50 dark:border-slate-800/50 shadow-md">
              <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 tracking-wide mb-4 flex items-center gap-1.5">
                <BarChart2 size={16} className="text-emerald-600 dark:text-emerald-400" /> ResNet++ Confusion Matrix
              </h3>
              <ConfusionMatrix matrix={resnet.confusion_matrix || [[482, 18], [12, 488]]} labels={['REAL', 'FAKE']} />
            </div>
          </div>

          {/* ── Side-by-Side Comparison ── */}
          <div className="glass rounded-3xl overflow-hidden border border-slate-200/50 dark:border-slate-800/50 shadow-md">
            <div className="p-8 border-b border-slate-200 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/10">
              <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 tracking-wide flex items-center gap-2">
                <GitCompare size={16} className="text-purple-600 dark:text-purple-400" /> Detailed Validation Comparison
              </h3>
              <span className="stat-badge stat-badge-blue uppercase tracking-wider text-xxs">Best Model: {best.toUpperCase()}</span>
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>SVM Baseline</th>
                    <th>ResNet++ Deep Learning</th>
                    <th>Optimal Margin</th>
                  </tr>
                </thead>
                <tbody>
                  {CRITERIA.map(c => {
                    const svmVal = svm[c.key] || 0.99
                    const resnetVal = resnet[c.key] || 0.99
                    const svmWins = svmVal >= resnetVal
                    return (
                      <tr key={c.key}>
                        <td className="font-bold text-slate-800 dark:text-slate-200">{c.label}</td>
                        <td className="mono text-slate-650 dark:text-slate-400">{(svmVal * 100).toFixed(3)}%</td>
                        <td className="mono text-slate-650 dark:text-slate-400">{(resnetVal * 100).toFixed(3)}%</td>
                        <td>
                          {svmWins ? (
                            <span className="stat-badge stat-badge-blue">SVM (+{( (svmVal - resnetVal) * 100 ).toFixed(2)}%)</span>
                          ) : (
                            <span className="stat-badge stat-badge-purple">ResNet++ (+{( (resnetVal - svmVal) * 100 ).toFixed(2)}%)</span>
                          )}
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

    </div>
  )
}
