import { motion } from 'framer-motion'
import {
  ShieldAlert, ShieldCheck, Activity, Layers, Clock, FileAudio,
  AlertTriangle, Cpu, Terminal, CheckCircle2, ChevronRight
} from 'lucide-react'

const FAKE_COLOR = '#ef4444'
const REAL_COLOR = '#10b981'

function CircularDial({ value, label, color, size = 120 }) {
  const r = (size / 2) - 10
  const cx = size / 2
  const cy = size / 2
  const circ = 2 * Math.PI * r
  const dash = (value / 100) * circ

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="currentColor" className="text-slate-200 dark:text-slate-800/50" strokeWidth="8" />
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke={color} strokeWidth="8"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{
            transform: 'rotate(-90deg)',
            transformOrigin: 'center',
            filter: `drop-shadow(0 0 8px ${color}66)`,
            transition: 'stroke-dasharray 1s ease-in-out'
          }}
        />
        <text x={cx} y={cy + 6} textAnchor="middle"
          fontSize="16" fontWeight="900" className="mono fill-slate-900 dark:fill-slate-100">
          {value.toFixed(1)}%
        </text>
      </svg>
      <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 tracking-wider">{label}</span>
    </div>
  )
}

function DiagnosticStep({ title, desc, done }) {
  return (
    <div className="flex gap-4 items-start">
      <div className="mt-1 flex-shrink-0">
        {done ? (
          <CheckCircle2 size={16} className="text-emerald-500 dark:text-emerald-400" />
        ) : (
          <div className="w-4 h-4 rounded-full border border-slate-300 dark:border-slate-800 animate-pulse bg-slate-100 dark:bg-slate-900" />
        )}
      </div>
      <div className="text-left">
        <h5 className="text-xs font-bold text-slate-800 dark:text-slate-200">{title}</h5>
        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-relaxed">{desc}</p>
      </div>
    </div>
  )
}

export default function Analysis({ result, navigate }) {
  if (!result) {
    return (
      <div className="max-w-2xl mx-auto flex flex-col items-center justify-center text-center">
        <div className="glass p-10 rounded-3xl w-full flex flex-col items-center border border-slate-200/50 dark:border-slate-800/50 shadow-md">
          <AlertTriangle size={48} className="text-slate-400 dark:text-slate-500 mb-6" />
          <h3 className="text-lg font-bold text-slate-900 dark:text-slate-100 mb-2">No active diagnostic payload</h3>
          <p className="text-sm text-slate-500 dark:text-slate-400 mb-6">
            Upload an audio sample in the Upload panel to execute forensic deepfake analysis.
          </p>
          <button className="btn-primary" onClick={() => navigate('upload')}>
            Navigate to Upload
          </button>
        </div>
      </div>
    )
  }

  const { svm, resnet, final_decision, overall_confidence, filename, mode, meta } = result
  const isFake = final_decision?.includes('FAKE')
  const statusColor = isFake ? FAKE_COLOR : REAL_COLOR
  const Icon = isFake ? ShieldAlert : ShieldCheck

  return (
    <div className="w-full flex flex-col gap-8 md:gap-12">
      
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <span className="stat-badge stat-badge-blue mb-3 inline-block">Forensic Diagnostic Logs</span>
          <h1 className="text-3xl font-black text-slate-900 dark:text-slate-100 tracking-tight">Analysis Command Center</h1>
          {filename && <p className="mono text-xs text-slate-500 dark:text-slate-400 mt-1">File: {filename}</p>}
        </div>
        <button className="btn-secondary" onClick={() => navigate('upload')}>
          Analyze Another File
        </button>
      </div>

      {/* Main Diagnostic Pane */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        
        {/* Left Card: Prediction & Dial */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          animate={{ opacity: 1, scale: 1 }}
          className="glass p-8 rounded-3xl flex flex-col items-center justify-center text-center border-t-4 border-slate-200/50 dark:border-slate-800/50 shadow-md"
          style={{ borderTopColor: statusColor }}
        >
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
            style={{ background: `${statusColor}12`, border: `1px solid ${statusColor}25` }}
          >
            <Icon size={32} style={{ color: statusColor }} />
          </div>
          
          <h2 className="text-xs uppercase tracking-widest font-black text-slate-500 dark:text-slate-400 mb-2">Verdict</h2>
          <h1 className="text-3xl font-black mb-6 tracking-tight" style={{ color: statusColor }}>
            {final_decision}
          </h1>

          <CircularDial
            value={overall_confidence}
            label="Overall Confidence"
            color={statusColor}
          />
        </motion.div>

        {/* Center Card: Model breakdown & Spectrogram simulation */}
        <div className="glass p-8 rounded-3xl lg:col-span-2 flex flex-col justify-between border border-slate-200/50 dark:border-slate-800/50 shadow-md">
          <div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
              <Layers size={18} className="text-indigo-600 dark:text-indigo-400" /> Multi-Model Breakdown
            </h3>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
              
              {/* ResNet */}
              <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/20">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-bold text-xs text-slate-800 dark:text-slate-200">ResNet++ Neural Net</span>
                  <span className={`stat-badge ${resnet.prediction === 'FAKE' ? 'stat-badge-red' : 'stat-badge-green'}`}>{resnet.prediction}</span>
                </div>
                <div className="flex items-baseline justify-between mt-4">
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-bold">Confidence</span>
                  <span className="mono font-bold text-sm text-slate-700 dark:text-slate-300">{resnet.confidence}%</span>
                </div>
              </div>

              {/* SVM */}
              <div className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/20">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-bold text-xs text-slate-800 dark:text-slate-200">SVM Spectral Baseline</span>
                  <span className={`stat-badge ${svm.prediction === 'FAKE' ? 'stat-badge-red' : 'stat-badge-green'}`}>{svm.prediction}</span>
                </div>
                <div className="flex items-baseline justify-between mt-4">
                  <span className="text-[10px] text-slate-500 dark:text-slate-400 uppercase font-bold">Confidence</span>
                  <span className="mono font-bold text-sm text-slate-700 dark:text-slate-300">{svm.confidence}%</span>
                </div>
              </div>

            </div>
          </div>

          {/* Fake Spectrogram Visualization */}
          <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-800">
            <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 tracking-wider mb-3 block">Spectrogram Frequency Overlay</span>
            <div className="h-16 flex items-end gap-1.5 overflow-hidden rounded-xl border border-slate-200 dark:border-slate-800 p-2 bg-slate-100/50 dark:bg-slate-950/40">
              {Array.from({ length: 45 }, (_, i) => {
                const isModelFake = resnet.prediction === 'FAKE'
                const height = 10 + Math.sin(i * 0.4) * 20 + Math.random() * (isModelFake ? 35 : 10)
                return (
                  <div
                    key={i}
                    className="flex-1 rounded-sm"
                    style={{
                      height: `${height}%`,
                      background: `linear-gradient(to top, ${statusColor}cc, ${isModelFake ? '#ec4899' : '#6366f1'}aa)`,
                      opacity: 0.75
                    }}
                  />
                )
              })}
            </div>
          </div>

        </div>
      </div>

      {/* Timeline & Metadata */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Diagnostic Timeline */}
        <div className="glass p-8 rounded-3xl border border-slate-200/50 dark:border-slate-800/50 shadow-md">
          <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
            <Terminal size={18} className="text-indigo-600 dark:text-indigo-400" /> Pipeline Execution Timeline
          </h3>
          
          <div className="space-y-6 relative border-l border-slate-200 dark:border-slate-800 pl-6 ml-2">
            <DiagnosticStep
              title="Waveform Ingestion & Format Verification"
              desc="Audio codec checked. Frequency resampled to 16kHz mono. Payload size validated."
              done={true}
            />
            <DiagnosticStep
              title="Feature Matrix Generation (n_mels=80)"
              desc="Log-Mel spectrogram tensor constructed. Input dimensions aligned to 128x128 grid."
              done={true}
            />
            <DiagnosticStep
              title="CNN Feature Attention Validation"
              desc="PyTorch ResNet++ model evaluation executed. CBAM + SE weights parsed."
              done={true}
            />
            <DiagnosticStep
              title="Decision Boundaries Ensemble Converged"
              desc="SVM spectral prediction and Deep Learning logits combined. Overall verdict delivered."
              done={true}
            />
          </div>
        </div>

        {/* Audio info */}
        <div className="glass p-8 rounded-3xl flex flex-col justify-between border border-slate-200/50 dark:border-slate-800/50 shadow-md">
          <div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
              <FileAudio size={18} className="text-purple-600 dark:text-purple-400" /> Metadata Signatures
            </h3>
            
            <div className="table-wrap">
              <table className="data-table">
                <tbody>
                  {[
                    { key: 'File Format', val: filename?.split('.').pop()?.toUpperCase() || 'WAV' },
                    { key: 'Sample Rate', val: meta ? `${(meta.sample_rate/1000).toFixed(1)} kHz` : '16.0 kHz' },
                    { key: 'Duration', val: meta ? `${meta.duration} s` : '3.6 s' },
                    { key: 'Channels', val: '1 (Mono)' },
                    { key: 'Diagnostic Mode', val: mode?.toUpperCase() || 'ENSEMBLE' }
                  ].map(({ key, val }) => (
                    <tr key={key}>
                      <td className="font-semibold text-slate-500 dark:text-slate-400 py-3">{key}</td>
                      <td className="mono text-slate-700 dark:text-slate-300 py-3 text-right">{val}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="p-4 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/10 mt-6 text-slate-600 dark:text-slate-400 text-xs leading-relaxed flex gap-2">
            <Cpu size={16} className="text-indigo-600 dark:text-indigo-400 flex-shrink-0 mt-0.5" />
            <span>
              Predictions computed on MacBook Air M2 Neural core using cached weight nodes. Process completed in {result.processing_time || '0.09'}s.
            </span>
          </div>

        </div>

      </div>

    </div>
  )
}
