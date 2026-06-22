import { motion } from 'framer-motion'
import { Database, FileAudio, Users, Share2, Layers, HardDrive } from 'lucide-react'

function DataTile({ icon: Icon, value, label, sub, color }) {
  return (
    <motion.div
      whileHover={{ y: -8 }}
      className="glass p-8 rounded-3xl flex flex-col gap-4 relative overflow-hidden border border-slate-200/50 dark:border-slate-800/50 shadow-md hover:shadow-hover transition-all duration-300 text-left"
    >
      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider">{label}</span>
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center"
          style={{ background: `${color}18`, border: `1px solid ${color}30` }}
        >
          <Icon size={18} style={{ color }} />
        </div>
      </div>
      <p className="text-3xl md:text-4xl font-black tracking-tight text-slate-900 dark:text-slate-100 mt-2 mono">{value}</p>
      {sub && <p className="text-xs text-slate-500 dark:text-slate-400 font-semibold">{sub}</p>}
    </motion.div>
  )
}

export default function Dataset() {
  return (
    <div className="w-full flex flex-col gap-12 md:gap-16">
      
      {/* ── Title ── */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.span
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="stat-badge stat-badge-blue mb-4 inline-block"
        >
          Intelligence Repository
        </motion.span>
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-4xl md:text-5xl font-black tracking-tight text-slate-900 dark:text-slate-100"
        >
          Release in the Wild <span className="gradient-text">Dataset</span>
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-base text-slate-600 dark:text-slate-400 mt-4 leading-relaxed font-medium"
        >
          Academic voice forgery benchmark containing genuine and AI-synthesized deepfake speech samples.
        </motion.p>
      </div>

      {/* ── Key stats grids ── */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8">
        <DataTile
          icon={Database}
          label="Total Repository"
          value="31,779"
          sub="Audio file samples"
          color="#6366f1"
        />
        <DataTile
          icon={FileAudio}
          label="File Formats"
          value="3 Types"
          sub="WAV, MP3, FLAC supported"
          color="#8b5cf6"
        />
        <DataTile
          icon={Users}
          label="Classes"
          value="2 Labels"
          sub="Real Speech vs Fake Audio"
          color="#10b981"
        />
        <DataTile
          icon={HardDrive}
          label="Metadata Cache"
          value="Lazy loaded"
          sub="No memory footprint at startup"
          color="#f59e0b"
        />
      </div>

      {/* ── Split details and specs ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        {/* Dataset split */}
        <div className="glass p-8 rounded-3xl border border-slate-200/50 dark:border-slate-800/50 shadow-md text-left">
          <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
            <Layers size={18} className="text-indigo-600 dark:text-indigo-400" /> Manifest Partitions
          </h3>
          
          <div className="space-y-4">
            {[
              { label: 'Train Manifest', value: '2,000 Subset Limit', desc: 'Used for ResNet++ fine-tuning and weight optimization', percent: '80%' },
              { label: 'Validation Manifest', value: '400 Subset Limit', desc: 'Used for metric valuation and validation convergence checks', percent: '15%' },
              { label: 'Test Manifest', value: '200 Subset Limit', desc: 'Used for independent model testing and roc-auc checks', percent: '5%' }
            ].map(({ label, value, desc, percent }) => (
              <div key={label} className="p-5 rounded-2xl border border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/20">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-bold text-sm text-slate-800 dark:text-slate-200">{label}</span>
                  <span className="mono text-xs font-bold text-indigo-600 dark:text-indigo-400">{percent}</span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400 font-semibold mt-1">{value}</p>
                <p className="text-xs text-slate-400 dark:text-slate-500 mt-1 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Technical specs */}
        <div className="glass p-8 rounded-3xl border border-slate-200/50 dark:border-slate-800/50 shadow-md text-left">
          <h3 className="font-bold text-lg text-slate-900 dark:text-slate-100 mb-6 flex items-center gap-2">
            <Share2 size={18} className="text-purple-600 dark:text-purple-400" /> Preprocessing Configurations
          </h3>
          
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Parameters</th>
                  <th>Value / Spec</th>
                </tr>
              </thead>
              <tbody>
                {[
                  { param: 'Sample Rate', val: '16,000 Hz' },
                  { param: 'Channels', val: '1 (Mono)' },
                  { param: 'Max Audio Length', val: '4.0 Seconds (Trimmed)' },
                  { param: 'Mel Bins (n_mels)', val: '80 Bands' },
                  { param: 'FFT Size (n_fft)', val: '1024 Samples' },
                  { param: 'Hop Length', val: '256 Samples' }
                ].map(({ param, val }) => (
                  <tr key={param}>
                    <td className="font-bold text-slate-700 dark:text-slate-300">{param}</td>
                    <td className="mono text-slate-600 dark:text-slate-400">{val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

      </div>

    </div>
  )
}
