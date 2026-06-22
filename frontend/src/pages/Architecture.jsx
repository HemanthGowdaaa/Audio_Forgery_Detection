import { motion } from 'framer-motion'
import { Cpu, Eye, Sliders, Layers, Network, Activity, ArrowRight, ArrowDown } from 'lucide-react'
import { useState } from 'react'

function ArchNode({ title, subtitle, icon: Icon, color, delay, details, isActive, onClick, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
      onClick={onClick}
      className={`glass glass-card-interactive p-8 rounded-3xl flex flex-col gap-4 relative cursor-pointer border-l-4 transition-all duration-300 border-slate-200 dark:border-slate-800 ${
        isActive 
          ? 'scale-[1.02] shadow-xl z-10 bg-slate-100/80 dark:bg-slate-900/40' 
          : 'scale-100 opacity-90 hover:opacity-100'
      }`}
      style={{ 
        borderLeftColor: color,
        boxShadow: isActive ? `0 20px 40px ${color}1a, 0 0 0 1px ${color}30` : 'var(--shadow-card)'
      }}
    >
      <div className="absolute top-6 right-6 text-slate-400 dark:text-slate-500 font-mono text-xs font-bold opacity-60">
        STAGE 0{index}
      </div>
      <div className="flex items-center gap-4">
        <div
          className={`w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0 transition-colors duration-300`}
          style={{ background: `${color}16`, border: `1px solid ${color}30` }}
        >
          <Icon size={22} style={{ color }} />
        </div>
        <div>
          <h4 className="font-bold text-base text-slate-900 dark:text-slate-100 tracking-wide">{title}</h4>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1 font-bold uppercase tracking-wider">{subtitle}</p>
        </div>
      </div>
      
      <motion.div 
        initial={false}
        animate={{ height: isActive ? 'auto' : 0, opacity: isActive ? 1 : 0, marginTop: isActive ? 16 : 0 }}
        className="overflow-hidden"
      >
        <div className="p-4 rounded-2xl bg-slate-200/50 dark:bg-black/20 border border-slate-300/50 dark:border-white/5 backdrop-blur-sm">
          <p className="text-sm text-slate-700 dark:text-slate-300 leading-relaxed font-medium">{details}</p>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default function Architecture() {
  const [activeIndex, setActiveIndex] = useState(0)

  const pipelineStages = [
    {
      title: "Audio Input Waveform",
      subtitle: "Raw Sound Stream",
      icon: Activity,
      color: "#6366f1",
      details: "Accepts WAV, MP3, FLAC files. Auto-downsampled to 16kHz mono to ensure consistency and prevent computational OOM on Apple Silicon.",
      group: "1. Input Pipeline"
    },
    {
      title: "Log-Mel Extractor",
      subtitle: "Spectrogram Generator",
      icon: Sliders,
      color: "#8b5cf6",
      details: "Computes n_mels=80 mel-frequency bands using n_fft=1024 and hop_length=256. Generates a memory-friendly 128x128 feature tensor.",
      group: "1. Input Pipeline"
    },
    {
      title: "ResNet50 Backbone",
      subtitle: "Stages 1 & 2 Frozen",
      icon: Layers,
      color: "#10b981",
      details: "Extracts initial spatial feature hierarchies. To conserve Apple Silicon RAM, early convolutional layers (layer1+2) are frozen during inference.",
      group: "2. Processing"
    },
    {
      title: "Dual Attention Engine",
      subtitle: "CBAM + SE Block",
      icon: Eye,
      color: "#f59e0b",
      details: "Convolutional Block Attention (CBAM) targets spatial voice correlations, while Squeeze-and-Excitation (SE) highlights forged channels dynamically.",
      group: "2. Processing"
    },
    {
      title: "Transformer Branch",
      subtitle: "Global Context",
      icon: Network,
      color: "#ec4899",
      details: "Framer layers capture long-range voice patterns over time. Configured with a memory-efficient FFN hidden dimension of 512 for rapid processing.",
      group: "3. Fusion & Head"
    },
    {
      title: "Multi-Scale Fusion",
      subtitle: "Dilated Head",
      icon: Cpu,
      color: "#06b6d4",
      details: "Replaces traditional heavy layers with dilated 3x3 convolutions (dilation 1, 2, 3) to capture multi-resolution forgery artifacts precisely.",
      group: "3. Fusion & Head"
    }
  ]

  return (
    <div className="w-full flex flex-col gap-12 md:gap-16">
      
      {/* ── Title ── */}
      <div className="text-center max-w-3xl mx-auto">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-indigo-500/20 bg-indigo-500/10 mb-6"
        >
          <Network size={14} className="text-indigo-600 dark:text-indigo-400" />
          <span className="text-xs font-bold tracking-widest text-indigo-700 dark:text-indigo-300 uppercase">Neural Architecture</span>
        </motion.div>
        
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="text-4xl md:text-5xl lg:text-6xl font-black tracking-tight text-slate-900 dark:text-slate-100 mb-6"
        >
          ResNet++ <span className="gradient-text">Hybrid Classifier</span>
        </motion.h1>
        
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-base md:text-lg text-slate-600 dark:text-slate-400 leading-relaxed font-medium"
        >
          An interactive visualization of our advanced Deep Learning framework fusing CNN convolutional extractors, self-attention mechanisms, and transformer modules.
        </motion.p>
      </div>

      {/* ── Roomy Neural Pipeline Grid ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {pipelineStages.map((stage, i) => (
          <div key={i} className="flex flex-col">
            <div className="mb-3 text-left">
              <span className="text-[10px] font-black tracking-[0.2em] text-slate-500 dark:text-slate-400 uppercase">{stage.group}</span>
            </div>
            <ArchNode
              {...stage}
              index={i + 1}
              delay={0.08 * i}
              isActive={activeIndex === i}
              onClick={() => setActiveIndex(i)}
            />
          </div>
        ))}
      </div>

      {/* ── Summary Card ── */}
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.7, delay: 0.2 }}
        className="glass p-8 md:p-12 rounded-3xl flex flex-col md:flex-row items-center gap-10 border border-slate-200 dark:border-emerald-500/20 relative overflow-hidden shadow-md"
      >
        <div className="absolute top-0 right-0 w-64 h-64 bg-emerald-500/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none"></div>
        
        <div className="flex-1 relative z-10 text-left">
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="text-emerald-500 dark:text-emerald-400" size={24} />
            <h3 className="font-bold text-2xl text-slate-900 dark:text-white tracking-tight">Hardware Acceleration Optimized</h3>
          </div>
          <p className="text-base text-slate-600 dark:text-slate-300 leading-relaxed mb-6 font-medium">
            The architecture is structurally optimized to run end-to-end inference under <strong>100 milliseconds</strong> on 8 GB unified memory devices (Apple Silicon). By applying resolution compression to spectrograms, FFN scale-downs, and freezing early feature layers, peak active memory consumption is capped below 3 GB while retaining high-fidelity forensic signals.
          </p>
          <div className="flex flex-wrap gap-4">
            <div className="px-4 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-300 text-sm font-mono font-bold">
              RAM Cap: {'<'} 3GB
            </div>
            <div className="px-4 py-2 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-600 dark:text-indigo-300 text-sm font-mono font-bold">
              Latency: {'<'} 100ms
            </div>
            <div className="px-4 py-2 rounded-lg bg-purple-500/10 border border-purple-500/20 text-purple-600 dark:text-purple-300 text-sm font-mono font-bold">
              Inference: CPU / MPS
            </div>
          </div>
        </div>
        
        <div className="flex-shrink-0 relative z-10">
          <div className="relative w-48 h-48 rounded-full border border-slate-200 dark:border-slate-800 flex items-center justify-center bg-slate-100/50 dark:bg-black/40 shadow-sm dark:shadow-[0_0_50px_rgba(16,185,129,0.15)]">
            <svg className="absolute inset-0 w-full h-full -rotate-90">
              <circle cx="96" cy="96" r="88" fill="none" stroke="currentColor" className="text-slate-200 dark:text-slate-800" strokeWidth="8" />
              <motion.circle 
                initial={{ strokeDasharray: "0 1000" }}
                whileInView={{ strokeDasharray: "547 1000" }}
                viewport={{ once: true }}
                transition={{ duration: 2, ease: "easeOut", delay: 0.5 }}
                cx="96" cy="96" r="88" fill="none" stroke="#10b981" strokeWidth="8" strokeLinecap="round" 
              />
            </svg>
            <div className="text-center">
              <div className="text-4xl font-black text-slate-900 dark:text-white tracking-tighter">99.1<span className="text-2xl text-emerald-500">%</span></div>
              <div className="text-xs text-slate-500 dark:text-slate-400 font-bold uppercase tracking-widest mt-1">Test AUC</div>
            </div>
          </div>
        </div>
      </motion.div>

    </div>
  )
}
