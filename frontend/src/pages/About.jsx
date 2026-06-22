import { motion, useScroll, useTransform } from 'framer-motion'
import { useRef } from 'react'
import { Award, ShieldAlert, Cpu, Heart, CheckCircle2, ChevronDown, Lock, Code2, Globe } from 'lucide-react'

function StoryCard({ children, icon: Icon, title, index, total }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.8, delay: 0.1 }}
      className="relative z-10 glass p-8 md:p-12 rounded-[32px] max-w-4xl mx-auto border border-slate-200 dark:border-white/5 hover:border-indigo-500/30 transition-colors duration-500 group shadow-md"
    >
      <div className="flex items-center gap-4 mb-8">
        <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 backdrop-blur-xl border border-indigo-500/20 flex items-center justify-center text-indigo-600 dark:text-indigo-400 group-hover:scale-110 group-hover:bg-indigo-500/20 transition-all duration-500 shadow-sm">
          <Icon size={24} />
        </div>
        <div className="text-slate-400 dark:text-slate-500 font-mono text-xs tracking-widest uppercase font-bold">
          Phase 0{index} / 0{total}
        </div>
      </div>

      <div className="text-left">
        <h3 className="text-2xl md:text-3xl font-black mb-6 tracking-tight text-slate-900 dark:text-slate-100">
          <span className="gradient-text">{title}</span>
        </h3>
        <div className="text-slate-600 dark:text-slate-300 leading-relaxed text-base md:text-lg font-medium space-y-6">
          {children}
        </div>
      </div>
    </motion.div>
  )
}

export default function About() {
  const containerRef = useRef(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  })

  const opacity = useTransform(scrollYProgress, [0, 0.2], [1, 0])

  return (
    <div ref={containerRef} className="relative w-full bg-transparent">
      
      {/* ── Fixed Storytelling Background Line ── */}
      <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-transparent via-indigo-500/30 to-transparent -translate-x-1/2 z-0 hidden md:block">
        <motion.div 
          className="absolute top-0 left-1/2 w-1 h-32 bg-indigo-500 -translate-x-1/2 blur-sm rounded-full"
          style={{ top: useTransform(scrollYProgress, [0, 1], ["0%", "100%"]) }}
        />
      </div>

      <div className="w-full flex flex-col gap-12">
        
        {/* ── Hero Title Section ── */}
        <div className="min-h-[70vh] flex flex-col items-center justify-center text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="mb-8 relative"
          >
            <div className="absolute inset-0 bg-indigo-500/20 blur-3xl rounded-full"></div>
            <span className="relative z-10 px-6 py-2 rounded-full border border-indigo-500/20 bg-slate-100/80 dark:bg-black/40 text-indigo-600 dark:text-indigo-300 font-bold tracking-widest text-xs uppercase flex items-center gap-2 backdrop-blur-md">
              <Globe size={14} /> System Genesis
            </span>
          </motion.div>
          
          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tighter mb-8 max-w-5xl leading-[1.1] text-slate-900 dark:text-slate-100"
          >
            Securing the Future of <br/>
            <span className="gradient-text">Voice Identity</span>
          </motion.h1>
          
          <motion.p
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="text-lg md:text-xl text-slate-600 dark:text-slate-400 max-w-2xl mx-auto leading-relaxed mb-12 font-medium"
          >
            An academic research project dedicated to building robust defenses against synthetic audio and generative AI forgery.
          </motion.p>

          <motion.div
            style={{ opacity }}
            className="flex flex-col items-center gap-3 text-slate-500 dark:text-slate-400 mt-12"
          >
            <span className="text-xs uppercase tracking-[0.3em] font-bold">Discover the Mission</span>
            <motion.div
              animate={{ y: [0, 10, 0] }}
              transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
            >
              <ChevronDown size={24} className="text-indigo-500/50" />
            </motion.div>
          </motion.div>
        </div>

        {/* ── Storytelling Chapters ── */}
        <div className="pt-4 pb-16 relative flex flex-col gap-8">
          
          <StoryCard title="The Forgery Threat" icon={ShieldAlert} index={1} total={4}>
            <p>
              As generative AI models like Voicebox and VALL-E reach unprecedented levels of fidelity, synthesizing human speech is no longer a complex task—it is trivial. 
            </p>
            <p>
              This democratization of voice cloning introduces critical security threats across multiple domains: biometric spoofing for banking access, forensic fraud in legal proceedings, and highly persuasive synthetic misinformation campaigns targeting the public.
            </p>
            <div className="p-5 rounded-2xl bg-red-500/10 border border-red-500/20 text-red-700 dark:text-red-300 text-sm mt-4 font-bold">
              <strong>Vulnerability Vector:</strong> Current acoustic-only verification systems fail to detect modern neural vocoder artifacts.
            </div>
          </StoryCard>

          <StoryCard title="Our Intelligence Solution" icon={Code2} index={2} total={4}>
            <p>
              We constructed a multi-modal analysis platform that treats audio not just as sound, but as an encrypted visual landscape. By converting raw waveforms into high-resolution Log-Mel spectrograms, we reveal the hidden spectral signatures that generative models inadvertently leave behind.
            </p>
            <p>
              This visual-acoustic approach allows us to deploy state-of-the-art Computer Vision algorithms on audio data, detecting microscopic anomalies in phase and frequency continuity that are entirely imperceptible to the human ear.
            </p>
          </StoryCard>

          <StoryCard title="Hybrid Multi-Model Architecture" icon={Cpu} index={3} total={4}>
            <p>
              Instead of relying on a single neural network, our backend engine fuses two distinct intelligence layers to maximize detection confidence:
            </p>
            <ul className="space-y-4 mt-6">
              <li className="flex items-start gap-4">
                <CheckCircle2 size={24} className="text-indigo-600 dark:text-indigo-400 flex-shrink-0 mt-1" />
                <div>
                  <strong className="text-slate-900 dark:text-white block mb-1">SVM Baseline Engine</strong>
                  <span className="text-sm text-slate-500 dark:text-slate-400 font-semibold">Analyzes high-resolution speech coefficients (MFCCs and LFCCs) to spot statistical anomalies in statistical frequency distributions.</span>
                </div>
              </li>
              <li className="flex items-start gap-4">
                <CheckCircle2 size={24} className="text-purple-600 dark:text-purple-400 flex-shrink-0 mt-1" />
                <div>
                  <strong className="text-slate-900 dark:text-white block mb-1">ResNet++ Attentional Network</strong>
                  <span className="text-sm text-slate-500 dark:text-slate-400 font-semibold">Processes spatial matrices using Convolutional Block Attention (CBAM) and sequential Transformer self-attention.</span>
                </div>
              </li>
            </ul>
          </StoryCard>

          <StoryCard title="Enterprise-Grade Capabilities" icon={Lock} index={4} total={4}>
            <p>
              The platform is engineered to deliver real-time, enterprise-grade forensic evaluations. It runs complete audio structural analysis and deep spectral pattern checking concurrently.
            </p>
            <p>
              Despite the heavy computational workload, the system is strictly optimized for constrained hardware. It achieves a <strong>99.1% F1 Score</strong> on standard voice benchmarks while maintaining fully localized processing under 100ms, entirely on Apple Silicon unified memory without dedicated GPU clusters.
            </p>
          </StoryCard>

        </div>

        {/* ── Footer ── */}
        <motion.div 
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          className="text-center text-sm text-slate-500 dark:text-slate-400 py-16 flex items-center justify-center gap-2 border-t border-slate-200 dark:border-white/5 mt-16 font-semibold"
        >
          <span>Developed with</span>
          <Heart size={14} className="text-red-500 fill-red-500 mx-1 animate-pulse" />
          <span>for the Audio Forgery Detection System Academic Project.</span>
        </motion.div>

      </div>
    </div>
  )
}
