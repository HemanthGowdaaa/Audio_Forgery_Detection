import { useState, useRef, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload as UploadIcon, FileAudio, X, AlertCircle,
  AlertTriangle, Clock, Cpu, Activity, Play, CheckCircle
} from 'lucide-react'

const API = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const ACCEPT = ['.wav', '.mp3', '.flac']

function Wavebar({ delay }) {
  return (
    <motion.div
      animate={{ height: [4, 26, 4] }}
      transition={{ duration: 1.2, repeat: Infinity, delay, ease: "easeInOut" }}
      className="w-1 rounded-full bg-indigo-500/80 dark:bg-indigo-400/80"
    />
  )
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
}

export default function Upload({ navigate }) {
  const [file, setFile] = useState(null)
  const [meta, setMeta] = useState(null)
  const [dragging, setDrag] = useState(false)
  const [status, setStatus] = useState('idle') // idle | uploading | analyzing | done | error
  const [errMsg, setErrMsg] = useState('')
  const [modelStatus, setModelStatus] = useState(null)
  const inputRef = useRef()

  useEffect(() => {
    fetch(`${API}/status`)
      .then(r => r.json())
      .then(d => setModelStatus(d))
      .catch(() => setModelStatus({ mode: 'unavailable', svm_loaded: false, resnet_loaded: false }))
  }, [])

  const modelsAvailable = modelStatus?.mode !== 'unavailable'

  const handleFile = useCallback(async f => {
    if (!f) return
    const ext = '.' + f.name.split('.').pop().toLowerCase()
    if (!ACCEPT.includes(ext)) {
      setErrMsg(`Unsupported format "${ext}". Supported formats are WAV, MP3, or FLAC.`)
      setStatus('error')
      return
    }
    setFile(f)
    setStatus('uploading')
    setErrMsg('')
    setMeta(null)

    const form = new FormData()
    form.append('file', f)
    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setMeta(data)
      setStatus('idle')
    } catch (e) {
      setErrMsg(e.message)
      setStatus('error')
    }
  }, [])

  const onDrop = e => {
    e.preventDefault()
    setDrag(false)
    handleFile(e.dataTransfer.files[0])
  }

  const analyze = async () => {
    if (!file) return
    setStatus('analyzing')
    setErrMsg('')
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${API}/predict`, { method: 'POST', body: form })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Prediction failed')
      setStatus('done')
      navigate('analysis', { ...data, filename: file.name, meta })
    } catch (e) {
      setErrMsg(e.message)
      setStatus('error')
    }
  }

  const reset = () => {
    setFile(null)
    setMeta(null)
    setStatus('idle')
    setErrMsg('')
  }

  const isAnalyzing = status === 'analyzing'
  const isUploading = status === 'uploading'
  const canAnalyze = !!file && !isAnalyzing && !isUploading && modelsAvailable

  return (
    <div className="max-w-3xl mx-auto flex flex-col gap-8">
      
      {/* ── Header ── */}
      <div className="text-center">
        <span className="stat-badge stat-badge-blue mb-4 inline-block">Forensic Vault</span>
        <h1 className="text-3xl font-black gradient-text tracking-tight mb-2">Spectral Analysis Node</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 max-w-sm mx-auto">
          Upload vocal waveforms and frequency streams into the multi-model intelligence cores.
        </p>
      </div>

      {/* ── Drag & Drop Zone ── */}
      <div
        className={`upload-zone p-12 text-center flex flex-col items-center justify-center min-h-64 ${dragging ? 'active' : ''} ${file ? 'has-file' : ''} ${isAnalyzing ? 'scan-overlay' : ''}`}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        onClick={() => !file && inputRef.current.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".wav,.mp3,.flac"
          className="hidden"
          onChange={e => handleFile(e.target.files[0])}
        />

        {!file ? (
          <>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5 bg-indigo-500/10 border border-indigo-500/20"
            >
              <UploadIcon size={24} className="text-indigo-600 dark:text-indigo-400" />
            </motion.div>
            <h4 className="font-bold text-base text-slate-800 dark:text-slate-200">Drag and drop audio file here</h4>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-2">WAV, MP3, FLAC (Max 100MB)</p>
          </>
        ) : (
          <div className="flex flex-col items-center gap-6 w-full" onClick={e => e.stopPropagation()}>
            
            {/* Waveform graphic */}
            <div className="flex items-end gap-1 h-8">
              {Array.from({ length: 18 }, (_, i) => (
                <Wavebar key={i} delay={(i * 0.06) % 1.2} />
              ))}
            </div>

            {/* File Info Card */}
            <div className="glass p-5 rounded-2xl flex items-center gap-4 w-full max-w-md border border-slate-200 dark:border-slate-800 bg-slate-100/50 dark:bg-slate-900/30">
              <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 bg-indigo-500/10 border border-indigo-500/25">
                <FileAudio size={18} className="text-indigo-600 dark:text-indigo-400" />
              </div>
              
              <div className="flex-1 overflow-hidden text-left">
                <h5 className="font-bold text-xs text-slate-800 dark:text-slate-200 truncate">{file.name}</h5>
                <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1 font-semibold">
                  {formatBytes(file.size)}
                  {meta && ` · ${meta.duration}s · ${(meta.sample_rate / 1000).toFixed(1)} kHz`}
                  {isUploading && ' · Decoding metadata...'}
                </p>
              </div>

              {!isAnalyzing && (
                <button
                  onClick={reset}
                  className="w-7 h-7 rounded-full flex items-center justify-center bg-rose-500/10 border border-rose-500/20 cursor-pointer"
                >
                  <X size={13} className="text-rose-600 dark:text-rose-400" />
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* ── Scanning overlay log ── */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="glass p-6 rounded-2xl flex items-start gap-4 border border-indigo-500/20"
          >
            <Activity size={18} className="text-indigo-600 dark:text-indigo-400 mt-0.5 animate-pulse" />
            <div className="text-left">
              <p className="font-bold text-xs text-slate-800 dark:text-slate-200">Executing Multi-Model Cryptanalysis...</p>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                Fusing SVM frequency coefficients and ResNet++ CBAM attention layers. Fusing logits...
              </p>
              
              {/* Progress bar */}
              <div className="progress-bar-track mt-3 w-48 bg-slate-200 dark:bg-slate-800">
                <div className="progress-bar-fill w-[88%]" />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Error message ── */}
      {status === 'error' && errMsg && (
        <div className="glass p-4 rounded-2xl flex items-start gap-3 border border-rose-500/20 bg-rose-500/5">
          <AlertCircle size={16} className="text-rose-500 dark:text-rose-400 mt-0.5 flex-shrink-0" />
          <p className="text-xs text-rose-600 dark:text-rose-400">{errMsg}</p>
        </div>
      )}

      {/* ── Actions ── */}
      <div className="flex gap-4">
        <button
          onClick={() => inputRef.current.click()}
          disabled={isAnalyzing}
          className="btn-secondary flex-1 justify-center"
        >
          Change File
        </button>
        <button
          onClick={analyze}
          disabled={!canAnalyze}
          className="btn-primary flex-1 justify-center"
        >
          {isAnalyzing ? 'Scanning...' : 'Analyze Audio'}
        </button>
      </div>

    </div>
  )
}
