import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Mic2, Settings, Bell, Sun, Moon, Sparkles, User, Menu, X
} from 'lucide-react'

// Components & Pages
import Background3D from './components/Background3D'
import Dashboard from './pages/Dashboard'
import Upload from './pages/Upload'
import Analysis from './pages/Analysis'
import Architecture from './pages/Architecture'
import Metrics from './pages/Metrics'
import Dataset from './pages/Dataset'
import About from './pages/About'

const NAVBAR_ITEMS = [
  { id: 'dashboard',    label: 'Dashboard' },
  { id: 'upload',       label: 'Upload' },
  { id: 'analysis',     label: 'Analysis' },
  { id: 'architecture', label: 'Architecture' },
  { id: 'metrics',      label: 'Metrics' },
  { id: 'dataset',      label: 'Dataset' },
  { id: 'about',        label: 'About' },
]

export default function App() {
  const [page, setPage] = useState('dashboard')
  const [theme, setTheme] = useState('dark')
  const [result, setResult] = useState(null)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [notifCount, setNotifCount] = useState(2)

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [theme])

  const navigate = (id, payload) => {
    if (payload) setResult(payload)
    setPage(id)
    setMobileOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const toggleTheme = () => {
    setTheme(t => t === 'dark' ? 'light' : 'dark')
  }

  return (
    <div className="min-h-screen relative flex flex-col transition-colors duration-300">
      
      {/* ── 3D Constellation Layer ── */}
      <Background3D theme={theme} />

      {/* ── Ambient Glowing Blobs ── */}
      <div className="blob-container">
        <div className="blob-indigo" />
        <div className="blob-purple" />
      </div>

      {/* ── Floating Premium Navbar ── */}
      <header className="sticky top-4 z-50 w-full px-4 sm:px-6 md:px-8">
        <div className="max-w-[1440px] mx-auto glass rounded-2xl border border-slate-200/50 dark:border-slate-800/50 shadow-lg px-6 h-[88px] flex items-center justify-between backdrop-blur-xl">
          
          {/* Left: AI Logo */}
          <div className="flex items-center gap-3 cursor-pointer" onClick={() => navigate('dashboard')}>
            <motion.div
              whileHover={{ rotate: 15 }}
              className="w-10 h-10 rounded-xl flex items-center justify-center glow-accent bg-gradient-to-br from-indigo-500 to-purple-600"
            >
              <Mic2 size={18} color="white" />
            </motion.div>
            <div className="hidden sm:block">
              <h1 className="font-extrabold text-sm tracking-wide text-slate-800 dark:text-slate-100 flex items-center gap-1.5">
                DEEPVOICE <Sparkles size={11} className="text-purple-500 dark:text-purple-400 animate-pulse" />
              </h1>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 font-semibold tracking-widest uppercase">Forensic Node</p>
            </div>
          </div>

          {/* Center: Nav links (desktop) */}
          <nav className="hidden lg:flex items-center gap-1">
            {NAVBAR_ITEMS.map(({ id, label }) => {
              const active = page === id
              return (
                <button
                  key={id}
                  onClick={() => navigate(id)}
                  className={`px-4 py-2 text-xs font-bold tracking-wider uppercase transition-colors relative cursor-pointer ${
                    active ? 'text-indigo-600 dark:text-indigo-400' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                  }`}
                >
                  {label}
                  {active && (
                    <motion.div
                      layoutId="nav-underline"
                      className="absolute bottom-0 left-4 right-4 h-0.5 bg-gradient-to-r from-indigo-500 to-purple-600"
                    />
                  )}
                </button>
              )
            })}
          </nav>

          {/* Right: Controls */}
          <div className="flex items-center gap-3">
            
            {/* Theme switch button */}
            <button
              onClick={toggleTheme}
              className="w-9 h-9 rounded-xl flex items-center justify-center border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-100/50 dark:bg-slate-900/10 cursor-pointer theme-switch-btn"
              title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Mode`}
            >
              {theme === 'dark' ? (
                <Sun size={16} className="text-amber-400" />
              ) : (
                <Moon size={16} className="text-indigo-600" />
              )}
            </button>

            {/* Notification bell */}
            <button
              onClick={() => setNotifCount(0)}
              className="w-9 h-9 rounded-xl flex items-center justify-center border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-100/50 dark:bg-slate-900/10 cursor-pointer relative"
              title="Forensic updates"
            >
              <Bell size={16} className="text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200" />
              {notifCount > 0 && (
                <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-rose-500 text-[9px] font-extrabold flex items-center justify-center text-white">
                  {notifCount}
                </span>
              )}
            </button>

            {/* Profile Avatar */}
            <div className="w-8 h-8 rounded-full border border-slate-200 dark:border-slate-800 flex items-center justify-center bg-slate-100/50 dark:bg-slate-900/50 cursor-pointer hover:border-slate-300 dark:hover:border-slate-600">
              <User size={14} className="text-slate-600 dark:text-slate-400" />
            </div>

            {/* Mobile menu trigger */}
            <button
              onClick={() => setMobileOpen(t => !t)}
              className="lg:hidden w-9 h-9 rounded-xl flex items-center justify-center border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-100/50 dark:bg-slate-900/10 cursor-pointer"
            >
              {mobileOpen ? <X size={18} className="text-slate-600 dark:text-slate-300" /> : <Menu size={18} className="text-slate-600 dark:text-slate-300" />}
            </button>

          </div>

        </div>
      </header>

      {/* ── Mobile menu overlay ── */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden fixed top-28 left-4 right-4 glass z-40 border border-slate-200/50 dark:border-slate-800/50 rounded-2xl overflow-hidden shadow-xl"
          >
            <div className="flex flex-col p-4 gap-1">
              {NAVBAR_ITEMS.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => navigate(id)}
                  className={`w-full py-3 px-4 text-left font-bold text-xs uppercase tracking-wider rounded-xl transition-colors ${
                    page === id ? 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 border border-indigo-500/20' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-900/50'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Container Layout System ── */}
      <main className="flex-1 page-container section-padding relative z-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={page}
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            transition={{ duration: 0.35 }}
          >
            {page === 'dashboard'    && <Dashboard navigate={navigate} />}
            {page === 'upload'       && <Upload navigate={navigate} />}
            {page === 'analysis'     && <Analysis result={result} navigate={navigate} />}
            {page === 'architecture' && <Architecture />}
            {page === 'metrics'      && <Metrics />}
            {page === 'dataset'      && <Dataset />}
            {page === 'about'        && <About />}
          </motion.div>
        </AnimatePresence>
      </main>

    </div>
  )
}

