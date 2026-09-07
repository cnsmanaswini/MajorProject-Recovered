import React from 'react'
import { Routes, Route, NavLink, useLocation, Navigate } from 'react-router-dom'
import {
  Home, MessageCircle, BarChart2, User,
  Brain, Sparkles, Compass, Bell
} from 'lucide-react'
import clsx from 'clsx'

import { useAuth } from './context/AuthContext.jsx'

// Pages
import AuthPage from './pages/AuthPage.jsx'
import FeedPage from './components/Feed/FeedPage.jsx'
import ExplorePage from './components/Feed/Explorepage.jsx'
import MessagesPage from './components/Messages/MessagesPage.jsx'
import DashboardPage from './components/Dashboard/DashboardPage.jsx'
import ProfilePage from './components/Profile/ProfilePage.jsx'

const NAV_ITEMS = [
  { to: '/',          icon: Home,          label: 'Feed'      },
  { to: '/explore',   icon: Compass,       label: 'Explore'   },
  { to: '/messages',  icon: MessageCircle, label: 'Messages'  },
  { to: '/dashboard', icon: BarChart2,     label: 'Insights'  },
  { to: '/profile',   icon: User,          label: 'Profile'   },
]

// ── Protected Route ───────────────────────────────────────────
function Protected({ children }) {
  const { user, loading } = useAuth()
  if (loading) return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="w-8 h-8 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
    </div>
  )
  if (!user) return <Navigate to="/auth" replace />
  return children
}

// ── Sidebar ───────────────────────────────────────────────────
function Sidebar() {
  const { user, logout } = useAuth()

  return (
    <aside className="hidden md:flex fixed left-0 top-0 h-full w-64 glass-dark border-r border-white/10
                  flex-col z-40">
      {/* Logo */}
      <div className="p-6 pb-4">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-brand-500 to-brand-700
                          flex items-center justify-center">
            <Brain size={16} className="text-white" />
          </div>
          <span className="font-display text-xl text-white tracking-tight">MindGram</span>
        </div>
        <p className="text-xs text-gray-500 mt-1.5 font-mono">social media that cares</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-2 space-y-1">
        {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) => clsx(
              'flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200',
              isActive
                ? 'bg-brand-500/15 text-brand-300 border border-brand-500/20'
                : 'text-gray-400 hover:text-white hover:bg-white/5',
            )}
          >
            <Icon size={20} />
            <span className="font-medium text-sm">{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* User */}
      {user && (
        <div className="p-4 border-t border-white/10">
          <div className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-white/5">
            <img
              src={user.avatar_url || `https://api.dicebear.com/9.x/avataaars/svg?seed=${user.username}`}
              alt={user.display_name}
              className="w-8 h-8 rounded-full bg-gray-800 border-2 border-brand-500/30"
            />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{user.display_name}</p>
              <p className="text-xs text-gray-500 truncate">@{user.username}</p>
            </div>
            <button
              onClick={logout}
              className="text-gray-500 hover:text-red-400 transition-colors text-xs"
              title="Logout"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </aside>
  )
}

// ── Mobile Nav ────────────────────────────────────────────────
function MobileNav() {
  return (
    <nav className="fixed bottom-0 left-0 right-0 glass-dark border-t border-white/10
                    flex items-center justify-around px-2 py-2 z-40 md:hidden">
      {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) => clsx(
            'flex flex-col items-center gap-0.5 px-3 py-1.5 rounded-xl transition-all',
            isActive ? 'text-brand-400' : 'text-gray-500'
          )}
        >
          <Icon size={22} />
          <span className="text-[10px]">{label}</span>
        </NavLink>
      ))}
    </nav>
  )
}

// ── App ───────────────────────────────────────────────────────
export default function App() {
  const { user } = useAuth()

  return (
    <div className="min-h-screen flex">
      {user && (
        <>
          <div className="hidden md:block w-64 flex-shrink-0" />
          <Sidebar />
        </>
      )}

      <main className="flex-1 min-h-screen pb-16 md:pb-0">
        <Routes>
          {/* Public */}
          <Route path="/auth" element={
            user ? <Navigate to="/" replace /> : <AuthPage />
          } />
          <Route path="/auth/callback" element={<AuthPage />} />

          {/* Protected */}
          <Route path="/" element={
            <Protected><FeedPage /></Protected>
          } />
          <Route path="/explore" element={
            <Protected><ExplorePage /></Protected>
          } />
          <Route path="/messages" element={
            <Protected><MessagesPage /></Protected>
          } />
          <Route path="/dashboard" element={
            <Protected><DashboardPage /></Protected>
          } />
          <Route path="/profile" element={
            <Protected><ProfilePage /></Protected>
          } />
          <Route path="/profile/:username" element={
            <Protected><ProfilePage /></Protected>
          } />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      {user && <MobileNav />}
    </div>
  )
}