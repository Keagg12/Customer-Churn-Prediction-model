import { Activity, ChartNoAxesCombined, CircleHelp, Database, Gauge, House, UserRoundSearch, Users } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { ErrorNotice } from '../components/ui.jsx'

const links = [
  { to: '/overview', label: 'Overview', icon: House },
  { to: '/explorer', label: 'Customer Explorer', icon: Users },
  { to: '/customers', label: 'Customer Intelligence', icon: UserRoundSearch },
  { to: '/model', label: 'Model Insights', icon: ChartNoAxesCombined },
  { to: '/retention', label: 'Retention Center', icon: Activity },
  { to: '/workspace', label: 'Prediction Workspace', icon: Database },
]

export default function AppLayout({ bootstrap, startupError, onRetry, children }) {
  const ready = bootstrap?.ready
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <NavLink className="brand" to="/overview" aria-label="Churn Intelligence home">
          <span className="brand-mark"><Gauge size={19} /></span>
          <span><strong>Churn Intelligence</strong><small>Predict · Explain · Retain</small></span>
        </NavLink>
        <div className="nav-caption">Workspace</div>
        <nav className="primary-nav" aria-label="Main navigation">
          {links.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} className={({ isActive }) => `nav-link ${isActive ? 'nav-active' : ''}`}>
              <Icon size={17} strokeWidth={1.8} /><span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span className={`service-dot ${ready ? 'service-ready' : ''}`} />
          <span>{ready ? 'Model and portfolio ready' : bootstrap ? 'Setup required' : 'Connecting to API'}</span>
          <CircleHelp size={15} />
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="mobile-brand"><span className="brand-mark"><Gauge size={18} /></span><strong>Churn Intelligence</strong></div>
          <div className="topbar-context">
            {bootstrap?.data?.portfolio_name ? <span>{bootstrap.data.portfolio_name}</span> : <span>Customer risk operations</span>}
            {bootstrap?.model?.name && <span className="topbar-divider">Model · {bootstrap.model.name}</span>}
          </div>
          <span className="status-label"><span className={`service-dot ${ready ? 'service-ready' : ''}`} />{ready ? 'Live data' : 'Needs setup'}</span>
        </header>
        <main className="main-content">
          {startupError && <ErrorNotice message={`API connection failed: ${startupError}`} onRetry={onRetry} />}
          {bootstrap?.errors?.map((message) => <ErrorNotice key={message} message={message} />)}
          {children}
        </main>
      </div>
    </div>
  )
}
