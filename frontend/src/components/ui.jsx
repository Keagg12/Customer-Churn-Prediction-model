import { AlertCircle, LoaderCircle, Search } from 'lucide-react'

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <header className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        {description && <p className="page-description">{description}</p>}
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </header>
  )
}

export function Panel({ title, note, action, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(title || action) && (
        <div className="panel-heading">
          <div>
            {title && <h2>{title}</h2>}
            {note && <p>{note}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  )
}

export function Metric({ label, value, note, accent = false }) {
  return (
    <div className={`metric ${accent ? 'metric-accent' : ''}`}>
      <span className="metric-label">{label}</span>
      <strong>{value ?? '—'}</strong>
      {note && <span className="metric-note">{note}</span>}
    </div>
  )
}

export function RiskBadge({ level }) {
  return <span className={`risk-badge risk-${String(level || 'none').toLowerCase()}`}>{level || '—'}</span>
}

export function Loading({ label = 'Loading data' }) {
  return <div className="state-message"><LoaderCircle className="spin" size={18} />{label}</div>
}

export function ErrorNotice({ message, onRetry }) {
  if (!message) return null
  return (
    <div className="notice notice-error" role="alert">
      <AlertCircle size={18} />
      <span>{message}</span>
      {onRetry && <button className="text-button" onClick={onRetry}>Retry</button>}
    </div>
  )
}

export function EmptyState({ title, detail }) {
  return <div className="empty-state"><strong>{title}</strong><span>{detail}</span></div>
}

export function SearchBox(props) {
  return <label className="search-box"><Search size={16} /><input {...props} /></label>
}

export function DataTable({ columns, rows, rowKey = 'customer_id', onRowClick, empty = 'No rows to display.' }) {
  if (!rows?.length) return <EmptyState title="No matching rows" detail={empty} />
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr>{columns.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row[rowKey] ?? index} onClick={() => onRowClick?.(row)} className={onRowClick ? 'clickable-row' : ''}>
              {columns.map((column) => <td key={column.key}>{column.render ? column.render(row[column.key], row) : (row[column.key] ?? '—')}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Button({ icon: Icon, children, variant = 'secondary', ...props }) {
  return <button className={`button button-${variant}`} {...props}>{Icon && <Icon size={16} />}{children}</button>
}
