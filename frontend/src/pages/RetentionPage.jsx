import { useState } from 'react'
import { Download, ExternalLink } from 'lucide-react'
import { Link } from 'react-router-dom'
import { queryString } from '../services/api.js'
import { ErrorNotice, Loading, PageHeader, Panel, RiskBadge } from '../components/ui.jsx'
import { useApi } from '../hooks/useApi.js'

const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

export default function RetentionPage() {
  const [risk, setRisk] = useState(['High'])
  const [rule, setRule] = useState('')
  const [segment, setSegment] = useState('')
  const [segmentValue, setSegmentValue] = useState('')
  const [sort, setSort] = useState('probability')
  const [limit, setLimit] = useState(100)
  const path = `/api/retention${queryString({ risk_levels: risk, rule_ids: rule ? [rule] : [], segment, segment_values: segmentValue ? [segmentValue] : [], sort_by: sort, limit })}`
  const { data, error, loading } = useApi(path)

  const download = () => {
    const rows = data?.rows || []
    const headers = ['customer_id', 'probability', 'risk_level', 'top_factors', 'actions']
    const contents = [headers.join(','), ...rows.map((row) => [row.customer_id, row.probability, row.risk_level, row.top_factors.join('; '), row.actions.join('; ')].map((value) => JSON.stringify(value)).join(','))].join('\n')
    const link = document.createElement('a')
    link.href = URL.createObjectURL(new Blob([contents], { type: 'text/csv' }))
    link.download = 'retention_queue.csv'
    link.click()
    URL.revokeObjectURL(link.href)
  }
  const segmentValues = data?.segment_options?.[segment] || []

  return (
    <div className="page-stack">
      <PageHeader eyebrow="Retention operations" title="Retention Center" description="Prioritize customers by predicted churn risk and review which configured actions apply." actions={<button className="button button-secondary" onClick={download} disabled={!data?.rows?.length}><Download size={16} />Export CSV</button>} />
      <Panel title="Queue filters" note="Rule conditions and suggested actions are business heuristics, not validated treatments.">
        <div className="filter-grid filter-grid-five">
          <label className="field-label">Risk level<select className="control" value={risk.join(',')} onChange={(event) => setRisk(event.target.value ? event.target.value.split(',') : [])}><option value="High">High</option><option value="High,Medium">High and Medium</option><option value="Medium">Medium</option><option value="High,Medium,Low">All levels</option></select></label>
          <label className="field-label">Rule<select className="control" value={rule} onChange={(event) => setRule(event.target.value)}><option value="">All matched rules</option>{(data?.rules || []).map((item) => <option key={item.id} value={item.id}>{item.id} · {item.action}</option>)}</select></label>
          <label className="field-label">Segment<select className="control" value={segment} onChange={(event) => { setSegment(event.target.value); setSegmentValue('') }}><option value="">All customers</option>{(data?.segment_columns || []).map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
          <label className="field-label">Value<select className="control" value={segmentValue} onChange={(event) => setSegmentValue(event.target.value)} disabled={!segment}><option value="">All values</option>{segmentValues.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label className="field-label">Sort by<select className="control" value={sort} onChange={(event) => setSort(event.target.value)}><option value="probability">Probability</option><option value="rules">Rules matched</option></select></label>
          <label className="field-label">Queue size<select className="control" value={limit} onChange={(event) => setLimit(Number(event.target.value))}><option value={50}>50</option><option value={100}>100</option><option value={250}>250</option><option value={500}>500</option><option value={1000}>1,000</option></select></label>
        </div>
      </Panel>
      <div className="metric-grid metric-grid-three">
        <MetricLite label="Customers shown" value={data?.shown?.toLocaleString() ?? '—'} note={data ? `${data.total_matching.toLocaleString()} risk-matching customers before rule filter` : ''} />
        <MetricLite label="Expected churn in queue" value={data ? data.expected_churn.toFixed(1) : '—'} note="Sum of probabilities, not known outcomes." accent />
        <MetricLite label="Customers with no rule" value={data?.no_rule_count?.toLocaleString() ?? '—'} note="Among the displayed queue." />
      </div>
      <Panel title="Priority queue" note={data ? `${data.shown.toLocaleString()} customers displayed · at most three actions per customer` : 'Loading queue'}>
        {loading && !data ? <Loading /> : <>
          {error && <ErrorNotice message={error} />}
          {data?.action_warning && <div className="notice notice-warn">{data.action_warning}</div>}
          <div className="table-scroll"><table className="data-table"><thead><tr><th>Customer</th><th>Probability</th><th>Risk</th><th>Main model factors</th><th>Applicable actions</th><th /></tr></thead><tbody>
            {(data?.rows || []).map((row) => <tr key={row.customer_id}><td className="mono">{row.customer_id}</td><td>{pct(row.probability)}</td><td><RiskBadge level={row.risk_level} /></td><td>{row.top_factors.slice(0, 3).join(', ') || '—'}</td><td>{row.actions.join('; ') || 'No rule matched'}</td><td><Link className="icon-link" title="Open customer" to={`/customers/${encodeURIComponent(row.customer_id)}`}><ExternalLink size={16} /></Link></td></tr>)}
          </tbody></table></div>
          <div className="rule-summary-grid">{(data?.rule_counts || []).map((item) => <div className="rule-summary" key={item.rule_id}><span className="mono">{item.rule_id}</span><strong>{item.customers}</strong><span>{item.action}</span></div>)}</div>
        </>}
      </Panel>
      {(error || data?.errors || []).length > 0 && <ErrorNotice message={error || data.errors.join(' ')} />}
    </div>
  )
}

function MetricLite({ label, value, note, accent = false }) {
  return <div className={`metric ${accent ? 'metric-accent' : ''}`}><span className="metric-label">{label}</span><strong>{value}</strong><span className="metric-note">{note}</span></div>
}
