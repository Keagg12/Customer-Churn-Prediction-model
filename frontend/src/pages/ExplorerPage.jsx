import { useState } from 'react'
import { Download, Search } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { queryString } from '../services/api.js'
import { ErrorNotice, Loading, PageHeader, Panel, RiskBadge } from '../components/ui.jsx'
import { useApi } from '../hooks/useApi.js'

const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

export default function ExplorerPage() {
  const [filters, setFilters] = useState({ query: '', risk: ['High', 'Medium', 'Low'], min: 0, max: 100, segment: '', segmentValue: '', sort: 'probability', descending: true, limit: 100, offset: 0 })
  const navigate = useNavigate()
  const path = `/api/customers${queryString({ query: filters.query, risk_levels: filters.risk, probability_min: filters.min / 100, probability_max: filters.max / 100, segment: filters.segment, segment_values: filters.segmentValue ? [filters.segmentValue] : [], sort_by: filters.sort, descending: filters.descending, limit: filters.limit, offset: filters.offset })}`
  const { data, error, loading } = useApi(path)

  function update(key, value) { setFilters((state) => ({ ...state, [key]: value, offset: key === 'offset' ? value : 0 })) }
  function downloadCsv() {
    const columns = ['customer_id', 'probability', 'risk_level', 'suggested_action', 'top_factors']
    const rows = data?.items || []
    const attributes = Object.keys(rows[0]?.attributes || {})
    const headers = [...columns.slice(0, 3), ...attributes, ...columns.slice(3)]
    const csv = [headers.join(','), ...rows.map((row) => headers.map((column) => {
      const value = Object.hasOwn(row.attributes || {}, column) ? row.attributes[column] : row[column]
      return JSON.stringify(column === 'top_factors' ? (value || []).join('; ') : value ?? '')
    }).join(','))].join('\n')
    const anchor = document.createElement('a')
    anchor.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    anchor.download = 'customer_explorer_view.csv'
    anchor.click()
    URL.revokeObjectURL(anchor.href)
  }

  const segments = data?.segment_columns || []
  const segmentValues = data?.segment_options?.[filters.segment] || []
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Scored portfolio" title="Customer Explorer" description="Search, filter, compare, and open a customer profile." actions={<button className="button button-secondary" onClick={downloadCsv} disabled={!data?.items?.length}><Download size={16} />Export CSV</button>} />
      <Panel title="Filters" note="Filters operate on model scores and the current scored customer portfolio.">
        <div className="filter-grid">
          <label className="field-label">Customer ID contains<div className="search-control"><Search size={16} /><input value={filters.query} onChange={(event) => update('query', event.target.value)} placeholder="Search ID" /></div></label>
          <label className="field-label">Risk level<select className="control" value={filters.risk.join(',')} onChange={(event) => update('risk', event.target.value ? event.target.value.split(',') : [])}><option value="High,Medium,Low">All levels</option><option value="High">High</option><option value="High,Medium">High and Medium</option><option value="Medium,Low">Medium and Low</option><option value="Low">Low</option></select></label>
          <label className="field-label">Segment<select className="control" value={filters.segment} onChange={(event) => { update('segment', event.target.value); setFilters((state) => ({ ...state, segmentValue: '', offset: 0 })) }}><option value="">All customers</option>{segments.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
          <label className="field-label">Value<select className="control" value={filters.segmentValue} onChange={(event) => update('segmentValue', event.target.value)} disabled={!filters.segment}><option value="">All values</option>{segmentValues.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label className="field-label">Minimum probability (%)<input className="control" type="number" min="0" max="100" value={filters.min} onChange={(event) => update('min', Number(event.target.value))} /></label>
          <label className="field-label">Maximum probability (%)<input className="control" type="number" min="0" max="100" value={filters.max} onChange={(event) => update('max', Number(event.target.value))} /></label>
          <label className="field-label">Sort by<select className="control" value={filters.sort} onChange={(event) => update('sort', event.target.value)}><option value="probability">Churn probability</option>{(data?.numeric_columns || []).map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
          <label className="field-label">Order<select className="control" value={String(filters.descending)} onChange={(event) => update('descending', event.target.value === 'true')}><option value="true">Highest first</option><option value="false">Lowest first</option></select></label>
          <label className="field-label">Rows<select className="control" value={filters.limit} onChange={(event) => update('limit', Number(event.target.value))}><option value={100}>100</option><option value={250}>250</option><option value={500}>500</option></select></label>
        </div>
      </Panel>
      <Panel title="Matching customers" note={data ? `${data.total.toLocaleString()} matching · showing ${data.items.length.toLocaleString()} from row ${data.offset + 1}` : 'Loading matching customers'}>
        {loading && !data ? <Loading /> : <>
          {error && <ErrorNotice message={error} />}
          {data?.action_warning && <div className="notice notice-warn">{data.action_warning}</div>}
          <div className="table-scroll"><table className="data-table"><thead><tr><th>Customer</th><th>Probability</th><th>Risk</th>{Object.keys(data?.items?.[0]?.attributes || {}).map((column) => <th key={column}>{column}</th>)}<th>Suggested action</th><th>Top risk factors</th><th /></tr></thead><tbody>
            {(data?.items || []).map((row) => <tr key={row.customer_id} onDoubleClick={() => navigate(`/customers/${encodeURIComponent(row.customer_id)}`)}><td className="mono">{row.customer_id}</td><td>{pct(row.probability)}</td><td><RiskBadge level={row.risk_level} /></td>{Object.keys(row.attributes || {}).map((column) => <td key={column}>{row.attributes[column] ?? '—'}</td>)}<td>{row.suggested_action || 'No rule matched'}</td><td>{(row.top_factors || []).join(', ') || '—'}</td><td><button className="text-button" onClick={() => navigate(`/customers/${encodeURIComponent(row.customer_id)}`)}>Open</button></td></tr>)}
          </tbody></table></div>
          <div className="table-footer"><span>{(data?.items || []).length} of {(data?.total || 0).toLocaleString()} matches on this page</span><div className="inline-actions"><button className="button button-secondary" disabled={!data || data.offset === 0 || loading} onClick={() => update('offset', Math.max(0, filters.offset - filters.limit))}>Previous</button><button className="button button-secondary" disabled={!data || filters.offset + filters.limit >= data.total || loading} onClick={() => update('offset', filters.offset + filters.limit)}>Next</button></div></div>
        </>}
      </Panel>
    </div>
  )
}
