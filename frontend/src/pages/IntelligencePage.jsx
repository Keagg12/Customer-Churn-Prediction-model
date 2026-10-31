import { useEffect, useState } from 'react'
import { ArrowLeft, ArrowUpRight, CircleHelp, Search } from 'lucide-react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ErrorNotice, Loading, PageHeader, Panel, RiskBadge } from '../components/ui.jsx'
import { useApi } from '../hooks/useApi.js'

export default function IntelligencePage() {
  const { customerId } = useParams()
  const navigate = useNavigate()
  const [lookup, setLookup] = useState('')
  const [selectedId, setSelectedId] = useState('')
  const suggestions = useApi(customerId ? null : '/api/customers?sort_by=probability&descending=true&limit=100')
  const activeId = customerId || selectedId || suggestions.data?.items?.[0]?.customer_id
  const profile = useApi(activeId ? `/api/customers/${encodeURIComponent(activeId)}` : null)
  const { data, error, loading } = profile
  const inputError = customerId ? error : suggestions.error || error

  useEffect(() => {
    if (!customerId && !selectedId && suggestions.data?.items?.length) {
      setSelectedId(suggestions.data.items[0].customer_id)
    }
  }, [customerId, selectedId, suggestions.data])

  function findCustomer(event) {
    event.preventDefault()
    if (lookup.trim()) navigate(`/customers/${encodeURIComponent(lookup.trim())}`)
  }

  const contributions = data?.explanation || []
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Customer intelligence" title={data ? `Customer ${data.customer_id}` : 'Customer profile'} description="Model estimate and customer-level explanations, followed by separate business-rule recommendations." actions={<Link className="button button-secondary" to="/explorer"><ArrowLeft size={16} />Back to Explorer</Link>} />
      {!customerId && <Panel title="Find a customer" note="Search for an exact customer ID or choose from the 100 highest predicted-risk customers.">
        <form className="inline-form" onSubmit={findCustomer}><label className="field-label grow">Customer ID<input className="control" value={lookup} onChange={(event) => setLookup(event.target.value)} placeholder="Enter an exact customer ID" /></label><button className="button button-primary" disabled={!lookup.trim()}><Search size={16} />Find</button></form>
        <label className="field-label picker-label">Highest predicted risk<select className="control" value={selectedId} onChange={(event) => { setSelectedId(event.target.value); if (event.target.value) navigate(`/customers/${encodeURIComponent(event.target.value)}`) }}><option value="">Select a customer</option>{(suggestions.data?.items || []).map((item) => <option key={item.customer_id} value={item.customer_id}>{item.customer_id} · {(item.probability * 100).toFixed(1)}%</option>)}</select></label>
      </Panel>}
      {inputError && <ErrorNotice message={inputError} />}
      {loading && !data && <Loading label="Loading customer profile" />}
      {data && <>
        <section className="profile-summary">
          <div><span className="metric-label">Predicted churn probability</span><strong className="profile-probability">{(data.probability * 100).toFixed(1)}%</strong><RiskBadge level={data.risk_level} /></div>
          <div className="profile-context"><span>Scored by {data.model_name}</span><span>Higher than {(data.rank_percentile * 100).toFixed(0)}% of scored customers</span><span>Portfolio mean {(data.portfolio_mean * 100).toFixed(1)}%</span></div>
        </section>
        <div className="dashboard-grid dashboard-grid-main">
          <Panel title="Why this score" note={`Positive values raise the score; negative values lower it. Units: ${data.explanation_unit || 'not available'}. These are model contributions, not causes.`}>
            {contributions.length ? <div className="chart-wrap chart-large">
              <ResponsiveContainer width="100%" height="100%"><BarChart data={contributions.slice(0, 10)} layout="vertical" margin={{ top: 4, right: 20, bottom: 4, left: 12 }}>
                <CartesianGrid horizontal={false} stroke="#e9e5db" /><XAxis type="number" tickLine={false} axisLine={false} /><YAxis type="category" dataKey="label" width={150} tickLine={false} axisLine={false} tick={{ fontSize: 12 }} /><ReferenceLine x={0} stroke="#73766f" /><Tooltip formatter={(value) => [Number(value).toFixed(4), 'Contribution']} /><Bar dataKey="contribution" radius={[0, 2, 2, 0]}>{contributions.slice(0, 10).map((entry) => <Cell key={entry.feature} fill={entry.contribution > 0 ? '#a8412f' : '#4f7a5a'} />)}</Bar>
              </BarChart></ResponsiveContainer>
            </div> : <div className="empty-state"><strong>Explanation unavailable</strong><span>{data.explanation_error || 'No contributions were returned for this customer.'}</span></div>}
          </Panel>
          <Panel title="Retention playbook" note="Rule-based suggestions are separate from the model and have not been tested for churn reduction.">
            {data.recommendations?.actions?.length ? <div className="action-list">{data.recommendations.actions.map((action) => <article className="action-row" key={action.rule_id}><span className="action-priority">{action.rule_id}</span><div><strong>{action.action}</strong><p>{action.because}</p></div><ArrowUpRight size={17} /></article>)}</div> : <div className="empty-state"><strong>No configured rule matched</strong><span>{data.recommendations?.summary || 'No recommendation rules are available.'}</span></div>}
            {data.rule_warnings?.map((warning) => <ErrorNotice key={warning} message={warning} />)}
          </Panel>
        </div>
        <Panel title="Customer attributes" note="Values used by the active model. The typical values are drawn from the current scored portfolio.">
          <div className="attribute-grid">{Object.entries(data.attributes || {}).map(([key, value]) => <div className="attribute" key={key}><span>{data.descriptions?.[key] || key.replace(/([a-z])([A-Z])/g, '$1 $2')}</span><strong>{value == null ? '—' : String(value)}</strong>{data.explanation.find((item) => item.feature === key)?.typical != null && <small>Typical: {String(data.explanation.find((item) => item.feature === key).typical)}</small>}</div>)}</div>
        </Panel>
        <div className="notice notice-info"><CircleHelp size={17} /><span>Model contributions describe what the fitted model relies on for this score. They do not prove why the customer churned or that changing an input will change the outcome.</span></div>
      </>}
    </div>
  )
}
