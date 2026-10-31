import { useContext, useState } from 'react'
import { ArrowRight, Download, RefreshCw } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, CartesianGrid, Cell, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ErrorNotice, Loading, Metric, PageHeader, Panel, RiskBadge } from '../components/ui.jsx'
import { AppContext } from '../AppContext.jsx'
import { useApi } from '../hooks/useApi.js'

const RISK_COLORS = { Low: '#4f7a5a', Medium: '#b7862b', High: '#a8412f' }
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

export default function OverviewPage() {
  const { refresh } = useContext(AppContext)
  const { data, error, loading } = useApi('/api/overview')
  const [feature, setFeature] = useState('')
  if (loading && !data) return <Loading label="Loading portfolio overview" />
  if (error) return <ErrorNotice message={error} />
  if (!data) return null

  const featureName = feature || data.patterns?.[0]?.feature || ''
  const groups = data.pattern_groups?.filter((item) => item.feature === featureName) || []
  const riskBands = data.predicted?.risk_bands || []
  const mean = data.predicted?.mean_probability
  const download = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'churn-overview.json'
    anchor.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="page-stack">
      <PageHeader eyebrow="Portfolio intelligence" title="Overview" description="Observed churn history and predicted portfolio risk, shown separately."
        actions={<><button className="icon-button" title="Refresh data" onClick={refresh}><RefreshCw size={17} /></button><button className="icon-button" title="Download overview" onClick={download}><Download size={17} /></button></>} />
      {data.stale_model && <div className="notice notice-warn">Training data changed after this model was saved. The current model may be stale.</div>}
      {data.in_sample && <div className="notice notice-warn">No unlabeled portfolio was found, so training rows are shown in-sample. These predictions are not a new-customer evaluation.</div>}
      <div className="metric-grid metric-grid-three">
        <Metric label="Observed churn" value={data.observed ? percent(data.observed.rate) : 'Not available'} note={data.observed ? `${data.observed.customers.toLocaleString()} labeled customers · ${data.observed.churned.toLocaleString()} churned` : 'Add labeled training data'} />
        <Metric label="High-risk customers" value={riskBands.find((band) => band.risk_level === 'High')?.customers?.toLocaleString() ?? '—'} note={riskBands.find((band) => band.risk_level === 'High') ? `${percent(riskBands.find((band) => band.risk_level === 'High').share)} of scored portfolio` : 'Model score required'} accent />
        <Metric label="Average predicted probability" value={mean == null ? '—' : percent(mean)} note={`${(data.predicted?.customers || 0).toLocaleString()} customers scored`} />
      </div>
      <div className="dashboard-grid dashboard-grid-main">
        <Panel title="Risk distribution" note="Predicted probability grouped by the configured operational risk bands.">
          <div className="chart-wrap chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={riskBands} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid vertical={false} stroke="#e9e5db" />
                <XAxis dataKey="risk_level" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => value.toLocaleString()} />
                <Tooltip formatter={(value, name) => [Number(value).toLocaleString(), name === 'customers' ? 'Customers' : name]} />
                <Bar dataKey="customers" radius={[2, 2, 0, 0]}>
                  {riskBands.map((entry) => <Cell key={entry.risk_level} fill={RISK_COLORS[entry.risk_level]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="band-summary">{riskBands.map((band) => <span key={band.risk_level}><RiskBadge level={band.risk_level} />{band.customers.toLocaleString()} · {percent(band.share)}</span>)}</div>
        </Panel>
        <Panel title="Where churn concentrates" note="Observed historical rate by customer group. Association is not causation." action={
          <select className="control control-compact" value={featureName} onChange={(event) => setFeature(event.target.value)} aria-label="Feature to compare">
            {(data.patterns || []).map((item) => <option key={item.feature} value={item.feature}>{item.feature.replace(/([a-z])([A-Z])/g, '$1 $2')}</option>)}
          </select>
        }>
          {groups.length ? <div className="chart-wrap chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={groups} margin={{ top: 10, right: 16, bottom: 4, left: 0 }}>
                <CartesianGrid vertical={false} stroke="#e9e5db" />
                <XAxis dataKey="group" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => `${Math.round(value * 100)}%`} domain={[0, 'auto']} />
                <Tooltip formatter={(value, name) => [percent(value), name === 'churn_rate' ? 'Observed churn' : name]} />
                {data.observed?.rate != null && <ReferenceLine y={data.observed.rate} stroke="#73766f" strokeDasharray="4 4" />}
                <Bar dataKey="churn_rate" name="Observed churn" fill="#20231f" radius={[2, 2, 0, 0]} />
                <Line type="monotone" dataKey="lift" name="Lift" hide />
              </ComposedChart>
            </ResponsiveContainer>
          </div> : <div className="empty-state"><strong>No historical groups</strong><span>Load labeled history to compare churn rates.</span></div>}
        </Panel>
      </div>
      <Panel title="Predicted probability distribution" note="Number of scored customers in each probability interval; colors show the configured risk band.">
        {data.predicted?.histogram?.length ? <div className="chart-wrap chart-medium">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.predicted.histogram} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
              <CartesianGrid vertical={false} stroke="#e9e5db" />
              <XAxis dataKey="low" tickFormatter={(value) => `${Math.round(value * 100)}%`} interval={3} tickLine={false} axisLine={false} />
              <YAxis tickFormatter={(value) => value.toLocaleString()} tickLine={false} axisLine={false} />
              <Tooltip formatter={(value) => [Number(value).toLocaleString(), 'Customers']} labelFormatter={(value) => `${percent(value)} probability`} />
              <Bar dataKey="customers" radius={[1, 1, 0, 0]}>
                {data.predicted.histogram.map((item, index) => <Cell key={`${item.low}-${index}`} fill={RISK_COLORS[item.risk_level]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div> : <div className="empty-state"><strong>No predictions</strong><span>Set up a model and scoring portfolio to see this distribution.</span></div>}
      </Panel>
      <Panel title="Highest predicted risk" note="Top customers by model score. A score estimates risk; it is not an observed outcome." action={<Link className="text-link" to="/retention">Open Retention Center <ArrowRight size={15} /></Link>}>
        {data.predicted?.attention?.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Customer</th><th>Probability</th><th>Risk</th><th>Main risk-raising factors</th><th /></tr></thead><tbody>
          {data.predicted.attention.map((customer) => <tr key={customer.customer_id}><td className="mono">{customer.customer_id}</td><td>{percent(customer.probability)}</td><td><RiskBadge level={customer.risk_level} /></td><td>{(customer.why || []).join(', ') || '—'}</td><td><Link className="text-link" to={`/customers/${encodeURIComponent(customer.customer_id)}`}>Profile <ArrowRight size={14} /></Link></td></tr>)}
        </tbody></table></div> : <div className="empty-state"><strong>No scores available</strong><span>Set up labeled data and a scoring portfolio in Prediction Workspace.</span><Link className="text-link" to="/workspace">Open Workspace <ArrowRight size={15} /></Link></div>}
      </Panel>
      {(data.errors || []).map((message) => <ErrorNotice key={message} message={message} />)}
    </div>
  )
}
