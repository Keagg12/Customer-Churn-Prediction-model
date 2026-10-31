import { useState } from 'react'
import { Area, AreaChart, Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ErrorNotice, Loading, Metric, PageHeader, Panel, RiskBadge } from '../components/ui.jsx'
import { useApi } from '../hooks/useApi.js'

const fmt = (value) => Number(value).toFixed(3)
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

export default function ModelPage() {
  const [fraction, setFraction] = useState(10)
  const { data, error, loading } = useApi(`/api/model-insights?fraction=${fraction / 100}`)
  if (loading && !data) return <Loading label="Loading held-out model evaluation" />

  const selected = data?.metrics?.find((metric) => metric.model === data.selected_model)
  const rocModels = [...new Set((data?.roc || []).map((point) => point.model))]
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Held-out evaluation" title="Model Insights" description="Evaluation uses only labeled customers held out from training. Scores for the unlabeled portfolio are not accuracy results." />
      {error && <ErrorNotice message={error} />}
      {data && <>
        <div className="notice notice-info">Selected: <strong>{data.selected_model}</strong> · chosen by cross-validated {data.selection_metric.replaceAll('_', ' ')}. Holdout: {data.split.train.toLocaleString()} train / {data.split.test.toLocaleString()} held out. Balancing {data.balance ? 'on' : 'off'}.</div>
        {data.warnings.map((warning) => <div className="notice notice-warn" key={warning}>{warning}</div>)}
        <div className="metric-grid metric-grid-three">
          <Metric label="ROC-AUC" value={fmt(selected?.roc_auc)} note="Ranking discrimination; 0.5 is chance." accent />
          <Metric label="Average precision" value={fmt(selected?.average_precision)} note={`Churn prevalence baseline ${fmt(data.split.churn_rate_test)}.`} />
          <Metric label="Brier score" value={fmt(selected?.brier)} note="Mean squared probability error; lower is better." />
        </div>
        <div className="dashboard-grid dashboard-grid-main">
          <Panel title="Candidate comparison" note="Classification metrics use the 30% threshold. Cross-validation average precision selects the model.">
            <div className="table-scroll"><table className="data-table"><thead><tr><th>Model</th><th>CV AP</th><th>ROC-AUC</th><th>Avg precision</th><th>Brier</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>
              {data.metrics.map((row) => <tr key={row.model} className={row.model === data.selected_model ? 'selected-row' : ''}><td>{row.model}{row.model === data.selected_model ? ' · Selected' : ''}</td><td>{fmt(row[`cv_${data.selection_metric}`])}</td><td>{fmt(row.roc_auc)}</td><td>{fmt(row.average_precision)}</td><td>{fmt(row.brier)}</td><td>{fmt(row.accuracy)}</td><td>{fmt(row.precision)}</td><td>{fmt(row.recall)}</td><td>{fmt(row.f1)}</td></tr>)}
            </tbody></table></div>
          </Panel>
          <Panel title="Confusion matrix" note={`Held-out rows at ${(data.confusion.threshold * 100).toFixed(0)}% probability threshold.`}>
            <div className="confusion-grid"><span /><span>Predicted churn</span><span>Predicted stay</span><strong>Actually churned</strong><b>{data.confusion.tp.toLocaleString()}<small>True positive</small></b><b>{data.confusion.fn.toLocaleString()}<small>False negative</small></b><strong>Actually stayed</strong><b>{data.confusion.fp.toLocaleString()}<small>False positive</small></b><b>{data.confusion.tn.toLocaleString()}<small>True negative</small></b></div>
          </Panel>
        </div>
        <div className="dashboard-grid dashboard-grid-main">
          <Panel title="ROC curves" note="Higher and farther left is better; all points use the labeled holdout.">
            <div className="chart-wrap chart-large"><ResponsiveContainer width="100%" height="100%"><LineChart margin={{ top: 8, right: 16, bottom: 8, left: 8 }}><CartesianGrid stroke="#e9e5db" /><XAxis type="number" dataKey="fpr" domain={[0, 1]} tickFormatter={percent} tickLine={false} axisLine={false} /><YAxis type="number" dataKey="tpr" domain={[0, 1]} tickFormatter={percent} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => percent(value)} /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#b6b1a7" strokeDasharray="4 4" />{rocModels.map((name, index) => <Line key={name} data={data.roc.filter((point) => point.model === name)} dataKey="tpr" name={name} type="stepAfter" stroke={['#b64b36', '#20231f', '#4f7a5a', '#b7862b'][index % 4]} dot={false} isAnimationActive={false} />)}</LineChart></ResponsiveContainer></div>
          </Panel>
          <Panel title="Calibration" note="Observed churn rate versus mean predicted probability in ten score groups.">
            <div className="chart-wrap chart-large"><ResponsiveContainer width="100%" height="100%"><LineChart data={data.calibration} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}><CartesianGrid stroke="#e9e5db" /><XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} tickFormatter={percent} tickLine={false} axisLine={false} /><YAxis dataKey="observed_rate" type="number" domain={[0, 1]} tickFormatter={percent} tickLine={false} axisLine={false} /><Tooltip formatter={(value) => percent(value)} /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#b6b1a7" strokeDasharray="4 4" /><Line dataKey="observed_rate" type="monotone" stroke="#b64b36" dot={{ r: 3 }} isAnimationActive={false} /></LineChart></ResponsiveContainer></div>
          </Panel>
        </div>
        <div className="dashboard-grid dashboard-grid-main">
          <Panel title="Risk-band check" note="Observed outcomes inside each band on held-out data.">
            <div className="table-scroll"><table className="data-table"><thead><tr><th>Band</th><th>Customers</th><th>Portfolio share</th><th>Observed churn</th><th>Share of churners</th></tr></thead><tbody>{data.bands.map((row) => <tr key={row.risk_level}><td><RiskBadge level={row.risk_level} /></td><td>{row.customers.toLocaleString()}</td><td>{percent(row.share_of_customers)}</td><td>{percent(row.observed_churn_rate)}</td><td>{percent(row.share_of_churners)}</td></tr>)}</tbody></table></div>
            <label className="field-label slider-label">Top-scored share <strong>{fraction}%</strong><input type="range" min="1" max="50" value={fraction} onChange={(event) => setFraction(Number(event.target.value))} /></label>
            <p className="small-copy">Top {fraction}% ({data.top_fraction.customers.toLocaleString()} customers) churned at {percent(data.top_fraction.observed_churn_rate)} and held {percent(data.top_fraction.share_of_churners)} of held-out churners. This measures ranking, not action effectiveness.</p>
          </Panel>
          <Panel title="What the selected model relies on" note="Mean absolute contribution on the currently scored portfolio; descriptive, not causal.">
            {data.importance.length ? <div className="chart-wrap chart-large"><ResponsiveContainer width="100%" height="100%"><BarChart data={data.importance.slice(0, 10)} layout="vertical" margin={{ top: 4, right: 14, bottom: 4, left: 10 }}><CartesianGrid horizontal={false} stroke="#e9e5db" /><XAxis type="number" tickFormatter={(value) => `${(value * 100).toFixed(0)}%`} tickLine={false} axisLine={false} /><YAxis type="category" dataKey="label" width={150} tickLine={false} axisLine={false} tick={{ fontSize: 11 }} /><Tooltip formatter={(value) => `${(Number(value) * 100).toFixed(2)}%`} /><Bar dataKey="value" fill="#20231f" radius={[0, 2, 2, 0]}>{data.importance.slice(0, 10).map((item, index) => <Cell key={item.feature} fill={index === 0 ? '#b64b36' : '#20231f'} />)}</Bar></BarChart></ResponsiveContainer></div> : <div className="empty-state"><strong>Importance unavailable</strong><span>{data.warnings.join(' ')}</span></div>}
          </Panel>
        </div>
        <Panel title="Limitations"><ul className="limitation-list"><li>There is no date field, so evaluation is a random holdout, not a future-period test.</li><li>Feature contributions describe model reliance, not causal reasons for churn.</li><li>Retention rules are heuristics; their impact has not been measured.</li><li>Risk-band boundaries are operational defaults and are not proven optimal.</li></ul></Panel>
      </>}
    </div>
  )
}
