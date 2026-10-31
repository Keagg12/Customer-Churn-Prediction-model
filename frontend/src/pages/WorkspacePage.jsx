import { useContext, useEffect, useMemo, useState } from 'react'
import { CheckCircle2, Download, FileUp, RotateCcw, Send, WandSparkles } from 'lucide-react'
import { AppContext } from '../AppContext.jsx'
import { Button, DataTable, ErrorNotice, Loading, PageHeader, Panel, RiskBadge } from '../components/ui.jsx'
import { useApi } from '../hooks/useApi.js'
import { api, apiDownload } from '../services/api.js'

const TABS = ['Status', 'Train', 'Score', 'Risk bands']
const number = (value) => Number(value || 0).toLocaleString()
const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`

export default function WorkspacePage() {
  const [tab, setTab] = useState('Status')
  const { data: status, error: statusError, loading } = useApi('/api/workspace/status')
  const { refresh } = useContext(AppContext)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [trainFile, setTrainFile] = useState(null)
  const [trainSummary, setTrainSummary] = useState(null)
  const [target, setTarget] = useState('')
  const [positive, setPositive] = useState([])
  const [idColumn, setIdColumn] = useState('')
  const [features, setFeatures] = useState([])
  const [modelName, setModelName] = useState('')
  const [testSize, setTestSize] = useState(0.2)
  const [seed, setSeed] = useState(42)
  const [balance, setBalance] = useState(false)
  const [persist, setPersist] = useState(false)
  const [trainingResult, setTrainingResult] = useState(null)
  const [scoreFile, setScoreFile] = useState(null)
  const [scoreResult, setScoreResult] = useState(null)
  const [bands, setBands] = useState({ low_max: 0.15, high_min: 0.3 })
  const [bandResult, setBandResult] = useState('')

  const columns = trainSummary?.columns || (status?.schema ? [...new Set([status.schema.target, status.schema.id_column, ...status.schema.feature_columns].filter(Boolean))] : [])
  const targetCounts = trainSummary?.target_counts || status?.target_counts || {}
  const targetOptions = useMemo(() => Object.keys(targetCounts[target] || {}), [targetCounts, target])
  const featureChoices = columns.filter((column) => column !== target && column !== idColumn)

  useEffect(() => {
    if (!status) return
    setBands(status.bands || { low_max: 0.15, high_min: 0.3 })
    if (status.schema) {
      setTarget(status.schema.target)
      setPositive(status.schema.positive_labels.map(String))
      setIdColumn(status.schema.id_column || '')
      setFeatures(status.schema.feature_columns)
    }
  }, [status])

  useEffect(() => setPersist(!trainSummary), [trainSummary])

  useEffect(() => {
    if (!trainSummary) return
    const configured = status?.schema
    const nextTarget = configured?.target && trainSummary.columns.includes(configured.target) ? configured.target : ''
    setTarget(nextTarget)
    setPositive(nextTarget && configured?.positive_labels ? configured.positive_labels.map(String).filter((value) => Object.hasOwn(trainSummary.target_counts?.[nextTarget] || {}, value)) : [])
    setIdColumn(configured?.id_column && trainSummary.columns.includes(configured.id_column) ? configured.id_column : '')
    setFeatures(trainSummary.columns.filter((column) => column !== nextTarget && column !== configured?.id_column))
  }, [trainSummary])

  async function submitTrainUpload(file) {
    if (!file) return
    setBusy(true); setError(''); setTrainingResult(null)
    try {
      const form = new FormData(); form.append('file', file)
      const summary = await api('/api/workspace/train-upload', { method: 'POST', body: form })
      setTrainFile(file); setTrainSummary(summary)
    } catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }

  async function trainModel(event) {
    event.preventDefault(); setBusy(true); setError(''); setTrainingResult(null)
    try {
      const result = await api('/api/workspace/train', {
        method: 'POST',
        body: JSON.stringify({ target, positive_labels: positive, feature_columns: features, id_column: idColumn || null, test_size: Number(testSize), seed: Number(seed), balance, persist, model_name: modelName || null }),
      })
      setTrainingResult(result); refresh()
    } catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }

  async function scoreCustomers(event) {
    event.preventDefault(); if (!scoreFile) return
    setBusy(true); setError(''); setScoreResult(null)
    try {
      const form = new FormData(); form.append('file', scoreFile)
      const result = await api('/api/workspace/score-upload', { method: 'POST', body: form })
      setScoreResult(result); refresh()
    } catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }

  async function downloadScores() {
    setError('')
    try { await apiDownload('/api/workspace/scores.csv', 'churn_scores.csv') }
    catch (reason) { setError(reason.message) }
  }

  async function applyBands(event) {
    event.preventDefault(); setBusy(true); setError(''); setBandResult('')
    try {
      const result = await api('/api/workspace/bands', { method: 'POST', body: JSON.stringify({ low_max: Number(bands.low_max), high_min: Number(bands.high_min) }) })
      setBandResult(`Risk bands applied: Low below ${percent(result.low_max)}, High from ${percent(result.high_min)}.`); refresh()
    } catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }

  async function resetSession() {
    setBusy(true); setError('')
    try { await api('/api/workspace/reset', { method: 'POST' }); setTrainFile(null); setTrainSummary(null); setScoreFile(null); setScoreResult(null); setTrainingResult(null); refresh() }
    catch (reason) { setError(reason.message) } finally { setBusy(false) }
  }

  if (loading && !status) return <Loading label="Checking data and model status" />
  return (
    <div className="page-stack">
      <PageHeader eyebrow="Data and model operations" title="Prediction Workspace" description="Inspect local inputs, train and evaluate models, score customer files, and adjust session risk bands." actions={<Button icon={RotateCcw} onClick={resetSession} disabled={busy}>Reset session</Button>} />
      {(error || statusError) && <ErrorNotice message={error || statusError} />}
      <div className="tab-bar" role="tablist">{TABS.map((item) => <button key={item} role="tab" aria-selected={tab === item} className={tab === item ? 'tab-active' : ''} onClick={() => setTab(item)}>{item}</button>)}</div>
      {tab === 'Status' && <StatusPanel status={status} />}
      {tab === 'Train' && <Panel title="Train and compare models" note="Training uses labeled rows only. Holdout metrics are computed on a random split of that labeled file.">
        <form className="form-stack" onSubmit={trainModel}>
          <label className="field-label">Labeled customer file (optional)<input className="control file-control" type="file" accept=".csv,.tsv,.txt,.xlsx,.xlsm,.xls" onChange={(event) => submitTrainUpload(event.target.files?.[0])} /></label>
          {trainSummary && <div className="notice notice-info"><FileUp size={17} />{trainFile?.name} · {number(trainSummary.rows)} rows · {trainSummary.columns.length} columns</div>}
          {columns.length > 0 && <>
            <div className="form-grid">
              <label className="field-label">Churn target<select required className="control" value={target} onChange={(event) => { setTarget(event.target.value); setPositive([]) }}><option value="">Choose column</option>{columns.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
              <label className="field-label">Customer ID (optional)<select className="control" value={idColumn} onChange={(event) => { setIdColumn(event.target.value); setFeatures((current) => current.filter((feature) => feature !== event.target.value)) }}><option value="">No ID column</option>{columns.filter((column) => column !== target).map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
            </div>
            <label className="field-label">Label values that mean churn<select required multiple className="control multi-select" value={positive} onChange={(event) => setPositive(Array.from(event.target.selectedOptions, (option) => option.value))}>{targetOptions.map((value) => <option key={value} value={value}>{value} ({number(targetCounts[target]?.[value])})</option>)}</select></label>
            <fieldset className="feature-fieldset"><legend>Model input columns</legend><div className="feature-grid">{featureChoices.map((column) => <label key={column} className="check-row"><input type="checkbox" checked={features.includes(column)} onChange={(event) => setFeatures((current) => event.target.checked ? [...current, column] : current.filter((item) => item !== column))} /><span>{column}</span></label>)}</div></fieldset>
          </>}
          <div className="form-grid form-grid-four">
            <label className="field-label">Held-out test share<input className="control" type="number" min="0.1" max="0.4" step="0.05" value={testSize} onChange={(event) => setTestSize(Number(event.target.value))} /></label>
            <label className="field-label">Random seed<input className="control" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label>
            <label className="field-label">Activate<select className="control" value={modelName} onChange={(event) => setModelName(event.target.value)}><option value="">Best by cross-validation</option><option>Logistic Regression</option><option>Gradient Boosting</option><option>Random Forest</option><option>Decision Tree</option></select></label>
            <label className="field-label">Class balancing<select className="control" value={String(balance)} onChange={(event) => setBalance(event.target.value === 'true')}><option value="false">Off (recommended)</option><option value="true">On</option></select></label>
          </div>
          <label className="check-row"><input type="checkbox" checked={persist} onChange={(event) => setPersist(event.target.checked)} /><span>Save as <code>models/model.joblib</code> (replaces saved package)</span></label>
          {balance && <div className="notice notice-warn">Balanced class weights can inflate predicted probabilities. Compare probability calibration before acting on scores.</div>}
          <div><Button icon={WandSparkles} variant="primary" disabled={busy || !target || !positive.length || !features.length}>{busy ? 'Training models…' : 'Train and compare'}</Button></div>
        </form>
        {trainingResult && <TrainingResult result={trainingResult} />}
      </Panel>}
      {tab === 'Score' && <Panel title="Score a customer file" note="Unlabeled customers are scored only; the endpoint does not calculate accuracy without outcomes.">
        <form className="inline-form" onSubmit={scoreCustomers}><label className="field-label grow">Customer file<input className="control file-control" type="file" accept=".csv,.tsv,.txt,.xlsx,.xlsm,.xls" onChange={(event) => setScoreFile(event.target.files?.[0] || null)} /></label><Button icon={Send} variant="primary" disabled={busy || !scoreFile}>{busy ? 'Scoring…' : 'Score file'}</Button></form>
        {scoreResult && <><div className="notice notice-success"><CheckCircle2 size={17} />{scoreResult.filename} · {number(scoreResult.rows)} rows · mean probability {percent(scoreResult.mean_probability)}</div><UploadWarnings checks={scoreResult.checks} /><RiskSummary rows={scoreResult.risk_bands} /><div className="download-row"><Button icon={Download} onClick={downloadScores}>Download all scored customers</Button></div><DataTable columns={[{ key: 'customer_id', label: 'Customer' }, { key: 'churn_probability', label: 'Probability', render: (value) => percent(value) }, { key: 'risk_level', label: 'Risk', render: (value) => <RiskBadge level={value} /> }]} rows={scoreResult.sample} empty="Scoring completed; no preview rows returned." /></>}
      </Panel>}
      {tab === 'Risk bands' && <Panel title="Session risk bands" note="These cut-offs assign operational labels; they are not proven optimal thresholds and do not change model probabilities.">
        <form className="form-grid form-grid-three" onSubmit={applyBands}>
          <label className="field-label">Low risk below (%)<input className="control" type="number" min="1" max="89" step="1" value={Math.round(bands.low_max * 100)} onChange={(event) => setBands((current) => ({ ...current, low_max: Number(event.target.value) / 100 }))} /></label>
          <label className="field-label">High risk from (%)<input className="control" type="number" min="2" max="99" step="1" value={Math.round(bands.high_min * 100)} onChange={(event) => setBands((current) => ({ ...current, high_min: Number(event.target.value) / 100 }))} /></label>
          <div className="form-action"><Button variant="primary" disabled={busy || Number(bands.low_max) >= Number(bands.high_min)}>Apply bands</Button></div>
        </form>
        {bandResult && <div className="notice notice-success"><CheckCircle2 size={17} />{bandResult}</div>}
        <p className="small-copy">Current values are session-scoped and reset when this browser session is cleared. Defaults: Low below 15%, Medium 15% to below 30%, High 30% and above.</p>
      </Panel>}
    </div>
  )
}

function StatusPanel({ status }) {
  if (!status) return <Loading />
  const rows = [
    ['Labeled history', status.train_exists ? `${number(status.labeled_rows)} rows` : 'Not found'],
    ['Scoring portfolio', status.test_exists ? 'data/test.csv' : status.portfolio_name || 'Training rows (in-sample fallback)'],
    ['Active model', status.model_name || 'No usable model'],
    ['Model saved', status.model_exists ? 'Yes' : 'No'],
    ['Model stale', status.stale_model ? 'Yes · training data changed' : 'No'],
    ['Customer rules', `${status.rules?.length || 0} configured`],
  ]
  return <Panel title="Data and model status" note="Files are read from the backend’s configured data, config, and model directories.">
    <div className="status-list">{rows.map(([label, value]) => <div className="status-row" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div>
    {status.notes?.map((message) => <div className="notice notice-warn" key={message}>{message}</div>)}
    {status.rule_warnings?.map((message) => <div className="notice notice-warn" key={message}>{message}</div>)}
  </Panel>
}

function TrainingResult({ result }) {
  return <div className="result-block"><div className="notice notice-success"><CheckCircle2 size={17} />Selected {result.model_name} · {result.saved ? 'saved to model storage' : 'session-only model'}</div><div className="table-scroll"><table className="data-table"><thead><tr><th>Model</th><th>CV average precision</th><th>ROC-AUC</th><th>Average precision</th><th>Accuracy</th><th>F1</th></tr></thead><tbody>{result.metrics.map((row) => <tr key={row.model}><td>{row.model}</td><td>{Number(row.cv_average_precision).toFixed(3)}</td><td>{Number(row.roc_auc).toFixed(3)}</td><td>{Number(row.average_precision).toFixed(3)}</td><td>{Number(row.accuracy).toFixed(3)}</td><td>{Number(row.f1).toFixed(3)}</td></tr>)}</tbody></table></div>{result.warnings.map((warning) => <div className="notice notice-warn" key={warning}>{warning}</div>)}</div>
}

function RiskSummary({ rows }) {
  return <div className="band-summary band-summary-spaced">{rows.map((row) => <span key={row.risk_level}><RiskBadge level={row.risk_level} />{number(row.customers)} · {percent(row.share)}</span>)}</div>
}

function UploadWarnings({ checks }) {
  if (!checks) return null
  const notices = [
    Object.keys(checks.missing_values || {}).length && `Missing values will be imputed: ${Object.entries(checks.missing_values).map(([key, count]) => `${key} (${count})`).join(', ')}.`,
    Object.keys(checks.non_numeric || {}).length && `Non-numeric numeric inputs will be imputed: ${Object.entries(checks.non_numeric).map(([key, count]) => `${key} (${count})`).join(', ')}.`,
    Object.keys(checks.unseen_categories || {}).length && `Unseen categories are ignored by the fitted encoder: ${Object.entries(checks.unseen_categories).map(([key, values]) => `${key}: ${values.join(', ')}`).join('; ')}.`,
  ].filter(Boolean)
  return notices.map((notice) => <div className="notice notice-warn" key={notice}>{notice}</div>)
}
