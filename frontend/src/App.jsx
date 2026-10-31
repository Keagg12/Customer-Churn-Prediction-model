import { lazy, Suspense, useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { AppContext } from './AppContext.jsx'
import AppLayout from './layouts/AppLayout.jsx'
import { Loading } from './components/ui.jsx'
import { api } from './services/api.js'

const OverviewPage = lazy(() => import('./pages/OverviewPage.jsx'))
const ExplorerPage = lazy(() => import('./pages/ExplorerPage.jsx'))
const IntelligencePage = lazy(() => import('./pages/IntelligencePage.jsx'))
const ModelPage = lazy(() => import('./pages/ModelPage.jsx'))
const RetentionPage = lazy(() => import('./pages/RetentionPage.jsx'))
const WorkspacePage = lazy(() => import('./pages/WorkspacePage.jsx'))

export default function App() {
  const [bootstrap, setBootstrap] = useState(null)
  const [error, setError] = useState('')
  const [refreshKey, setRefreshKey] = useState(0)
  const refresh = () => setRefreshKey((value) => value + 1)

  useEffect(() => {
    let alive = true
    api('/api/bootstrap')
      .then((data) => { if (alive) { setBootstrap(data); setError('') } })
      .catch((reason) => { if (alive) setError(reason.message) })
    return () => { alive = false }
  }, [refreshKey])

  return (
    <AppContext.Provider value={{ refreshKey, refresh }}>
      <AppLayout bootstrap={bootstrap} startupError={error} onRetry={refresh}>
        <Suspense fallback={<Loading label="Loading page" />}>
          <Routes>
            <Route path="/overview" element={<OverviewPage />} />
            <Route path="/explorer" element={<ExplorerPage />} />
            <Route path="/customers" element={<IntelligencePage />} />
            <Route path="/customers/:customerId" element={<IntelligencePage />} />
            <Route path="/model" element={<ModelPage />} />
            <Route path="/retention" element={<RetentionPage />} />
            <Route path="/workspace" element={<WorkspacePage />} />
            <Route path="*" element={<Navigate to="/overview" replace />} />
          </Routes>
        </Suspense>
      </AppLayout>
    </AppContext.Provider>
  )
}
