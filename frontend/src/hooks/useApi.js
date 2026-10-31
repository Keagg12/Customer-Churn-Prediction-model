import { useContext, useEffect, useState } from 'react'
import { AppContext } from '../AppContext.jsx'
import { api } from '../services/api.js'

export function useApi(path, options = {}) {
  const { refreshKey = 0 } = useContext(AppContext)
  const [state, setState] = useState({ data: null, error: '', loading: Boolean(path) })

  useEffect(() => {
    if (!path) return undefined
    const controller = new AbortController()
    setState((current) => ({ ...current, loading: true, error: '' }))
    api(path, { signal: controller.signal, ...options })
      .then((data) => setState({ data, error: '', loading: false }))
      .catch((error) => {
        if (error.name !== 'AbortError') setState({ data: null, error: error.message, loading: false })
      })
    return () => controller.abort()
  }, [path, refreshKey])

  return state
}
