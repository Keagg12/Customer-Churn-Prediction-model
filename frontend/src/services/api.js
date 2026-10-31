const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
const SESSION_KEY = 'churn-session-id'

function sessionId() {
  let value = localStorage.getItem(SESSION_KEY)
  if (!value) {
    value = crypto.randomUUID()
    localStorage.setItem(SESSION_KEY, value)
  }
  return value
}

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export async function api(path, options = {}) {
  const headers = new Headers(options.headers || {})
  headers.set('X-Session-ID', sessionId())
  if (options.body && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json')
  }
  let response
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers })
  } catch (error) {
    if (error?.name === 'AbortError') throw error
    throw new ApiError(error instanceof Error ? error.message : 'Could not reach the API.')
  }
  const returnedSession = response.headers.get('X-Session-ID')
  if (returnedSession) localStorage.setItem(SESSION_KEY, returnedSession)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new ApiError(body.detail || `Request failed (${response.status}).`, response.status)
  }
  return body
}

export async function apiDownload(path, filename) {
  const headers = new Headers({ 'X-Session-ID': sessionId() })
  const response = await fetch(`${API_BASE}${path}`, { headers })
  const returnedSession = response.headers.get('X-Session-ID')
  if (returnedSession) localStorage.setItem(SESSION_KEY, returnedSession)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new ApiError(body.detail || `Download failed (${response.status}).`, response.status)
  }
  const blob = await response.blob()
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

export function queryString(values) {
  const query = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return
    if (Array.isArray(value)) value.forEach((item) => query.append(key, item))
    else query.set(key, String(value))
  })
  const result = query.toString()
  return result ? `?${result}` : ''
}
