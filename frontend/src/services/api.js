// services/api.js
//
// All network access for DocMind lives here: plain fetch calls for the
// request/response endpoints, and small EventSource wrappers for the two
// SSE streams (research + QA). Nothing else in the app should reach for
// `fetch` or `EventSource` directly.

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

/**
 * Thrown for any non-2xx HTTP response, or a network-level failure,
 * so callers can branch on `.status` if they need to.
 */
export class ApiError extends Error {
  constructor(message, status = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function parseErrorDetail(response) {
  try {
    const data = await response.json()
    if (typeof data?.detail === 'string') return data.detail
    if (data?.detail) return JSON.stringify(data.detail)
  } catch {
    // body wasn't JSON, fall through
  }
  return `Request failed with status ${response.status}`
}

/**
 * POST /session/start
 * @returns {Promise<{ session_id: string }>}
 */
export async function startSession() {
  let response
  try {
    response = await fetch(`${API_URL}/session/start`, { method: 'POST' })
  } catch (err) {
    throw new ApiError('Could not reach the DocMind server. Is the backend running?')
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return response.json()
}

/**
 * DELETE /session/{sessionId}
 * @returns {Promise<{ message: string, session_id: string, chunks_deleted: number }>}
 */
export async function endSession(sessionId) {
  let response
  try {
    response = await fetch(`${API_URL}/session/${encodeURIComponent(sessionId)}`, {
      method: 'DELETE'
    })
  } catch (err) {
    throw new ApiError('Could not reach the DocMind server while ending the session.')
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return response.json()
}

/**
 * GET /session/{sessionId}/papers
 * @returns {Promise<{ session_id: string, papers: Paper[] }>}
 */
export async function getSessionPapers(sessionId) {
  let response
  try {
    response = await fetch(`${API_URL}/session/${encodeURIComponent(sessionId)}/papers`)
  } catch (err) {
    throw new ApiError('Could not reach the DocMind server while loading papers.')
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return response.json()
}

/**
 * Builds the export/download URL for a session. The caller is responsible
 * for triggering the download (see App.jsx / ExportButton.jsx), since a PDF
 * download is best done via an anchor tag rather than fetch + blob for
 * broad browser compatibility.
 */
export function getExportUrl(sessionId) {
  return `${API_URL}/session/${encodeURIComponent(sessionId)}/export`
}

/**
 * GET /health
 * @returns {Promise<{ status: string, sessions_active: number }>}
 */
export async function getHealth() {
  const response = await fetch(`${API_URL}/health`)
  if (!response.ok) {
    throw new ApiError(await parseErrorDetail(response), response.status)
  }
  return response.json()
}

// ---------------------------------------------------------------------------
// SSE streaming helpers
// ---------------------------------------------------------------------------
//
// Both /research/stream and /qa/stream emit the same envelope shape:
//   { type: "node_event", node, status, detail, timestamp }
//   { type: "done", ...payload }
//   { type: "error", detail }
//
// These helpers open an EventSource, route parsed messages to callbacks,
// and return a `close()` function so components can tear the connection
// down on unmount or when a new stream starts.

function buildStreamUrl(path, params) {
  const url = new URL(`${API_URL}${path}`)
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, value)
    }
  })
  return url.toString()
}

/**
 * Opens a generic SSE connection and dispatches parsed JSON messages.
 *
 * @param {string} url - full stream URL
 * @param {object} handlers
 * @param {(evt: object) => void} handlers.onNodeEvent - called for type "node_event"
 * @param {(evt: object) => void} handlers.onDone - called for type "done"
 * @param {(evt: object) => void} handlers.onError - called for type "error" or connection failure
 * @returns {() => void} close - call to close the EventSource
 */
function openStream(url, { onNodeEvent, onDone, onError } = {}) {
  const source = new EventSource(url)
  let closed = false

  const close = () => {
    if (!closed) {
      closed = true
      source.close()
    }
  }

  source.onmessage = (event) => {
    let payload
    try {
      payload = JSON.parse(event.data)
    } catch (err) {
      onError?.({ type: 'error', detail: 'Received a malformed message from the server.' })
      return
    }

    switch (payload.type) {
      case 'node_event':
        onNodeEvent?.(payload)
        break
      case 'done':
        onDone?.(payload)
        close()
        break
      case 'error':
        onError?.(payload)
        close()
        break
      default:
        // Unknown event type — ignore rather than break the stream.
        break
    }
  }

  source.onerror = () => {
    // EventSource fires a generic error both for network drops and for the
    // server closing the connection normally. If we haven't already closed
    // (e.g. via a "done"/"error" message), surface a connection-level error.
    if (!closed) {
      onError?.({
        type: 'error',
        detail: 'Lost connection to the server. Please try again.'
      })
      close()
    }
  }

  return close
}

/**
 * GET /research/stream?session_id=&topic=
 * @returns {() => void} close function
 */
export function streamResearch(sessionId, topic, handlers) {
  const url = buildStreamUrl('/research/stream', { session_id: sessionId, topic })
  return openStream(url, handlers)
}

/**
 * GET /qa/stream?session_id=&question=
 * @returns {() => void} close function
 */
export function streamQA(sessionId, question, handlers) {
  const url = buildStreamUrl('/qa/stream', { session_id: sessionId, question })
  return openStream(url, handlers)
}

export { API_URL }
