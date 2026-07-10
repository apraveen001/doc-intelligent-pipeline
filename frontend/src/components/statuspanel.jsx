import './StatusPanel.css'

/**
 * Maps a backend node name to a short label + icon glyph. Falls back
 * gracefully for any node name we don't explicitly recognize, since the
 * pipeline graph may grow nodes over time.
 */
const NODE_META = {
  planner: { label: 'Planning', glyph: '◆' },
  plan: { label: 'Planning', glyph: '◆' },
  search: { label: 'Searching ArXiv', glyph: '◈' },
  arxiv_search: { label: 'Searching ArXiv', glyph: '◈' },
  select: { label: 'Selecting papers', glyph: '◇' },
  paper_selection: { label: 'Selecting papers', glyph: '◇' },
  ingest: { label: 'Ingesting papers', glyph: '◉' },
  ingestion: { label: 'Ingesting papers', glyph: '◉' },
  retrieve: { label: 'Retrieving chunks', glyph: '◈' },
  retrieval: { label: 'Retrieving chunks', glyph: '◈' },
  generate: { label: 'Generating answer', glyph: '◆' },
  generation: { label: 'Generating answer', glyph: '◆' },
  validate: { label: 'Validating grounding', glyph: '◎' },
  validation: { label: 'Validating grounding', glyph: '◎' }
}

function metaFor(node) {
  const key = (node || '').toLowerCase().trim()
  return NODE_META[key] || { label: humanize(node), glyph: '○' }
}

function humanize(node) {
  if (!node) return 'Working'
  return node
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}

function statusClass(status) {
  const s = (status || '').toLowerCase()
  if (s.includes('error') || s.includes('fail')) return 'is-error'
  if (s.includes('done') || s.includes('complete') || s.includes('success')) return 'is-done'
  if (s.includes('start') || s.includes('progress') || s.includes('running')) return 'is-active'
  return 'is-active'
}

/**
 * events: array of { node, status, detail, timestamp }
 * active: whether the stream is still open (drives the pulsing dot on the last item)
 * title: small label above the timeline (e.g. "Building your paper set")
 */
export default function StatusPanel({ events, active, title }) {
  if (!events || events.length === 0) {
    return null
  }

  return (
    <div className="status-panel">
      {title && <div className="status-panel__title">{title}</div>}
      <ol className="status-panel__timeline">
        {events.map((evt, i) => {
          const meta = metaFor(evt.node)
          const isLast = i === events.length - 1
          const cls = statusClass(evt.status)
          const pulsing = isLast && active && cls === 'is-active'

          return (
            <li key={`${evt.node}-${evt.timestamp}-${i}`} className={`status-item ${cls}`}>
              <span className={`status-item__marker ${pulsing ? 'status-item__marker--pulse' : ''}`}>
                <span className="status-item__glyph">{meta.glyph}</span>
              </span>
              <div className="status-item__body">
                <div className="status-item__label">{meta.label}</div>
                {evt.detail && <div className="status-item__detail">{evt.detail}</div>}
              </div>
            </li>
          )
        })}
      </ol>
    </div>
  )
}
