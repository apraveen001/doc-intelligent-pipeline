import { useState } from 'react'
import './PaperCards.css'

function formatDate(dateStr) {
  if (!dateStr) return '—'
  const d = new Date(dateStr)
  if (Number.isNaN(d.getTime())) return dateStr
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' })
}

function truncate(text, length) {
  if (!text) return ''
  if (text.length <= length) return text
  return `${text.slice(0, length).trimEnd()}…`
}

function PaperCard({ paper }) {
  const [expanded, setExpanded] = useState(false)
  const isError = paper.status === 'error'
  const snippet = truncate(paper.abstract, 220)
  const hasMore = (paper.abstract || '').length > snippet.length

  return (
    <article className={`paper-card ${isError ? 'paper-card--error' : ''}`}>
      <header className="paper-card__header">
        <h3 className="paper-card__title">{paper.title}</h3>
        {paper.authors && <p className="paper-card__authors">{paper.authors}</p>}
      </header>

      <div className="paper-card__meta mono">
        <span>{formatDate(paper.published)}</span>
        <span className="paper-card__meta-dot">·</span>
        <span>{paper.arxiv_id}</span>
        <span className="paper-card__meta-dot">·</span>
        <span>{paper.chunks_stored ?? 0} chunks</span>
        {isError && (
          <>
            <span className="paper-card__meta-dot">·</span>
            <span className="paper-card__error-tag">ingest failed</span>
          </>
        )}
      </div>

      {paper.abstract && (
        <div className="paper-card__abstract">
          <p className={`paper-card__abstract-text ${expanded ? 'is-expanded' : ''}`}>
            {expanded ? paper.abstract : snippet}
          </p>
          {hasMore && (
            <button
              type="button"
              className="paper-card__toggle"
              onClick={() => setExpanded((v) => !v)}
              aria-expanded={expanded}
            >
              {expanded ? 'Show less' : 'Read more'}
            </button>
          )}
        </div>
      )}
    </article>
  )
}

/**
 * Grid of paper cards returned by the research phase's "done" event.
 */
export default function PaperCards({ papers }) {
  if (!papers || papers.length === 0) {
    return null
  }

  return (
    <div className="paper-cards">
      {papers.map((paper) => (
        <PaperCard key={paper.arxiv_id} paper={paper} />
      ))}
    </div>
  )
}
