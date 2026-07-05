import { useState } from 'react'
import { getExportUrl } from '../services/api'
import './ExportButton.css'

/**
 * Triggers the PDF export/download for a session, then reports completion
 * so the parent can move to the "session ended" state — export is a
 * destructive action server-side (it wipes the session), so we only ever
 * fire it in response to a direct click, and we confirm before doing so
 * since it can't be undone.
 */
export default function ExportButton({ sessionId, onExported, disabled }) {
  const [isExporting, setIsExporting] = useState(false)
  const [error, setError] = useState(null)

  async function handleExport() {
    if (!sessionId || disabled || isExporting) return

    const confirmed = window.confirm(
      'Exporting will end this session — the papers and conversation will be cleared afterward. Download the PDF now?'
    )
    if (!confirmed) return

    setError(null)
    setIsExporting(true)

    try {
      const url = getExportUrl(sessionId)
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`Export failed with status ${response.status}`)
      }
      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)

      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = `docmind-session-${sessionId}.pdf`
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      URL.revokeObjectURL(objectUrl)

      onExported?.()
    } catch (err) {
      setError('Could not export the session. Please try again.')
    } finally {
      setIsExporting(false)
    }
  }

  return (
    <div className="export-button">
      <button
        type="button"
        className="export-button__trigger"
        onClick={handleExport}
        disabled={disabled || isExporting}
      >
        {isExporting ? 'Exporting…' : 'Export as PDF'}
      </button>
      {error && <p className="export-button__error">{error}</p>}
    </div>
  )
}
