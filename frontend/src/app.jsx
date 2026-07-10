import { useCallback, useEffect, useRef, useState } from 'react'
import TopicSearch from './components/TopicSearch'
import PaperCards from './components/PaperCards'
import StatusPanel from './components/StatusPanel'
import ChatWindow from './components/ChatWindow'
import ExportButton from './components/ExportButton'
import { startSession, endSession, streamResearch, streamQA, ApiError } from './services/api'
import './App.css'

// Phases the app moves through, in order. `ended` is a terminal state
// reached only via export (which wipes the session server-side).
const PHASE = {
  LANDING: 'landing',
  RESEARCHING: 'researching',
  READY: 'ready',
  ENDED: 'ended'
}

let turnCounter = 0
function nextTurnId() {
  turnCounter += 1
  return `turn-${turnCounter}`
}

export default function App() {
  const [phase, setPhase] = useState(PHASE.LANDING)
  const [sessionId, setSessionId] = useState(null)
  const [topic, setTopic] = useState('')

  const [researchEvents, setResearchEvents] = useState([])
  const [researchActive, setResearchActive] = useState(false)
  const [papers, setPapers] = useState([])

  const [turns, setTurns] = useState([])
  const [streamingTurnId, setStreamingTurnId] = useState(null)

  const [globalError, setGlobalError] = useState(null)

  // Keep references to the "close" functions of any open EventSource so we
  // can tear them down on unmount or when a new stream starts.
  const closeResearchStreamRef = useRef(null)
  const closeQaStreamRef = useRef(null)

  // Close any open SSE connections when the component unmounts.
  useEffect(() => {
    return () => {
      closeResearchStreamRef.current?.()
      closeQaStreamRef.current?.()
    }
  }, [])

  const resetToLanding = useCallback(() => {
    closeResearchStreamRef.current?.()
    closeQaStreamRef.current?.()
    closeResearchStreamRef.current = null
    closeQaStreamRef.current = null

    setPhase(PHASE.LANDING)
    setSessionId(null)
    setTopic('')
    setResearchEvents([])
    setResearchActive(false)
    setPapers([])
    setTurns([])
    setStreamingTurnId(null)
    setGlobalError(null)
  }, [])

  const runResearch = useCallback(async (chosenTopic) => {
    setGlobalError(null)
    setTopic(chosenTopic)
    setResearchEvents([])
    setPapers([])
    setPhase(PHASE.RESEARCHING)
    setResearchActive(true)

    let newSessionId
    try {
      const res = await startSession()
      newSessionId = res.session_id
      setSessionId(newSessionId)
    } catch (err) {
      const message =
        err instanceof ApiError ? err.message : 'Could not start a new session. Please try again.'
      setGlobalError(message)
      setPhase(PHASE.LANDING)
      setResearchActive(false)
      return
    }

    closeResearchStreamRef.current = streamResearch(newSessionId, chosenTopic, {
      onNodeEvent: (evt) => {
        setResearchEvents((prev) => [...prev, evt])
      },
      onDone: (payload) => {
        setResearchActive(false)
        if (payload.error) {
          setGlobalError(payload.error)
          setPhase(PHASE.LANDING)
          return
        }
        setPapers(payload.papers || [])
        setPhase(PHASE.READY)
      },
      onError: (payload) => {
        setResearchActive(false)
        setGlobalError(payload.detail || 'Something went wrong while researching this topic.')
        setPhase(PHASE.LANDING)
      }
    })
  }, [])

  const askQuestion = useCallback(
    (question) => {
      if (!sessionId) return

      const turnId = nextTurnId()
      setTurns((prev) => [
        ...prev,
        {
          id: turnId,
          question,
          events: [],
          status: 'streaming',
          answer: null,
          sources: [],
          isGrounded: null,
          errorDetail: null
        }
      ])
      setStreamingTurnId(turnId)

      // Only one QA stream should be open at a time; a fresh question closes
      // any stream still lingering from a previous (unlikely, but possible)
      // in-flight request.
      closeQaStreamRef.current?.()

      closeQaStreamRef.current = streamQA(sessionId, question, {
        onNodeEvent: (evt) => {
          setTurns((prev) =>
            prev.map((t) => (t.id === turnId ? { ...t, events: [...t.events, evt] } : t))
          )
        },
        onDone: (payload) => {
          setStreamingTurnId(null)
          setTurns((prev) =>
            prev.map((t) =>
              t.id === turnId
                ? {
                    ...t,
                    status: payload.error ? 'error' : 'done',
                    answer: payload.answer,
                    sources: payload.sources || [],
                    isGrounded: payload.is_grounded,
                    errorDetail: payload.error
                  }
                : t
            )
          )
        },
        onError: (payload) => {
          setStreamingTurnId(null)
          setTurns((prev) =>
            prev.map((t) =>
              t.id === turnId
                ? {
                    ...t,
                    status: 'error',
                    errorDetail: payload.detail || 'Something went wrong answering this question.'
                  }
                : t
            )
          )
        }
      })
    },
    [sessionId]
  )

  const handleExported = useCallback(() => {
    closeQaStreamRef.current?.()
    setPhase(PHASE.ENDED)
  }, [])

  const handleEndSessionWithoutExport = useCallback(async () => {
    if (!sessionId) {
      resetToLanding()
      return
    }
    try {
      await endSession(sessionId)
    } catch {
      // Even if the delete call fails (e.g. session already gone), the user
      // still wants to leave — proceed to reset locally regardless.
    }
    resetToLanding()
  }, [sessionId, resetToLanding])

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <span className="app__brand-mark" aria-hidden="true" />
          <span className="app__brand-name">DocMind</span>
        </div>
        {phase === PHASE.READY && sessionId && (
          <div className="app__header-actions">
            <span className="app__session-id mono">session · {sessionId.slice(0, 8)}</span>
            <ExportButton sessionId={sessionId} onExported={handleExported} />
          </div>
        )}
      </header>

      <main className="app__main">
        {globalError && (
          <div className="app__banner app__banner--error">
            {globalError}
            <button
              type="button"
              className="app__banner-dismiss"
              onClick={() => setGlobalError(null)}
              aria-label="Dismiss error"
            >
              ×
            </button>
          </div>
        )}

        {phase === PHASE.LANDING && (
          <section className="app__section app__section--centered">
            <TopicSearch onSearch={runResearch} disabled={false} />
          </section>
        )}

        {phase === PHASE.RESEARCHING && (
          <section className="app__section app__section--centered">
            <div className="app__research-progress">
              <p className="app__research-topic">
                Researching <span className="mono">{topic}</span>
              </p>
              <StatusPanel
                events={researchEvents}
                active={researchActive}
                title="Building your paper set"
              />
            </div>
          </section>
        )}

        {phase === PHASE.READY && (
          <section className="app__section app__section--workspace">
            <div className="app__papers">
              <h2 className="app__section-title">
                Papers on <span className="mono">{topic}</span>
              </h2>
              <PaperCards papers={papers} />
            </div>

            <div className="app__chat">
              <h2 className="app__section-title">Ask DocMind</h2>
              <ChatWindow
                turns={turns}
                onAsk={askQuestion}
                disabled={streamingTurnId !== null}
                streamingTurnId={streamingTurnId}
              />
            </div>
          </section>
        )}

        {phase === PHASE.ENDED && (
          <section className="app__section app__section--centered">
            <div className="app__ended">
              <h2 className="app__ended-title">Session complete</h2>
              <p className="app__ended-body">
                Your PDF has been downloaded and the session has been cleared. Start a new
                session whenever you're ready to explore another topic.
              </p>
              <button type="button" className="app__ended-restart" onClick={resetToLanding}>
                Start a new session
              </button>
            </div>
          </section>
        )}
      </main>

      {phase === PHASE.READY && (
        <footer className="app__footer">
          <button type="button" className="app__end-session" onClick={handleEndSessionWithoutExport}>
            End session without exporting
          </button>
        </footer>
      )}
    </div>
  )
}
