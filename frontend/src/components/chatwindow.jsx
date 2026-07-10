import { useEffect, useRef, useState } from 'react'
import StatusPanel from './StatusPanel'
import './ChatWindow.css'

/**
 * turns: array of {
 *   id, question,
 *   answer, sources, isGrounded,
 *   events, status: 'streaming' | 'done' | 'error',
 *   errorDetail
 * }
 */
export default function ChatWindow({ turns, onAsk, disabled, streamingTurnId }) {
  const [question, setQuestion] = useState('')
  const scrollRef = useRef(null)
  const textareaRef = useRef(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [turns])

  function handleSubmit(e) {
    e.preventDefault()
    const q = question.trim()
    if (!q || disabled) return
    onAsk(q)
    setQuestion('')
    textareaRef.current?.focus()
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  return (
    <div className="chat-window">
      <div className="chat-window__scroll" ref={scrollRef}>
        {turns.length === 0 ? (
          <div className="chat-window__empty">
            Ask a question about the papers DocMind just read. Answers are grounded in the
            retrieved chunks, with sources listed underneath.
          </div>
        ) : (
          <div className="chat-window__turns">
            {turns.map((turn) => (
              <Turn key={turn.id} turn={turn} isStreaming={turn.id === streamingTurnId} />
            ))}
          </div>
        )}
      </div>

      <form className="chat-window__composer" onSubmit={handleSubmit}>
        <textarea
          ref={textareaRef}
          className="chat-window__input"
          placeholder={disabled ? 'Waiting for papers to finish loading…' : 'Ask a question about these papers…'}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
        />
        <button type="submit" className="chat-window__send" disabled={disabled || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

function Turn({ turn, isStreaming }) {
  return (
    <div className="turn">
      <div className="message message--user">
        <div className="message__bubble">{turn.question}</div>
      </div>

      <div className="message message--assistant">
        <div className="message__bubble">
          {turn.status === 'streaming' && (
            <StatusPanel events={turn.events} active={isStreaming} />
          )}

          {turn.status === 'error' && (
            <div className="turn__error">
              {turn.errorDetail || 'Something went wrong answering this question.'}
            </div>
          )}

          {turn.status === 'done' && (
            <>
              <p className="turn__answer">{turn.answer}</p>

              {turn.isGrounded === false && (
                <div className="turn__grounding-warning">
                  This answer may not be fully grounded in the retrieved papers.
                </div>
              )}

              {turn.sources && turn.sources.length > 0 && (
                <div className="turn__sources">
                  <span className="turn__sources-label">Sources</span>
                  <div className="turn__sources-chips">
                    {turn.sources.map((src, i) => (
                      <span key={`${src}-${i}`} className="source-chip mono">
                        {src}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
