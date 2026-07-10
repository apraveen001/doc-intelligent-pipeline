import { useState } from 'react'
import './TopicSearch.css'

const PRESET_TOPICS = [
  'LLM Reasoning',
  'Computer Vision',
  'Diffusion Models',
  'Reinforcement Learning',
  'Transformers',
  'RAG Systems',
  'Multimodal AI',
  'Graph Neural Networks',
  'Robotics',
  'Protein Folding'
]

/**
 * Landing search — free-text topic input plus a horizontally scrollable
 * row of preset chips. Both paths call the same onSearch(topic).
 */
export default function TopicSearch({ onSearch, disabled }) {
  const [value, setValue] = useState('')

  function handleSubmit(e) {
    e.preventDefault()
    const topic = value.trim()
    if (!topic || disabled) return
    onSearch(topic)
  }

  function handleChipClick(topic) {
    if (disabled) return
    setValue(topic)
    onSearch(topic)
  }

  return (
    <div className="topic-search">
      <div className="topic-search__intro">
        <h1 className="topic-search__title">DocMind</h1>
        <p className="topic-search__subtitle">
          Pick a research topic. DocMind will find, read, and ground itself in a handful of
          ArXiv papers before you start asking questions.
        </p>
      </div>

      <form className="topic-search__form" onSubmit={handleSubmit}>
        <input
          type="text"
          className="topic-search__input"
          placeholder="Search a topic, e.g. sparse mixture-of-experts routing"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={disabled}
          aria-label="Research topic"
        />
        <button
          type="submit"
          className="topic-search__submit"
          disabled={disabled || !value.trim()}
        >
          Search
        </button>
      </form>

      <div className="topic-search__chips" role="list" aria-label="Preset topics">
        {PRESET_TOPICS.map((topic) => (
          <button
            key={topic}
            type="button"
            role="listitem"
            className="topic-chip"
            onClick={() => handleChipClick(topic)}
            disabled={disabled}
          >
            {topic}
          </button>
        ))}
      </div>
    </div>
  )
}
