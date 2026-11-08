import { useState } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'
const THRESHOLD = 70

function UploadBox({ title, hint, file, preview, onChange }) {
  return (
    <label className="upload-box">
      <span className="upload-title">{title}</span>
      <span className="upload-hint">{hint}</span>
      <div className="upload-area">
        {preview ? (
          <img src={preview} alt={title} className="upload-preview" />
        ) : (
          <span className="upload-placeholder">Click to choose an image</span>
        )}
      </div>
      <span className="upload-file">{file ? file.name : 'No file selected'}</span>
      <input
        type="file"
        accept="image/png,image/jpeg"
        onChange={(e) => onChange(e.target.files[0])}
        hidden
      />
    </label>
  )
}

function App() {
  const [refFile, setRefFile] = useState(null)
  const [docFile, setDocFile] = useState(null)
  const [refPreview, setRefPreview] = useState('')
  const [docPreview, setDocPreview] = useState('')
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')
  const [error, setError] = useState('')
  const [score, setScore] = useState(null)

  const pick = (file, setFile, setPreview) => {
    setFile(file || null)
    setPreview(file ? URL.createObjectURL(file) : '')
    setScore(null)
    setError('')
  }

  const pollResult = async (jobId) => {
    for (let i = 0; i < 60; i++) {
      const s = await fetch(`${API_URL}/status/${jobId}`).then((r) => r.json())
      const state = s.status || s.state
      setStatus(state)
      if (state === 'completed') {
        return fetch(`${API_URL}/result/${jobId}`).then((r) => r.json())
      }
      if (state === 'failed') throw new Error('The job failed')
      await new Promise((r) => setTimeout(r, 2000))
    }
    throw new Error('Timed out waiting for the result')
  }

  const verify = async () => {
    if (!refFile || !docFile) {
      setError('Please choose both images.')
      return
    }
    setLoading(true)
    setError('')
    setScore(null)
    setStatus('uploading')
    try {
      const form = new FormData()
      form.append('image1', refFile) // reference signature
      form.append('image2', docFile) // full document
      const res = await fetch(`${API_URL}/verify`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Upload failed (${res.status})`)
      const job = await res.json()
      const result = await pollResult(job.job_id)
      setScore(Number(result.score ?? result.similarity_score ?? 0))
      setStatus('completed')
    } catch (e) {
      setError(e.message)
      setStatus('')
    } finally {
      setLoading(false)
    }
  }

  const isMatch = score !== null && score >= THRESHOLD

  return (
    <div className="page">
      <header className="header">
        <h1>Signature Verification</h1>
        <p>Upload a reference signature and a document. We find the signature in the document and compare the two.</p>
      </header>

      <main className="card">
        <div className="uploads">
          <UploadBox
            title="1. Reference signature"
            hint="The known, genuine signature"
            file={refFile}
            preview={refPreview}
            onChange={(f) => pick(f, setRefFile, setRefPreview)}
          />
          <UploadBox
            title="2. Document to verify"
            hint="The full document containing a signature"
            file={docFile}
            preview={docPreview}
            onChange={(f) => pick(f, setDocFile, setDocPreview)}
          />
        </div>

        <button className="verify-btn" onClick={verify} disabled={loading}>
          {loading ? 'Verifying...' : 'Verify signature'}
        </button>

        {status && status !== 'completed' && (
          <p className="status">Status: {status}</p>
        )}
        {error && <p className="error">{error}</p>}

        {score !== null && (
          <div className={`result ${isMatch ? 'match' : 'nomatch'}`}>
            <div className="score">{score.toFixed(1)}<span> / 100</span></div>
            <div className="bar"><div className="bar-fill" style={{ width: `${Math.min(score, 100)}%` }} /></div>
            <div className="verdict">
              {isMatch ? 'Signatures match' : 'Signatures do not match'}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App