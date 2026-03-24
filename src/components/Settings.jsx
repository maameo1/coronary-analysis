import { useState, useEffect, useRef } from 'react'
import { sv, AK, ZUK, ZKK } from '../utils'

function download(filename, content, type) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

function toBibtex(papers) {
  return papers.map((p, i) => {
    const key = (p.authors?.[0]?.split(' ').pop() || 'Unknown').toLowerCase() + (p.published?.slice(0, 4) || '') + String.fromCharCode(97 + (i % 26))
    const authors = (p.authors || []).join(' and ')
    const lines = ['@article{' + key + ',']
    lines.push('  title = {' + (p.title || '') + '},')
    if (authors) lines.push('  author = {' + authors + '},')
    if (p.published) lines.push('  year = {' + p.published.slice(0, 4) + '},')
    if (p.venue) lines.push('  journal = {' + p.venue + '},')
    if (p.abstract) lines.push('  abstract = {' + p.abstract.replace(/[{}]/g, '') + '},')
    if (p.sourceId?.match(/^10\./)) lines.push('  doi = {' + p.sourceId + '},')
    if (p.notes) lines.push('  note = {' + p.notes.replace(/[{}]/g, '') + '},')
    lines.push('}')
    return lines.join('\n')
  }).join('\n\n')
}

function toCsv(papers) {
  const esc = (s) => '"' + (s || '').replace(/"/g, '""') + '"'
  const headers = ['Title', 'Authors', 'Year', 'Venue', 'DOI/ID', 'Source', 'TLDR', 'Tags', 'Key Contributions', 'Methods', 'Limitations', 'Open Questions', 'Relevance', 'Notes', 'Read Status', 'Starred', 'Abstract']
  const rows = papers.map(p => {
    const s = p.summary || {}
    return [
      esc(p.title), esc((p.authors || []).join('; ')), esc(p.published?.slice(0, 4)),
      esc(p.venue), esc(p.sourceId), esc(p.source), esc(s.tldr),
      esc((s.tags || []).join('; ')), esc((s.key_contributions || []).join('; ')),
      esc((s.methods || []).join('; ')), esc((s.limitations || []).join('; ')),
      esc((s.open_questions || []).join('; ')), esc(s.relevance_to_medical_imaging),
      esc(p.notes), esc(p.readStatus), esc(p.starred ? 'Yes' : 'No'), esc(p.abstract),
    ].join(',')
  })
  return headers.join(',') + '\n' + rows.join('\n')
}

export default function Settings({ show, onClose, apiKey, setApiKey, zUid, setZUid, zKey, setZKey, zotImp, zotL, papers, setPapers, gap, setGap }) {
  const [tAK, setTAK] = useState(apiKey)
  const [tZU, setTZU] = useState(zUid)
  const [tZK, setTZK] = useState(zKey)
  const [msg, setMsg] = useState('')
  const restoreRef = useRef(null)

  useEffect(() => {
    if (show) { setTAK(apiKey); setTZU(zUid); setTZK(zKey); setMsg('') }
  }, [show, apiKey, zUid, zKey])

  if (!show) return null

  const isty = { width: '100%', padding: '10px 14px', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text-secondary)', fontSize: 14, fontFamily: 'var(--mono)', outline: 'none', boxSizing: 'border-box', marginBottom: 12 }
  const secHead = (label, color) => ({ fontSize: 13, color, fontFamily: 'var(--mono)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 })

  function exportBackup() {
    const data = { version: 1, exportedAt: new Date().toISOString(), papers: papers, gap: gap }
    const date = new Date().toISOString().slice(0, 10)
    download('researchhub-backup-' + date + '.json', JSON.stringify(data, null, 2), 'application/json')
    setMsg('Backup exported! (' + papers.length + ' papers)')
    setTimeout(() => setMsg(''), 5000)
  }

  function importBackup(file) {
    if (!file) return
    const reader = new FileReader()
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result)
        if (!data.papers || !Array.isArray(data.papers)) throw new Error('Invalid backup file')
        const count = data.papers.length
        if (!confirm('This will replace your current library (' + papers.length + ' papers) with the backup (' + count + ' papers). Continue?')) return
        setPapers(data.papers)
        if (data.gap) setGap(data.gap)
        setMsg('Restored ' + count + ' papers from backup!')
        setTimeout(() => setMsg(''), 5000)
      } catch (err) { setMsg('Import failed: ' + err.message) }
    }
    reader.readAsText(file)
    if (restoreRef.current) restoreRef.current.value = ''
  }

  function exportBibtex() {
    const bib = toBibtex(papers)
    const date = new Date().toISOString().slice(0, 10)
    download('researchhub-' + date + '.bib', bib, 'text/plain')
    setMsg('BibTeX exported! (' + papers.length + ' entries)')
    setTimeout(() => setMsg(''), 5000)
  }

  function exportCsv() {
    const csv = toCsv(papers)
    const date = new Date().toISOString().slice(0, 10)
    download('researchhub-' + date + '.csv', csv, 'text/csv')
    setMsg('CSV exported with summaries and notes!')
    setTimeout(() => setMsg(''), 5000)
  }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 12, padding: 32, width: 520, maxWidth: '90vw', maxHeight: '90vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24 }}>
          <h3 style={{ fontSize: 18, color: 'var(--text-primary)', fontWeight: 400, margin: 0 }}>Settings</h3>
          <span onClick={onClose} style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: 18 }}>✕</span>
        </div>

        {msg && <div style={{ padding: '8px 12px', background: '#0f1a15', border: '1px solid #1a3329', borderRadius: 6, fontSize: 12, color: 'var(--accent-green-light)', fontFamily: 'var(--mono)', marginBottom: 16 }}>{msg}</div>}

        {/* API Key */}
        <div style={{ marginBottom: 24 }}>
          <div style={secHead('Anthropic API Key', 'var(--accent-green-light)')}>Anthropic API Key</div>
          <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 8, lineHeight: 1.5 }}>Get at <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener" style={{ color: 'var(--accent-blue)' }}>console.anthropic.com</a></div>
          <input value={tAK} onChange={e => setTAK(e.target.value)} placeholder="sk-ant-..." type="password" style={isty} />
          <button onClick={() => { const clean = tAK.replace(/[^\x20-\x7E]/g, '').trim(); setApiKey(clean); sv(AK, clean); setTAK(clean) }} className="btn-primary" style={{ width: '100%', textAlign: 'center' }}>{apiKey ? '✓ Update' : 'Save Key'}</button>
        </div>

        {/* Zotero */}
        <div style={{ marginBottom: 24, borderTop: '1px solid var(--border)', paddingTop: 20 }}>
          <div style={secHead('Zotero', 'var(--accent-blue)')}>Zotero</div>
          <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 8, lineHeight: 1.5 }}>Get API key at <a href="https://www.zotero.org/settings/keys" target="_blank" rel="noopener" style={{ color: 'var(--accent-blue)' }}>zotero.org/settings/keys</a></div>
          <label style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>User ID</label>
          <input value={tZU} onChange={e => setTZU(e.target.value)} placeholder="e.g. 8968548" style={isty} />
          <label style={{ fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--mono)' }}>API Key</label>
          <input value={tZK} onChange={e => setTZK(e.target.value)} placeholder="Zotero API key" type="password" style={isty} />
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={() => { setZUid(tZU); setZKey(tZK); sv(ZUK, tZU); sv(ZKK, tZK) }} className="btn-sec" style={{ flex: 1, textAlign: 'center' }}>Save</button>
            <button onClick={() => { setZUid(tZU); setZKey(tZK); sv(ZUK, tZU); sv(ZKK, tZK); onClose(); zotImp() }} disabled={!tZU || !tZK || zotL} className="btn-primary" style={{ flex: 1, textAlign: 'center', background: '#1a2a33', color: 'var(--accent-blue)', borderColor: '#2a4a5a', opacity: (!tZU || !tZK) ? 0.4 : 1 }}>{zotL ? 'Importing...' : 'Import Zotero'}</button>
          </div>
        </div>

        {/* Backup & Restore */}
        <div style={{ marginBottom: 24, borderTop: '1px solid var(--border)', paddingTop: 20 }}>
          <div style={secHead('Backup & Restore', 'var(--accent-purple)')}>Backup & Restore</div>
          <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 12, lineHeight: 1.5 }}>Save a full backup of your library including summaries, notes, figures, schematics, and gap analysis. Restore anytime.</div>
          <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <button onClick={exportBackup} className="btn-primary" style={{ flex: 1, textAlign: 'center', background: '#1a1a2e', color: 'var(--accent-purple)', borderColor: '#2a2a4a' }}>📦 Export Full Backup</button>
            <button onClick={() => restoreRef.current?.click()} className="btn-sec" style={{ flex: 1, textAlign: 'center' }}>📂 Restore from Backup</button>
          </div>
          <input ref={restoreRef} type="file" accept=".json" onChange={e => { if (e.target.files?.[0]) importBackup(e.target.files[0]) }} style={{ display: 'none' }} />
          <div style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)' }}>Backup includes: {papers.length} papers, {papers.filter(p => p.summary).length} summaries, {papers.filter(p => p.notes).length} with notes, {papers.filter(p => p.figure).length} figures</div>
        </div>

        {/* Export */}
        <div style={{ marginBottom: 24, borderTop: '1px solid var(--border)', paddingTop: 20 }}>
          <div style={secHead('Export Citations & Notes', '#5bc4b0')}>Export Citations & Notes</div>
          <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 12, lineHeight: 1.5 }}>Export your library for use in other tools. BibTeX for citation managers, CSV for spreadsheets.</div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={exportBibtex} className="btn-sec" style={{ flex: 1, textAlign: 'center' }}>📝 Export BibTeX</button>
            <button onClick={exportCsv} className="btn-sec" style={{ flex: 1, textAlign: 'center' }}>📊 Export CSV</button>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginTop: 6 }}>CSV includes: titles, authors, abstracts, TLDR, tags, methods, limitations, notes, read status</div>
        </div>

        {/* Reset */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <button onClick={() => { if (confirm('Clear all papers and data? Export a backup first!')) { localStorage.removeItem('rh_papers'); localStorage.removeItem('rh_gap'); window.location.reload() } }} className="btn-sec" style={{ width: '100%', textAlign: 'center', color: 'var(--accent-red)', borderColor: '#3a1a1a' }}>Reset Library</button>
        </div>
      </div>
    </div>
  )
}
