import { useState, useEffect, useRef } from 'react'
import { PK, AK, ZUK, ZKK, GK, gid, ld, sv, genSummary, callAI, parseBib, parseCsv } from './utils'
import Settings from './components/Settings'
import DetailView from './components/DetailView'
import GraphView from './components/GraphView'
import GapView from './components/GapView'
import LibraryView from './components/LibraryView'

export default function App() {
  const [papers, setPapers] = useState(() => ld(PK, []))
  const [apiKey, setApiKey] = useState(() => localStorage.getItem(AK) || '')
  const [zUid, setZUid] = useState(() => localStorage.getItem(ZUK) || '')
  const [zKey, setZKey] = useState(() => localStorage.getItem(ZKK) || '')
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [loadingMsg, setLoadingMsg] = useState('')
  const [selected, setSelected] = useState(null)
  const [search, setSearch] = useState('')
  const [filterTag, setFilterTag] = useState(null)
  const [tab, setTab] = useState('library')
  const [error, setError] = useState(null)
  const [pdf, setPdf] = useState(null)
  const fRef = useRef(null), bRef = useRef(null)
  const [speaking, setSpeaking] = useState(false)
  const [gap, setGap] = useState(() => ld(GK, null))
  const [gapL, setGapL] = useState(false)
  const [sumL, setSumL] = useState(false)
  const [impMsg, setImpMsg] = useState('')
  const [showSet, setShowSet] = useState(false)
  const [zotL, setZotL] = useState(false)
  const [zotMsg, setZotMsg] = useState('')

  useEffect(() => {
    try { sv(PK, papers) }
    catch (e) { console.warn('Storage save failed — library may be too large for localStorage:', e) }
  }, [papers])
  useEffect(() => { if (gap) sv(GK, gap) }, [gap])

  // ── Add Paper ──────────────────────────────────────────────────────────
  async function addPaper() {
    if (!input.trim() && !pdf) return
    setLoading(true); setError(null)
    try {
      let md = {}
      if (pdf) {
        if (!apiKey) throw new Error('API key needed for PDF.')
        setLoadingMsg('Reading PDF...')
        const b64 = await new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(r.result.split(',')[1]); r.onerror = () => rej(new Error('Fail')); r.readAsDataURL(pdf) })
        setLoadingMsg('Extracting...')
        const resp = await fetch('https://api.anthropic.com/v1/messages', { method: 'POST', headers: { 'Content-Type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' }, body: JSON.stringify({ model: 'claude-sonnet-4-20250514', max_tokens: 1000, messages: [{ role: 'user', content: [{ type: 'document', source: { type: 'base64', media_type: 'application/pdf', data: b64 } }, { type: 'text', text: 'Return ONLY JSON: {"title":"...","authors":["..."],"abstract":"...","published":"YYYY","venue":"..."}' }] }] }) })
        md = JSON.parse((await resp.json()).content?.map(b => b.text || '').join('').replace(/```json|```/g, '').trim())
        md.source = 'PDF'; md.sourceId = pdf.name
      } else {
        const t = input.trim(), am = t.match(/(\d{4}\.\d{4,5})/), dm = t.match(/(10\.\d{4,}\/[^\s]+)/)
        if (am) { setLoadingMsg('arXiv...'); const r = await fetch('https://export.arxiv.org/api/query?id_list=' + am[1]); const x = new DOMParser().parseFromString(await r.text(), 'text/xml'); const e = x.querySelector('entry'); if (!e) throw new Error('Not found'); md = { title: e.querySelector('title')?.textContent?.replace(/\s+/g, ' ').trim() || '', abstract: e.querySelector('summary')?.textContent?.trim() || '', authors: [...e.querySelectorAll('author name')].map(n => n.textContent), published: e.querySelector('published')?.textContent?.slice(0, 10) || '', source: 'arXiv', sourceId: am[1] } }
        else if (dm) { setLoadingMsg('DOI...'); const r = await fetch('https://api.crossref.org/works/' + encodeURIComponent(dm[1])); if (!r.ok) throw new Error('DOI not found'); const di = (await r.json()).message; md = { title: (di.title || ['?'])[0], abstract: (di.abstract || '').replace(/<[^>]+>/g, ''), authors: (di.author || []).map(a => ((a.given || '') + ' ' + (a.family || '')).trim()), published: di.published?.['date-parts']?.[0]?.join('-') || '', venue: (di['container-title'] || [''])[0], source: 'DOI', sourceId: dm[1] } }
        else { setLoadingMsg('Searching...'); const r = await fetch('https://api.semanticscholar.org/graph/v1/paper/search?query=' + encodeURIComponent(t) + '&limit=1&fields=title,abstract,authors,year,externalIds,venue'); const d = await r.json(); if (!d.data?.length) throw new Error('Not found'); const p = d.data[0]; md = { title: p.title, abstract: p.abstract || '', authors: (p.authors || []).map(a => a.name), published: p.year ? String(p.year) : '', venue: p.venue || '', source: 'Search', sourceId: p.externalIds?.DOI || p.paperId } }
      }
      if (papers.some(p => p.title?.toLowerCase() === md.title?.toLowerCase())) { setError('Already in library.'); setLoading(false); return }
      let sum = null; if (apiKey) { try { sum = await genSummary(apiKey, md) } catch {} }
      const np = { id: gid(), ...md, summary: sum, addedAt: new Date().toISOString(), notes: '', readStatus: 'unread', figure: null, schematic: null }
      setPapers(prev => [np, ...prev]); setInput(''); setPdf(null); if (fRef.current) fRef.current.value = ''
      setSelected(np); setTab('detail')
    } catch (err) { setError(err.message) }
    finally { setLoading(false); setLoadingMsg('') }
  }

  async function bibImp(file) {
    if (!file) return; setLoading(true); setError(null); setImpMsg('Reading...')
    try {
      const text = await file.text(); const isBib = file.name.endsWith('.bib') || text.trim().startsWith('@')
      const entries = isBib ? parseBib(text) : parseCsv(text)
      if (!entries.length) { setError('No papers found.'); setLoading(false); return }
      const ex = new Set(papers.map(p => p.title?.toLowerCase()))
      const nw = entries.filter(e => e.title && !ex.has(e.title.toLowerCase()))
      if (!nw.length) { setError('All duplicates.'); setLoading(false); return }
      setPapers(prev => [...nw.map(e => ({ id: gid(), ...e, summary: null, addedAt: new Date().toISOString(), notes: '', readStatus: 'unread', figure: null, schematic: null })), ...prev])
      setImpMsg('Imported ' + nw.length + ' papers!'); setTimeout(() => setImpMsg(''), 6000)
    } catch (err) { setError('Import failed: ' + err.message) }
    finally { setLoading(false); if (bRef.current) bRef.current.value = '' }
  }

  async function zotImp() {
    if (!zUid || !zKey) { setError('Add Zotero credentials in ⚙'); return }
    setZotL(true); setZotMsg('Connecting...'); setError(null)
    try {
      let start = 0, all = []
      while (true) {
        const r = await fetch('https://api.zotero.org/users/' + zUid + '/items?format=json&itemType=-attachment%20||%20note&limit=50&start=' + start + '&sort=dateModified&direction=desc', { headers: { 'Zotero-API-Key': zKey, 'Zotero-API-Version': '3' } })
        if (!r.ok) throw new Error('Zotero error ' + r.status)
        const items = await r.json(); if (!items.length) break
        all = [...all, ...items]; setZotMsg('Found ' + all.length + '...')
        if (items.length < 50 || all.length > 500) break; start += 50
      }
      const types = new Set(['journalArticle', 'conferencePaper', 'preprint', 'book', 'bookSection', 'thesis', 'report'])
      const zi = all.filter(i => types.has(i.data?.itemType))
      const ex = new Set(papers.map(p => p.title?.toLowerCase())); const nw = []
      for (const item of zi) {
        const d = item.data; if (!d.title || ex.has(d.title.toLowerCase())) continue; ex.add(d.title.toLowerCase())
        nw.push({ id: gid(), title: d.title, authors: (d.creators || []).filter(c => c.creatorType === 'author').map(c => ((c.firstName || '') + ' ' + (c.lastName || '')).trim()), abstract: d.abstractNote || '', published: d.date || '', venue: d.publicationTitle || d.proceedingsTitle || d.bookTitle || '', source: 'Zotero', sourceId: d.DOI || d.key, url: d.url || '', summary: null, addedAt: new Date().toISOString(), notes: '', readStatus: 'unread', figure: null, schematic: null })
      }
      if (nw.length) setPapers(prev => [...nw, ...prev])
      setZotMsg('Imported ' + nw.length + ' papers.'); setTimeout(() => setZotMsg(''), 8000)
    } catch (err) { setError('Zotero failed: ' + err.message) }
    finally { setZotL(false) }
  }

  async function sumAll() {
    const un = papers.filter(p => !p.summary); if (!un.length) return
    if (!apiKey) { setError('API key required.'); return }
    setSumL(true); setError(null); let upd = [...papers]
    for (let i = 0; i < un.length; i++) {
      setLoadingMsg('Summarizing ' + (i + 1) + '/' + un.length + ': ' + un[i].title.slice(0, 35) + '...')
      try { const s = await genSummary(apiKey, un[i]); upd = upd.map(p => p.id === un[i].id ? { ...p, summary: s } : p) }
      catch { upd = upd.map(p => p.id === un[i].id ? { ...p, summary: { tldr: (p.abstract || '').slice(0, 100), tags: ['untagged'], key_contributions: [], methods: [], limitations: [], open_questions: [], key_citations_to_follow: [], relevance_to_medical_imaging: '' } } : p) }
    }
    setPapers(upd); setSumL(false); setLoadingMsg('')
  }

  async function runGap() {
    if (papers.length < 2 || !apiKey) { setError(papers.length < 2 ? 'Need 2+ papers' : 'API key required.'); return }
    setGapL(true); setError(null)
    try {
      const sm = papers.slice(0, 20).map((p, i) => {
        const s = p.summary
        const info = s ? ('TLDR: ' + (s.tldr || '') + '. Methods: ' + (s.methods || []).join(', ') + '. Limitations: ' + (s.limitations || []).join(', '))
          : ('Abstract: ' + (p.abstract || 'N/A').slice(0, 200))
        return '[' + (i + 1) + '] "' + p.title + '" - ' + info
      }).join('\n')
      const txt = await callAI(apiKey, 'You are a research gap analyst. Analyze these ' + papers.length + ' papers and identify research gaps.\n\n' + sm + '\n\nReturn ONLY valid JSON with no other text:\n{"themes":["theme1","theme2"],"gaps":["specific gap 1","specific gap 2"],"contradictions":["contradiction if any"],"suggested_directions":["direction 1","direction 2"],"missing_baselines":["missing comparison 1"],"suggested_search_queries":["arxiv query to fill gap 1"]}', 1200)
      const parsed = JSON.parse(txt)
      setGap(parsed); setTab('gaps')
    } catch (err) {
      const msg = err.message || 'Unknown error'
      setError('Gap analysis failed: ' + msg)
    }
    finally { setGapL(false) }
  }

  function del(id) { setPapers(p => p.filter(x => x.id !== id)); if (selected?.id === id) { setSelected(null); setTab('library') } }
  function togRead(id) { setPapers(p => p.map(x => x.id === id ? { ...x, readStatus: x.readStatus === 'read' ? 'unread' : 'read' } : x)) }
  function togStar(id) { setPapers(p => p.map(x => x.id === id ? { ...x, starred: !x.starred } : x)); if (selected?.id === id) setSelected(p => ({ ...p, starred: !p.starred })) }
  function updNotes(id, n) { setPapers(p => p.map(x => x.id === id ? { ...x, notes: n } : x)); if (selected?.id === id) setSelected(p => ({ ...p, notes: n })) }
  function updFigure(id, fig) {
    // Warn if figure is very large (base64 images can be huge)
    if (fig && fig.length > 2000000) {
      if (!confirm('This image is large and may exceed browser storage limits. Continue?')) return
    }
    setPapers(p => p.map(x => x.id === id ? { ...x, figure: fig } : x))
    if (selected?.id === id) setSelected(p => ({ ...p, figure: fig }))
  }
  function updSchematic(id, svg) { setPapers(p => p.map(x => x.id === id ? { ...x, schematic: svg } : x)); if (selected?.id === id) setSelected(p => ({ ...p, schematic: svg })) }
  function spk(p) { if (speaking) { speechSynthesis.cancel(); setSpeaking(false); return }; const u = new SpeechSynthesisUtterance(p.title + '. ' + (p.summary?.tldr || '')); u.rate = 0.95; u.onend = () => setSpeaking(false); setSpeaking(true); speechSynthesis.speak(u) }

  const [showStarred, setShowStarred] = useState(false)

  const allTags = [...new Set(papers.flatMap(p => p.summary?.tags || []))]
  const filtered = papers.filter(p => {
    const ms = !search || p.title?.toLowerCase().includes(search.toLowerCase()) || p.summary?.tldr?.toLowerCase().includes(search.toLowerCase()) || p.authors?.some(a => a.toLowerCase().includes(search.toLowerCase()))
    const tagMatch = !filterTag || p.summary?.tags?.includes(filterTag)
    const starMatch = !showStarred || p.starred
    return ms && tagMatch && starMatch
  })
  const unsum = papers.filter(p => !p.summary).length

  // ── Shared header + nav ────────────────────────────────────────────────
  const headerEl = (
    <>
      <header style={{ padding: '24px 32px 16px', borderBottom: '1px solid var(--border)', background: 'linear-gradient(180deg, var(--bg-tertiary) 0%, var(--bg-primary) 100%)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 style={{ fontSize: 26, fontWeight: 400, color: 'var(--text-primary)', letterSpacing: -0.5, margin: 0 }}>Research<span style={{ color: 'var(--accent-green)', fontStyle: 'italic' }}>Hub</span></h1>
          <div style={{ fontSize: 13, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginTop: 4 }}>{papers.length} papers</div>
        </div>
        <button onClick={() => setShowSet(true)} className="btn-sec" style={{ padding: '8px 14px', fontSize: 16 }}>⚙</button>
      </header>
      {!apiKey && <div style={{ padding: '10px 32px', background: '#1a1a15', borderBottom: '1px solid #3a3329', fontSize: 13, color: 'var(--accent-orange)', fontFamily: 'var(--mono)' }}>⚠ Add API key in ⚙ for AI features</div>}
      {(zotL || zotMsg) && <div style={{ padding: '10px 32px', background: '#0f1520', borderBottom: '1px solid #1a2a3a', fontSize: 13, color: 'var(--accent-blue)', fontFamily: 'var(--mono)' }}>{zotL ? '⟳ ' : '✓ '}{zotMsg}</div>}
      <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--border)' }}>
        {['library', 'graph', 'gaps'].map(t => <button key={t} onClick={() => setTab(t)} style={{ padding: '10px 24px', fontSize: 13, fontFamily: 'var(--mono)', color: (tab === t || (tab === 'detail' && t === 'library')) ? 'var(--accent-green-light)' : 'var(--text-faint)', background: (tab === t || (tab === 'detail' && t === 'library')) ? '#0f1a15' : 'transparent', cursor: 'pointer', border: 'none', borderBottom: (tab === t || (tab === 'detail' && t === 'library')) ? '2px solid var(--accent-green)' : '2px solid transparent', textTransform: 'capitalize' }}>{t === 'gaps' ? 'Gap Analysis' : t === 'graph' ? 'Relationships' : t}</button>)}
      </div>
      <div style={{ padding: '16px 32px', borderBottom: '1px solid var(--border)', display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
        <input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !loading) addPaper() }} placeholder="Paste arXiv URL, DOI, or paper title..." className="text-input" style={{ flex: 1, minWidth: 200 }} disabled={loading} />
        <input ref={fRef} type="file" accept=".pdf" onChange={e => setPdf(e.target.files?.[0] || null)} style={{ display: 'none' }} />
        <input ref={bRef} type="file" accept=".bib,.csv,.tsv,.txt" onChange={e => { if (e.target.files?.[0]) bibImp(e.target.files[0]) }} style={{ display: 'none' }} />
        <button onClick={() => fRef.current?.click()} className="btn-sec" disabled={loading}>{pdf ? '📄 ' + pdf.name.slice(0, 12) + '…' : 'PDF'}</button>
        <button onClick={() => bRef.current?.click()} className="btn-sec" style={{ color: 'var(--accent-blue)', borderColor: '#1e2a4a' }} disabled={loading}>BibTeX/CSV</button>
        <button onClick={addPaper} disabled={loading || (!input.trim() && !pdf)} className="btn-primary" style={{ opacity: loading || (!input.trim() && !pdf) ? 0.5 : 1 }}>{loading ? loadingMsg || '...' : '+ Add'}</button>
      </div>
      {impMsg && <div style={{ padding: '10px 32px', background: '#0f1a15', borderBottom: '1px solid #1a3329', fontSize: 13, color: 'var(--accent-green-light)', fontFamily: 'var(--mono)' }}>✓ {impMsg}</div>}
      {error && <div style={{ margin: '12px 32px 0', padding: '10px 14px', background: '#2a1515', border: '1px solid #4a2020', borderRadius: 6, fontSize: 13, color: '#d49090', fontFamily: 'var(--mono)', display: 'flex', justifyContent: 'space-between' }}>{error}<span onClick={() => setError(null)} style={{ cursor: 'pointer', color: '#8a5050' }}>✕</span></div>}
    </>
  )

  const settingsEl = <Settings show={showSet} onClose={() => setShowSet(false)} apiKey={apiKey} setApiKey={setApiKey} zUid={zUid} setZUid={setZUid} zKey={zKey} setZKey={setZKey} zotImp={zotImp} zotL={zotL} />

  // ── Route views ────────────────────────────────────────────────────────
  if (tab === 'detail' && selected) {
    return (<div>{headerEl}<DetailView paper={selected} apiKey={apiKey} onBack={() => { speechSynthesis.cancel(); setSpeaking(false); setTab('library') }} onDelete={del} onToggleRead={togRead} onToggleStar={togStar} onUpdateNotes={updNotes} onUpdateFigure={updFigure} onUpdateSchematic={updSchematic} speaking={speaking} onSpeak={spk} />{settingsEl}</div>)
  }
  if (tab === 'graph') {
    return (<div>{headerEl}<GraphView papers={papers} onSelect={id => { const p = papers.find(x => x.id === id); if (p) { setSelected(p); setTab('detail') } }} />{settingsEl}</div>)
  }
  if (tab === 'gaps') {
    return (<div>{headerEl}<GapView papers={papers} gap={gap} gapL={gapL} onRunGap={runGap} />{settingsEl}</div>)
  }

  return (<div>{headerEl}<LibraryView papers={papers} filtered={filtered} allTags={allTags} filterTag={filterTag} setFilterTag={setFilterTag} search={search} setSearch={setSearch} unsum={unsum} sumL={sumL} loadingMsg={loadingMsg} onSumAll={sumAll} onSelect={p => { setSelected(p); setTab('detail') }} onToggleStar={togStar} showStarred={showStarred} setShowStarred={setShowStarred} />{settingsEl}</div>)
}
