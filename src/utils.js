// Storage keys
export const PK = 'rh_papers', AK = 'rh_api', ZUK = 'rh_zu', ZKK = 'rh_zk', GK = 'rh_gap'

export function gid() { return Date.now().toString(36) + Math.random().toString(36).slice(2, 8) }

// Tag colors
const TC = [{bg:'#1a2332',fg:'#7eb8da'},{bg:'#2a1a2e',fg:'#c490d1'},{bg:'#1a2e1a',fg:'#7ec47e'},{bg:'#2e2a1a',fg:'#d1c490'},{bg:'#1a2e2e',fg:'#7ec4c4'},{bg:'#2e1a1a',fg:'#d19090'}]
export function gtc(t){let h=0;for(let i=0;i<t.length;i++)h=((h<<5)-h+t.charCodeAt(i))|0;return TC[Math.abs(h)%TC.length]}
export function tagStyle(t){const c=gtc(t);return{display:'inline-block',padding:'2px 8px',borderRadius:3,fontSize:11,fontFamily:'var(--mono)',background:c.bg,color:c.fg,marginRight:6,marginBottom:4}}

// Storage
export function ld(k,d){try{const v=localStorage.getItem(k);return v?JSON.parse(v):d}catch{return d}}
export function sv(k,v){try{localStorage.setItem(k,typeof v==='string'?v:JSON.stringify(v))}catch{}}

// Paper URL helpers
export function getPaperUrl(p) {
  if (p.sourceId?.match(/^10\./)) return 'https://doi.org/' + p.sourceId
  if (p.sourceId?.match(/^\d{4}\.\d{4,5}/)) return 'https://arxiv.org/abs/' + p.sourceId
  if (p.url) return p.url
  return null
}
export function getPdfUrl(p) {
  if (p.sourceId?.match(/^\d{4}\.\d{4,5}/)) return 'https://arxiv.org/pdf/' + p.sourceId + '.pdf'
  if (p.pdfLink) return p.pdfLink
  return null
}

// Claude API
export async function callAI(key, prompt, mt) {
  if (!key) throw new Error('API key required. Add it in Settings.')
  // Sanitize key - remove any non-ASCII characters from copy-paste
  const cleanKey = key.replace(/[^\x20-\x7E]/g, '').trim()
  if (!cleanKey) throw new Error('API key appears invalid. Re-enter it in Settings.')
  const r = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': cleanKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
    },
    body: JSON.stringify({ model: 'claude-sonnet-4-20250514', max_tokens: mt || 1500, messages: [{ role: 'user', content: prompt }] })
  })
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error?.message || 'API error ' + r.status) }
  const d = await r.json()
  return (d.content || []).map(b => b.text || '').join('').replace(/```json/g, '').replace(/```/g, '').trim()
}

export async function genSummary(key, p) {
  return JSON.parse(await callAI(key, `Research paper analyst. Return ONLY JSON.
Title: ${p.title}\nAuthors: ${(p.authors||[]).join(', ')}\nAbstract: ${p.abstract||'N/A'}
For key_citations_to_follow provide REAL papers as "Author et al., Title (Year)".
{"tldr":"max 30 words","key_contributions":["..."],"methods":["..."],"limitations":["..."],"tags":["t1","t2","t3"],"relevance_to_medical_imaging":"One sentence.","key_citations_to_follow":[{"citation":"Author et al., Title (Year)","reason":"Why"}],"open_questions":["..."]}`, 1500))
}

// Generate an SVG schematic of the paper's method
export async function genSchematic(key, p) {
  const svgText = await callAI(key, `You are a scientific diagram creator. Given this paper, create a SIMPLE SVG diagram showing the key method/architecture/pipeline.

Title: ${p.title}
Abstract: ${p.abstract || 'N/A'}
Methods: ${p.summary?.methods?.join(', ') || 'N/A'}
Key contributions: ${p.summary?.key_contributions?.join(', ') || 'N/A'}

Rules:
- Return ONLY the SVG code, nothing else. No markdown, no backticks.
- Use viewBox="0 0 600 300"
- Use dark theme: background #111822, text #d4d4d4, boxes #1a2535 with border #2a4a39
- Use colors: green #5b8a72, blue #7eb8da, purple #9070c4, orange #c49070 for different components
- Include labeled boxes connected by arrows showing the data/method flow
- Keep it simple: 3-7 boxes maximum
- Use rounded rectangles (rx=6) for boxes
- Use readable font: font-family="monospace" font-size="11"
- Start with <svg and end with </svg>`, 2000)
  return svgText.trim()
}

// BibTeX parser
export function parseBib(text) {
  const entries = []
  text.split(/\n@/).map((b, i) => i === 0 ? b : '@' + b).forEach(block => {
    if (!block.match(/@\w+\{/)) return
    const g = (k) => { const m = block.match(new RegExp(k + '\\s*=\\s*[{"]([\\s\\S]*?)[}"]', 'i')); return m ? m[1].replace(/[{}]/g, '').replace(/\s+/g, ' ').trim() : '' }
    const t = g('title'); if (!t) return
    const a = g('author')
    entries.push({ title: t, authors: a ? a.split(' and ').map(x => { const p = x.trim().split(','); return p.length > 1 ? p[1].trim() + ' ' + p[0].trim() : x.trim() }) : [], abstract: g('abstract'), published: g('year') || g('date') || '', venue: g('journal') || g('booktitle') || '', source: 'BibTeX', sourceId: g('doi') || t.slice(0, 30) })
  })
  return entries
}

// CSV parser
export function parseCsv(text) {
  const l = text.split('\n'); if (l.length < 2) return []
  const h = l[0].split(',').map(x => x.trim().replace(/^"|"$/g, '').toLowerCase())
  const ti = h.findIndex(x => x.includes('title')); if (ti === -1) return []
  const ai = h.findIndex(x => ['author', 'authors', 'creator'].includes(x))
  const ab = h.findIndex(x => x.includes('abstract'))
  const yi = h.findIndex(x => ['year', 'date'].includes(x) || x.includes('publication'))
  const vi = h.findIndex(x => x === 'journal' || x.includes('publication title'))
  const di = h.findIndex(x => x === 'doi')
  const entries = []
  for (let i = 1; i < l.length; i++) {
    if (!l[i].trim()) continue
    const c = []; let cur = '', inQ = false
    for (const ch of l[i]) { if (ch === '"') inQ = !inQ; else if (ch === ',' && !inQ) { c.push(cur.trim()); cur = '' } else cur += ch }
    c.push(cur.trim())
    const t = c[ti]?.replace(/^"|"$/g, ''); if (!t) continue
    const ar = ai >= 0 ? c[ai]?.replace(/^"|"$/g, '') || '' : ''
    entries.push({ title: t, authors: ar ? ar.split(';').map(a => a.trim()).filter(Boolean) : [], abstract: ab >= 0 ? c[ab]?.replace(/^"|"$/g, '') || '' : '', published: yi >= 0 ? c[yi]?.replace(/^"|"$/g, '') || '' : '', venue: vi >= 0 ? c[vi]?.replace(/^"|"$/g, '') || '' : '', source: 'CSV', sourceId: di >= 0 ? c[di]?.replace(/^"|"$/g, '') || t.slice(0, 30) : t.slice(0, 30) })
  }
  return entries
}

// Compute paper relationships
export function computeRelationships(papers) {
  const rels = []
  const stopwords = new Set(['a','an','the','of','for','and','in','on','with','to','from','by','using','based','deep','learning','network','method','model','image','segmentation','analysis','approach'])
  for (let i = 0; i < papers.length; i++) {
    for (let j = i + 1; j < papers.length; j++) {
      const a = papers[i], b = papers[j], reasons = []
      const aTags = new Set(a.summary?.tags || [])
      const shared = (b.summary?.tags || []).filter(t => aTags.has(t))
      if (shared.length > 0) reasons.push('Tags: ' + shared.join(', '))
      const aM = new Set((a.summary?.methods || []).map(m => m.toLowerCase()))
      const shM = (b.summary?.methods || []).filter(m => aM.has(m.toLowerCase()))
      if (shM.length > 0) reasons.push('Methods: ' + shM.join(', '))
      const aA = new Set((a.authors || []).map(x => x.toLowerCase()))
      const shA = (b.authors || []).filter(x => aA.has(x.toLowerCase()))
      if (shA.length > 0) reasons.push('Authors: ' + shA.join(', '))
      const aW = new Set((a.title || '').toLowerCase().split(/\W+/).filter(w => w.length > 3 && !stopwords.has(w)))
      const shW = (b.title || '').toLowerCase().split(/\W+/).filter(w => w.length > 3 && !stopwords.has(w) && aW.has(w))
      if (shW.length >= 2) reasons.push('Keywords: ' + shW.slice(0, 4).join(', '))
      if (reasons.length > 0) rels.push({ a: a.id, b: b.id, aTitle: a.title, bTitle: b.title, reasons, strength: reasons.length })
    }
  }
  return rels.sort((a, b) => b.strength - a.strength)
}
