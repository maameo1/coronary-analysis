import { tagStyle, getPaperUrl } from '../utils'

export default function LibraryView({ papers, filtered, allTags, filterTag, setFilterTag, search, setSearch, unsum, sumL, loadingMsg, onSumAll, onSelect }) {
  return (
    <div style={{ padding: '20px 32px' }}>
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        <span className="stat">{papers.length} papers</span>
        <span className="stat">{papers.filter(p => p.readStatus === 'read').length} read</span>
        <span className="stat">{allTags.length} tags</span>
        <div style={{ flex: 1 }} />
        {unsum > 0 && <button onClick={onSumAll} disabled={sumL} className="btn-primary" style={{ background: '#1a2a33', color: 'var(--accent-blue)', borderColor: '#2a4a5a' }}>{sumL ? loadingMsg || 'Summarizing...' : '⚡ Summarize All (' + unsum + ')'}</button>}
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap' }}>
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search papers..." className="text-input" style={{ flex: 1, minWidth: 200 }} />
        {filterTag && <button onClick={() => setFilterTag(null)} className="btn-sec" style={{ fontSize: 12 }}>✕ {filterTag}</button>}
      </div>

      {allTags.length > 0 && <div style={{ marginBottom: 16 }}>{allTags.map(t => <span key={t} onClick={() => setFilterTag(filterTag === t ? null : t)} style={{ ...tagStyle(t), cursor: 'pointer', opacity: filterTag && filterTag !== t ? 0.4 : 1 }}>{t}</span>)}</div>}

      {filtered.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-faint)' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>◇</div>
          <div style={{ fontSize: 14, fontFamily: 'var(--mono)' }}>{papers.length === 0 ? 'Add papers, import BibTeX/CSV, or sync Zotero via ⚙' : 'No matches'}</div>
        </div>
      ) : filtered.map(p => {
        const pUrl = getPaperUrl(p)
        return (
          <div key={p.id} className="paper-card" onClick={() => onSelect(p)}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  {p.readStatus === 'read' && <span style={{ color: 'var(--accent-green)', fontSize: 13 }}>✓</span>}
                  {!p.summary && <span style={{ color: 'var(--accent-orange)', fontSize: 13 }}>○</span>}
                  <span style={{ fontSize: 15, color: '#d4dae0', lineHeight: 1.4 }}>{p.title}</span>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginBottom: 6 }}>{(p.authors || []).slice(0, 3).join(', ')}{(p.authors || []).length > 3 ? ' et al.' : ''} · {p.published}</div>
                {p.summary?.tldr && <div style={{ fontSize: 13, color: '#7a8a9a', lineHeight: 1.4, marginBottom: 6 }}>{p.summary.tldr}</div>}
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
                  {(p.summary?.tags || []).map(t => <span key={t} style={tagStyle(t)}>{t}</span>)}
                  {p.figure && <span style={{ fontSize: 10, color: 'var(--accent-blue)', fontFamily: 'var(--mono)' }}>📷</span>}
                  {p.schematic && <span style={{ fontSize: 10, color: 'var(--accent-purple)', fontFamily: 'var(--mono)' }}>🎨</span>}
                  {pUrl && <a href={pUrl} target="_blank" rel="noopener" onClick={e => e.stopPropagation()} style={{ fontSize: 11, color: 'var(--accent-blue)', fontFamily: 'var(--mono)', textDecoration: 'none', marginLeft: 4 }}>🔗</a>}
                </div>
              </div>
              {p.figure && <img src={p.figure} alt="" style={{ width: 60, height: 45, objectFit: 'cover', borderRadius: 4, border: '1px solid var(--border)', flexShrink: 0 }} />}
              <div style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', whiteSpace: 'nowrap' }}>{p.source}</div>
            </div>
          </div>
        )
      })}
    </div>
  )
}
