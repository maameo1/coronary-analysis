import { tagStyle, getPaperUrl } from '../utils'

export default function LibraryView({ papers, filtered, allTags, filterTags, setFilterTags, readFilter, setReadFilter, search, setSearch, unsum, sumL, loadingMsg, onSumAll, onSelect, onToggleStar, showStarred, setShowStarred }) {
  const starredCount = papers.filter(p => p.starred).length
  const readCount = papers.filter(p => p.readStatus === 'read').length
  const unreadCount = papers.length - readCount

  function toggleTag(tag) {
    setFilterTags(prev => prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag])
  }

  const readBtnStyle = (active) => ({
    padding: '4px 10px', fontSize: 11, fontFamily: 'var(--mono)', cursor: 'pointer',
    background: active ? '#1a2535' : 'transparent',
    color: active ? 'var(--accent-green-light)' : 'var(--text-faint)',
    border: '1px solid ' + (active ? '#2a4a39' : '#1e2a3a'),
    borderRadius: 4,
  })

  return (
    <div style={{ padding: '20px 32px' }}>
      {/* Stats bar */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 20, flexWrap: 'wrap', alignItems: 'center' }}>
        <span className="stat">{papers.length} papers</span>
        <span className="stat" style={{ cursor: 'pointer', color: readFilter === 'read' ? 'var(--accent-green-light)' : 'var(--text-muted)' }}
          onClick={() => setReadFilter(readFilter === 'read' ? 'all' : 'read')}>{readCount} read</span>
        <span className="stat" style={{ cursor: 'pointer', color: readFilter === 'unread' ? 'var(--accent-orange)' : 'var(--text-muted)' }}
          onClick={() => setReadFilter(readFilter === 'unread' ? 'all' : 'unread')}>{unreadCount} unread</span>
        {starredCount > 0 && <span className="stat" onClick={() => setShowStarred(!showStarred)} style={{ cursor: 'pointer', color: showStarred ? '#c4c470' : 'var(--text-muted)', border: showStarred ? '1px solid #3a3a20' : undefined }}>★ {starredCount}</span>}
        <span className="stat">{allTags.length} tags</span>
        <div style={{ flex: 1 }} />
        {unsum > 0 && <button onClick={onSumAll} disabled={sumL} className="btn-primary" style={{ background: '#1a2a33', color: 'var(--accent-blue)', borderColor: '#2a4a5a' }}>{sumL ? loadingMsg || 'Summarizing...' : '⚡ Summarize All (' + unsum + ')'}</button>}
      </div>

      {/* Search + active filters */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search papers..." className="text-input" style={{ flex: 1, minWidth: 200 }} />

        {/* Read/Unread filter pills */}
        <div style={{ display: 'flex', gap: 4 }}>
          <span onClick={() => setReadFilter('all')} style={readBtnStyle(readFilter === 'all')}>All</span>
          <span onClick={() => setReadFilter('read')} style={readBtnStyle(readFilter === 'read')}>✓ Read</span>
          <span onClick={() => setReadFilter('unread')} style={readBtnStyle(readFilter === 'unread')}>○ Unread</span>
        </div>
      </div>

      {/* Active filter badges */}
      {(filterTags.length > 0 || showStarred || readFilter !== 'all') && (
        <div style={{ display: 'flex', gap: 6, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginRight: 4 }}>Active filters:</span>
          {filterTags.map(t => <span key={t} onClick={() => toggleTag(t)} style={{ ...tagStyle(t), cursor: 'pointer', paddingRight: 12, position: 'relative' }}>{t} <span style={{ position: 'absolute', right: 3, top: 0, fontSize: 9, opacity: 0.6 }}>✕</span></span>)}
          {showStarred && <span onClick={() => setShowStarred(false)} style={{ padding: '2px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'var(--mono)', background: '#2a2a1a', color: '#c4c470', cursor: 'pointer' }}>★ Starred ✕</span>}
          {readFilter !== 'all' && <span onClick={() => setReadFilter('all')} style={{ padding: '2px 8px', borderRadius: 3, fontSize: 11, fontFamily: 'var(--mono)', background: readFilter === 'read' ? '#1a2a1a' : '#2a1a15', color: readFilter === 'read' ? 'var(--accent-green)' : 'var(--accent-orange)', cursor: 'pointer' }}>{readFilter === 'read' ? '✓ Read' : '○ Unread'} ✕</span>}
          <span onClick={() => { setFilterTags([]); setShowStarred(false); setReadFilter('all') }} style={{ fontSize: 11, color: 'var(--accent-red)', fontFamily: 'var(--mono)', cursor: 'pointer', marginLeft: 4 }}>Clear all</span>
        </div>
      )}

      {/* Tag cloud - multi-select */}
      {allTags.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          {allTags.map(t => {
            const isActive = filterTags.includes(t)
            return <span key={t} onClick={() => toggleTag(t)}
              style={{ ...tagStyle(t), cursor: 'pointer', opacity: filterTags.length > 0 && !isActive ? 0.4 : 1, outline: isActive ? '1.5px solid ' + tagStyle(t).color : 'none', outlineOffset: 1 }}>{t}</span>
          })}
        </div>
      )}

      {/* Paper list */}
      {filtered.length === 0 ? (
        <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-faint)' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>◇</div>
          <div style={{ fontSize: 14, fontFamily: 'var(--mono)' }}>{papers.length === 0 ? 'Add papers, import BibTeX/CSV, or sync Zotero via ⚙' : 'No papers match your filters'}</div>
        </div>
      ) : (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-faint)', fontFamily: 'var(--mono)', marginBottom: 8 }}>Showing {filtered.length} of {papers.length}</div>
          {filtered.map(p => {
            const pUrl = getPaperUrl(p)
            return (
              <div key={p.id} className="paper-card" style={{ display: 'flex', gap: 0 }}>
                {/* Star button */}
                <div onClick={(e) => { e.stopPropagation(); onToggleStar(p.id) }}
                  style={{ padding: '18px 12px 18px 4px', cursor: 'pointer', fontSize: 16, color: p.starred ? '#c4c470' : '#2a3545', flexShrink: 0, transition: 'color 0.15s' }}
                  onMouseEnter={e => { if (!p.starred) e.currentTarget.style.color = '#6a6a30' }}
                  onMouseLeave={e => { if (!p.starred) e.currentTarget.style.color = '#2a3545' }}>
                  {p.starred ? '★' : '☆'}
                </div>

                {/* Paper content */}
                <div style={{ flex: 1, cursor: 'pointer' }} onClick={() => onSelect(p)}>
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
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
