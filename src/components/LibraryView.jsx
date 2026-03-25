import { useState } from 'react'
import { tagStyle, getPaperUrl } from '../utils'

export default function LibraryView({ papers, filtered, allTags, filterTags, setFilterTags, readFilter, setReadFilter, search, setSearch, unsum, sumL, loadingMsg, onSumAll, onSelect, onToggleStar, showStarred, setShowStarred, weeklyRecs, weeklyL, onGenWeekly }) {
  const starredCount = papers.filter(p => p.starred).length
  const readCount = papers.filter(p => p.readStatus === 'read').length
  const unreadCount = papers.length - readCount
  const [showWeekly, setShowWeekly] = useState(!!weeklyRecs)

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
        <button onClick={() => { if (weeklyRecs) setShowWeekly(!showWeekly); else onGenWeekly() }} disabled={weeklyL} className="btn-sec" style={{ color: 'var(--accent-purple)', borderColor: '#2a2040', fontSize: 12 }}>{weeklyL ? 'Thinking...' : '📚 Weekly Reading'}</button>
        {unsum > 0 && <button onClick={onSumAll} disabled={sumL} className="btn-primary" style={{ background: '#1a2a33', color: 'var(--accent-blue)', borderColor: '#2a4a5a' }}>{sumL ? loadingMsg || 'Summarizing...' : '⚡ Summarize All (' + unsum + ')'}</button>}
      </div>

      {/* Weekly Reading Panel */}
      {showWeekly && weeklyRecs && (
        <div style={{ background: '#151020', border: '1px solid #2a2040', borderRadius: 10, padding: '18px 22px', marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h3 style={{ fontSize: 15, color: 'var(--accent-purple)', fontWeight: 400, margin: 0, fontFamily: 'var(--mono)' }}>📚 Weekly Reading Suggestions</h3>
            <div style={{ display: 'flex', gap: 8 }}>
              <button onClick={onGenWeekly} disabled={weeklyL} className="btn-sec" style={{ fontSize: 10, padding: '3px 8px', color: 'var(--accent-purple)' }}>{weeklyL ? '...' : '↻ Refresh'}</button>
              <span onClick={() => setShowWeekly(false)} style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: 14 }}>✕</span>
            </div>
          </div>

          {/* Theme */}
          {weeklyRecs.theme_of_the_week && (
            <div style={{ fontSize: 14, color: '#c4b0e0', lineHeight: 1.4, marginBottom: 14, padding: '8px 12px', background: '#1a1530', borderRadius: 6, borderLeft: '3px solid #9070c4' }}>
              {weeklyRecs.theme_of_the_week}
            </div>
          )}

          {/* From your library */}
          {weeklyRecs.from_library?.length > 0 && (
            <div style={{ marginBottom: 14 }}>
              <div style={{ fontSize: 11, color: 'var(--accent-green)', fontFamily: 'var(--mono)', marginBottom: 8, textTransform: 'uppercase' }}>From your library</div>
              {weeklyRecs.from_library.map((rec, i) => {
                const match = papers.find(p => p.title?.toLowerCase() === rec.title?.toLowerCase())
                return (
                  <div key={i} onClick={() => { if (match) onSelect(match) }}
                    style={{ padding: '8px 12px', marginBottom: 4, borderRadius: 6, cursor: match ? 'pointer' : 'default', borderLeft: '3px solid ' + (rec.priority === 'high' ? '#5b8a72' : '#3a5a4a'), transition: 'background 0.15s' }}
                    onMouseEnter={e => e.currentTarget.style.background = '#1a2030'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      {rec.priority === 'high' && <span style={{ fontSize: 9, color: '#5b8a72', fontFamily: 'var(--mono)', background: '#1a2e1a', padding: '1px 5px', borderRadius: 3 }}>HIGH</span>}
                      <span style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.3 }}>{rec.title}</span>
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--mono)', marginTop: 3 }}>{rec.reason}</div>
                  </div>
                )
              })}
            </div>
          )}

          {/* New papers to search for */}
          {weeklyRecs.new_suggestions?.length > 0 && (
            <div style={{ marginBottom: 10 }}>
              <div style={{ fontSize: 11, color: 'var(--accent-blue)', fontFamily: 'var(--mono)', marginBottom: 8, textTransform: 'uppercase' }}>New papers to explore</div>
              {weeklyRecs.new_suggestions.map((sug, i) => (
                <div key={i} style={{ padding: '8px 12px', marginBottom: 4, borderRadius: 6, borderLeft: '3px solid #1e3a5a' }}>
                  <div style={{ fontSize: 12, color: '#7eb8da', fontFamily: 'var(--mono)' }}>{sug.topic}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{sug.reason}</div>
                  <a href={'https://scholar.google.com/scholar?q=' + encodeURIComponent(sug.search_query)} target="_blank" rel="noopener"
                    style={{ fontSize: 10, color: 'var(--accent-blue)', fontFamily: 'var(--mono)', textDecoration: 'none', marginTop: 4, display: 'inline-block' }}>
                    Search: {sug.search_query} →
                  </a>
                </div>
              ))}
            </div>
          )}

          {/* Reading order tip */}
          {weeklyRecs.reading_order_tip && (
            <div style={{ fontSize: 11, color: '#6a5a7a', fontFamily: 'var(--mono)', padding: '6px 10px', background: '#120e1a', borderRadius: 4 }}>
              💡 {weeklyRecs.reading_order_tip}
            </div>
          )}

          {weeklyRecs.generatedAt && (
            <div style={{ fontSize: 10, color: '#3a3050', fontFamily: 'var(--mono)', marginTop: 8 }}>Generated {new Date(weeklyRecs.generatedAt).toLocaleDateString()}</div>
          )}
        </div>
      )}

      {/* Search + active filters */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search papers..." className="text-input" style={{ flex: 1, minWidth: 200 }} />
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

      {/* Tag cloud */}
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
                <div onClick={(e) => { e.stopPropagation(); onToggleStar(p.id) }}
                  style={{ padding: '18px 12px 18px 4px', cursor: 'pointer', fontSize: 16, color: p.starred ? '#c4c470' : '#2a3545', flexShrink: 0, transition: 'color 0.15s' }}
                  onMouseEnter={e => { if (!p.starred) e.currentTarget.style.color = '#6a6a30' }}
                  onMouseLeave={e => { if (!p.starred) e.currentTarget.style.color = '#2a3545' }}>
                  {p.starred ? '★' : '☆'}
                </div>
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
