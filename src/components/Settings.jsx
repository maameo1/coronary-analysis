import { useState, useEffect } from 'react'
import { sv, AK, ZUK, ZKK } from '../utils'

export default function Settings({ show, onClose, apiKey, setApiKey, zUid, setZUid, zKey, setZKey, zotImp, zotL }) {
  const [tAK, setTAK] = useState(apiKey)
  const [tZU, setTZU] = useState(zUid)
  const [tZK, setTZK] = useState(zKey)

  // Sync temp values when modal opens or props change
  useEffect(() => {
    if (show) { setTAK(apiKey); setTZU(zUid); setTZK(zKey) }
  }, [show, apiKey, zUid, zKey])

  if (!show) return null
  const isty = { width: '100%', padding: '10px 14px', background: 'var(--bg-primary)', border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text-secondary)', fontSize: 14, fontFamily: 'var(--mono)', outline: 'none', boxSizing: 'border-box', marginBottom: 12 }

  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 100, display: 'flex', alignItems: 'center', justifyContent: 'center' }} onClick={e => { if (e.target === e.currentTarget) onClose() }}>
      <div style={{ background: 'var(--bg-secondary)', border: '1px solid var(--border)', borderRadius: 12, padding: 32, width: 480, maxWidth: '90vw', maxHeight: '90vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 24 }}>
          <h3 style={{ fontSize: 18, color: 'var(--text-primary)', fontWeight: 400, margin: 0 }}>Settings</h3>
          <span onClick={onClose} style={{ cursor: 'pointer', color: 'var(--text-muted)', fontSize: 18 }}>✕</span>
        </div>

        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 13, color: 'var(--accent-green-light)', fontFamily: 'var(--mono)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Anthropic API Key</div>
          <div style={{ fontSize: 12, color: 'var(--text-faint)', marginBottom: 8, lineHeight: 1.5 }}>Get at <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener" style={{ color: 'var(--accent-blue)' }}>console.anthropic.com</a></div>
          <input value={tAK} onChange={e => setTAK(e.target.value)} placeholder="sk-ant-..." type="password" style={isty} />
          <button onClick={() => { const clean = tAK.replace(/[^\x20-\x7E]/g, '').trim(); setApiKey(clean); sv(AK, clean); setTAK(clean) }} className="btn-primary" style={{ width: '100%', textAlign: 'center' }}>{apiKey ? '✓ Update' : 'Save Key'}</button>
        </div>

        <div style={{ marginBottom: 24, borderTop: '1px solid var(--border)', paddingTop: 20 }}>
          <div style={{ fontSize: 13, color: 'var(--accent-blue)', fontFamily: 'var(--mono)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: 1 }}>Zotero</div>
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

        <div style={{ borderTop: '1px solid var(--border)', paddingTop: 16 }}>
          <button onClick={() => { if (confirm('Clear all papers and data?')) { localStorage.removeItem('rh_papers'); localStorage.removeItem('rh_gap'); window.location.reload() } }} className="btn-sec" style={{ width: '100%', textAlign: 'center', color: 'var(--accent-red)', borderColor: '#3a1a1a' }}>Reset Library</button>
        </div>
      </div>
    </div>
  )
}
