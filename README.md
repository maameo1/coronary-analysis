# ResearchHub ◇

**AI-powered literature review and research intelligence tool.**

Import papers from BibTeX, CSV, arXiv, DOI, or PDF — get AI-generated summaries, gap analysis, citation recommendations, and audio overviews.

![ResearchHub](https://img.shields.io/badge/Built%20with-React%20%2B%20Vite-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Paper import** — Paste arXiv URLs, DOIs, paper titles, upload PDFs, or bulk-import via BibTeX/CSV from Zotero
- **AI summaries** — Auto-generates TLDR, key contributions, methods, limitations, tags, and real citation recommendations for each paper
- **Gap analysis** — Identifies themes, gaps, contradictions, and suggested research directions across your library
- **Search & filter** — Full-text search with tag-based filtering
- **Audio** — Listen to paper summaries via browser text-to-speech
- **Notes** — Add personal notes to any paper
- **Persistent** — Everything saved in localStorage

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/research-hub.git
cd research-hub
npm install
npm run dev
```

Open http://localhost:3000 and add your Anthropic API key in ⚙ Settings.

## Getting an API Key

1. Go to [console.anthropic.com](https://console.anthropic.com/settings/keys)
2. Create a new API key
3. Paste it in the ⚙ Settings panel in ResearchHub

Your key is stored in your browser's localStorage and only sent to Anthropic's API for generating summaries. It never touches any other server.

## Importing from Zotero

1. Open your Zotero library (desktop or web)
2. Select papers → File → Export → Choose **BibTeX** or **CSV**
3. Click **"Import BibTeX/CSV"** in ResearchHub
4. Click **"⚡ Summarize All"** to generate AI summaries

## Deploy on Vercel

```bash
npm run build
# Upload the `dist` folder to Vercel, Netlify, or any static host
```

Or connect your GitHub repo to [Vercel](https://vercel.com) for automatic deploys.

## Cost

AI features use the Claude API (claude-sonnet-4-20250514). Approximate costs:
- ~$0.003 per paper summary
- ~$0.002 per gap analysis
- A 100-paper library costs roughly $0.30 to summarize

## Tech Stack

- React 18 + Vite
- Anthropic Claude API
- arXiv API, Crossref API, Semantic Scholar API
- Browser SpeechSynthesis for TTS
- localStorage for persistence

## License

MIT
