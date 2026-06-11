# 🧠 Workplace Intelligence Hub

An AI agent command center for the modern workplace. Four specialized agents — Research, Comms, Data Analyst, and Ops Automation — each powered by **Claude Sonnet 4** with **real-time web search**.

---

## Architecture

```
workplace-intel-hub/
├── client/          # React frontend (Create React App)
│   └── src/
│       ├── App.js          # Main UI and state management
│       ├── agents.js       # Agent definitions and quick prompts
│       ├── api.js          # Backend communication layer
│       └── components.js  # Reusable UI components
│
├── server/          # Express.js backend proxy
│   └── index.js    # API proxy — holds your Anthropic key securely
│
└── package.json     # Root scripts (runs both together)
```

**Why a backend proxy?**
Your `ANTHROPIC_API_KEY` never touches the browser. All Claude API calls go through the Express server, which adds the key server-side before forwarding to Anthropic.

---

## Quick Start

### 1. Install dependencies

```bash
# From the project root:
npm run install:all
```

This installs packages for the root, client, and server in one shot.

### 2. Add your API key

```bash
cp server/.env.example server/.env
```

Open `server/.env` and replace the placeholder:

```
ANTHROPIC_API_KEY=sk-ant-your-real-key-here
```

Get a key at: https://console.anthropic.com/

### 3. Run in development mode

```bash
npm run dev
```

This starts both servers concurrently:
- **React frontend** → http://localhost:3000
- **Express backend** → http://localhost:3001

Open http://localhost:3000 in your browser.

---

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run install:all` | Install all dependencies (run once) |
| `npm run dev` | Start both frontend + backend in dev mode |
| `npm run dev:client` | Start only the React frontend |
| `npm run dev:server` | Start only the Express backend |
| `npm run build` | Build the React app for production |
| `npm start` | Run the production Express server |

---

## Agents

| Agent | What it does |
|-------|-------------|
| ⚡ Research Agent | Surfaces live market intel, competitor moves, industry signals |
| ✉️ Comms Agent | Drafts emails, Slack messages, announcements with the right tone |
| 📊 Data Analyst Agent | Interprets metrics, benchmarks performance, spots anomalies |
| ⚙️ Ops Automation Agent | Maps workflows, finds bottlenecks, recommends automation tools |

Every agent has web search enabled — they find real, current information rather than relying on training data.

---

## Customising Agents

**Add a new agent:** Edit `client/src/agents.js` to add an entry to `AGENT_PERSONAS` and `QUICK_PROMPTS`, then add the corresponding system prompt to the `AGENT_SYSTEM_PROMPTS` object in `server/index.js`.

**Change the model:** In `server/index.js`, update the `model` field in the Anthropic API call.

**Adjust response length:** Change `max_tokens` in `server/index.js` (currently 1500).

---

## Tech Stack

- **Frontend:** React 18, react-markdown
- **Backend:** Node.js, Express, node-fetch
- **AI:** Anthropic Claude Sonnet 4 with web_search tool
- **Dev tooling:** concurrently, nodemon

---

## Production Deployment

1. `npm run build` — builds the React app to `client/build/`
2. Serve `client/build/` as static files from your Express server (add `express.static` middleware), or deploy frontend and backend separately.
3. Set `ANTHROPIC_API_KEY` as an environment variable on your host — never hardcode it.
