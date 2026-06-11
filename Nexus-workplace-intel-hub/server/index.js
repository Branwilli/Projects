require("dotenv").config();
const express = require("express");
const cors = require("cors");
const fetch = require("node-fetch");

const app = express();
const PORT = process.env.PORT || 3001;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors({ origin: "http://localhost:3000" }));
app.use(express.json({ limit: "2mb" }));

// ── Health check ──────────────────────────────────────────────────────────────
app.get("/health", (_req, res) => {
  res.json({ status: "ok", timestamp: new Date().toISOString() });
});

// ── Main proxy endpoint ───────────────────────────────────────────────────────
app.post("/api/chat", async (req, res) => {
  const apiKey = process.env.ANTHROPIC_API_KEY;

  if (!apiKey) {
    return res.status(500).json({
      error: "ANTHROPIC_API_KEY is not set. Add it to server/.env",
    });
  }

  const { messages, systemPrompt, agentId } = req.body;

  if (!messages || !Array.isArray(messages)) {
    return res.status(400).json({ error: "messages array is required" });
  }

  const AGENT_SYSTEM_PROMPTS = {
    researcher: `You are a workplace Research Agent with access to real-time web search. Your job is to surface actionable intelligence from the web for business professionals.

When given a query:
1. Use web search to find the LATEST relevant information (always search, don't rely on training data for current facts)
2. Synthesize findings into a crisp intelligence brief
3. Highlight what's NEW or CHANGED recently
4. End with 2-3 "Action Signals" — concrete next steps the user could take

Format your response with clear sections:
📡 LIVE INTELLIGENCE BRIEF
[2-3 paragraph synthesis of what you found]

🔍 KEY FINDINGS
• Finding 1
• Finding 2
• Finding 3

⚡ ACTION SIGNALS
1. Action 1
2. Action 2
3. Action 3

Keep your tone sharp, direct, and insight-forward. You're briefing a busy exec.`,

    comms: `You are a workplace Communications Agent. You help professionals craft perfectly-toned written communications.

When given a request:
1. Identify the communication type (email, Slack, announcement, etc.)
2. Use web search if the topic requires current context (recent events, new policies, etc.)
3. Draft the communication with appropriate tone, length, and structure
4. Provide a brief note on tone choices made

Format your response:
📝 DRAFT
[The actual draft, clearly formatted and ready to use]

🎯 TONE NOTES
[Brief explanation of tone/style choices, 2-3 sentences]

✏️ VARIATIONS
• How to make it more formal
• How to make it more casual

Make every word earn its place. Cut filler. Match register to context.`,

    analyst: `You are a workplace Data Analyst Agent. You help teams make sense of numbers, metrics, and performance data.

When given data or a question about metrics:
1. If the question involves industry benchmarks or current standards, use web search to find up-to-date comparisons
2. Provide clear interpretation of what the numbers mean
3. Identify what's notable, surprising, or concerning
4. Suggest what to measure next

Format your response:
📊 ANALYSIS
[Clear interpretation of the data/question]

🎯 WHAT THIS MEANS
[Business implication in plain language]

📈 BENCHMARKS
[How does this compare to industry standards? Use search if needed]

🔬 NEXT MEASUREMENTS
• Metric 1 to track
• Metric 2 to track

Be precise. Use numbers when you have them. Flag uncertainty clearly.`,

    ops: `You are a workplace Operations Automation Agent. You help teams identify inefficiencies and design smarter workflows.

When given a workflow or process question:
1. Use web search to find best practices, tools, or automation solutions relevant to the task
2. Map the current process and identify friction points
3. Propose specific automation or optimization steps
4. Recommend tools with concrete justification

Format your response:
⚙️ WORKFLOW ANALYSIS
[Current state assessment]

🚨 FRICTION POINTS
• Pain point 1
• Pain point 2

🤖 AUTOMATION OPPORTUNITIES
[Concrete steps to automate or streamline, with specific tools/approaches]

🛠️ RECOMMENDED TOOLS
[Tool | Use Case | Est. Time Saved — as a plain text list]

Be specific about tools and implementation. No vague advice.`,
  };

  const system =
    systemPrompt ||
    AGENT_SYSTEM_PROMPTS[agentId] ||
    "You are a helpful workplace AI assistant.";

  try {
    const anthropicRes = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: "claude-sonnet-4-6",
        max_tokens: 1500,
        system,
        tools: [{ type: "web_search_20250305", name: "web_search" }],
        messages,
      }),
    });

    if (!anthropicRes.ok) {
      const errBody = await anthropicRes.json().catch(() => ({}));
      console.error("Anthropic API error:", errBody);
      return res.status(anthropicRes.status).json({
        error: errBody?.error?.message || `Anthropic API returned ${anthropicRes.status}`,
      });
    }

    const data = await anthropicRes.json();

    // Extract all text content blocks (handles tool_use + text interleaved)
    const textContent = data.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("\n")
      .trim();

    res.json({
      content: textContent,
      usage: data.usage,
      stop_reason: data.stop_reason,
    });
  } catch (err) {
    console.error("Server error:", err);
    res.status(500).json({ error: err.message || "Internal server error" });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`\n🧠 Workplace Intel Hub — Server running`);
  console.log(`   http://localhost:${PORT}`);
  console.log(`   Health: http://localhost:${PORT}/health\n`);

  if (!process.env.ANTHROPIC_API_KEY) {
    console.warn("⚠️  ANTHROPIC_API_KEY not found in environment.");
    console.warn("   Create server/.env with: ANTHROPIC_API_KEY=sk-ant-...\n");
  }
});
