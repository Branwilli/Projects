import { useState, useRef, useEffect, useCallback } from "react";
import { AGENT_PERSONAS, QUICK_PROMPTS } from "./agents";
import { sendAgentMessage, checkServerHealth } from "./api";
import {
  TypingIndicator,
  MessageBubble,
  AgentCard,
  QuickPromptButton,
  ErrorBanner,
  StatusBadge,
} from "./components";

const GLOBAL_STYLES = `
  @keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); }
    30% { transform: translateY(-6px); }
  }
  @keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes pulseGlow {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
  }
  * { box-sizing: border-box; }
  textarea:focus { outline: none; }
  textarea::placeholder { color: #94a3b8; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
`;

export default function App() {
  const [selectedAgent, setSelectedAgent] = useState(AGENT_PERSONAS[0]);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [serverOnline, setServerOnline] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // ── Server health check on mount ──────────────────────────────────────────
  useEffect(() => {
    checkServerHealth().then(setServerOnline);
  }, []);

  // ── Auto-scroll ───────────────────────────────────────────────────────────
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Agent switch — clear conversation ─────────────────────────────────────
  const handleAgentSwitch = useCallback((agent) => {
    setSelectedAgent(agent);
    setMessages([]);
    setError(null);
  }, []);

  // ── Send message ──────────────────────────────────────────────────────────
  const sendMessage = useCallback(
    async (text) => {
      const userText = (text || input).trim();
      if (!userText || loading) return;
      setInput("");
      setError(null);

      const newMessages = [...messages, { role: "user", content: userText }];
      setMessages(newMessages);
      setLoading(true);

      try {
        const result = await sendAgentMessage({
          agentId: selectedAgent.id,
          messages: newMessages,
        });
        setMessages([
          ...newMessages,
          { role: "assistant", content: result.content },
        ]);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, messages, selectedAgent.id]
  );

  // ── Keyboard shortcut: Enter to send, Shift+Enter for newline ─────────────
  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const agent = selectedAgent;
  const quickPrompts = QUICK_PROMPTS[agent.id] || [];

  return (
    <>
      <style>{GLOBAL_STYLES}</style>
      <div
        style={{
          height: "100vh",
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%)",
          fontFamily: "'Inter', -apple-system, sans-serif",
          overflow: "hidden",
        }}
      >
        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header
          style={{
            padding: "16px 28px",
            borderBottom: "1px solid rgba(255,255,255,0.07)",
            display: "flex",
            alignItems: "center",
            gap: 14,
            backdropFilter: "blur(12px)",
            flexShrink: 0,
          }}
        >
          <div
            style={{
              width: 38,
              height: 38,
              borderRadius: 10,
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 18,
              boxShadow: "0 0 20px #6366f140",
              flexShrink: 0,
            }}
          >
            🧠
          </div>
          <div>
            <div
              style={{ color: "white", fontWeight: 800, fontSize: 16, letterSpacing: "-0.02em" }}
            >
              Workplace Intelligence Hub
            </div>
            <div style={{ color: "#64748b", fontSize: 11, marginTop: 1 }}>
              4 specialized AI agents · Real-time web search · Powered by Claude
            </div>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
            {serverOnline !== null && <StatusBadge online={serverOnline} />}
            {["Claude Sonnet 4", "Web Search"].map((tag) => (
              <span
                key={tag}
                style={{
                  background: "rgba(99,102,241,0.15)",
                  border: "1px solid rgba(99,102,241,0.3)",
                  color: "#a5b4fc",
                  padding: "4px 10px",
                  borderRadius: 20,
                  fontSize: 11,
                  fontWeight: 600,
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        </header>

        {/* ── Body ────────────────────────────────────────────────────────── */}
        <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
          {/* Sidebar */}
          <aside
            style={{
              width: 252,
              padding: 16,
              borderRight: "1px solid rgba(255,255,255,0.07)",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              overflowY: "auto",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                color: "#475569",
                fontSize: 10,
                fontWeight: 800,
                letterSpacing: "0.1em",
                padding: "4px 2px 8px",
                textTransform: "uppercase",
              }}
            >
              Choose Agent
            </div>
            {AGENT_PERSONAS.map((a) => (
              <AgentCard
                key={a.id}
                agent={a}
                selected={a.id === agent.id}
                onClick={() => handleAgentSwitch(a)}
              />
            ))}

            {/* Session stats */}
            <div
              style={{
                marginTop: 6,
                background: "rgba(255,255,255,0.03)",
                borderRadius: 12,
                padding: 14,
                border: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <div
                style={{
                  color: "#334155",
                  fontSize: 10,
                  fontWeight: 800,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  marginBottom: 10,
                }}
              >
                Session
              </div>
              {[
                ["Messages", messages.length],
                ["Active Agent", `${agent.icon} ${agent.name.split(" ")[0]}`],
              ].map(([label, val]) => (
                <div
                  key={label}
                  style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}
                >
                  <span style={{ color: "#475569", fontSize: 12 }}>{label}</span>
                  <span style={{ color: "white", fontSize: 12, fontWeight: 700 }}>{val}</span>
                </div>
              ))}
            </div>
          </aside>

          {/* Chat panel */}
          <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            {/* Agent sub-header */}
            <div
              style={{
                padding: "14px 24px",
                background: "rgba(255,255,255,0.02)",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
                display: "flex",
                alignItems: "center",
                gap: 12,
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 12,
                  background: agent.bgLight,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: 20,
                  border: `2px solid ${agent.color}40`,
                  flexShrink: 0,
                }}
              >
                {agent.icon}
              </div>
              <div>
                <div style={{ color: "white", fontWeight: 700, fontSize: 14 }}>{agent.name}</div>
                <div style={{ color: "#475569", fontSize: 12 }}>{agent.description}</div>
              </div>
              <span
                style={{
                  marginLeft: "auto",
                  background: `${agent.color}18`,
                  border: `1px solid ${agent.color}40`,
                  color: agent.color,
                  padding: "4px 12px",
                  borderRadius: 20,
                  fontSize: 11,
                  fontWeight: 600,
                  flexShrink: 0,
                }}
              >
                Live Search ON
              </span>
            </div>

            {/* Messages */}
            <div style={{ flex: 1, overflowY: "auto", padding: "24px 28px" }}>
              {messages.length === 0 && (
                <div style={{ textAlign: "center", paddingTop: 36 }}>
                  <div style={{ fontSize: 44, marginBottom: 12 }}>{agent.icon}</div>
                  <div
                    style={{ color: "white", fontSize: 19, fontWeight: 800, marginBottom: 8 }}
                  >
                    {agent.name} is ready
                  </div>
                  <div
                    style={{
                      color: "#475569",
                      fontSize: 14,
                      maxWidth: 400,
                      margin: "0 auto 28px",
                      lineHeight: 1.6,
                    }}
                  >
                    {agent.description}. Ask anything — or start with a quick prompt.
                  </div>
                  <div
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      gap: 8,
                      maxWidth: 520,
                      margin: "0 auto",
                    }}
                  >
                    <div
                      style={{
                        color: "#334155",
                        fontSize: 10,
                        fontWeight: 800,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                        marginBottom: 4,
                      }}
                    >
                      Quick Prompts
                    </div>
                    {quickPrompts.map((p, i) => (
                      <QuickPromptButton
                        key={i}
                        text={p}
                        agentColor={agent.color}
                        onClick={() => sendMessage(p)}
                      />
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <MessageBubble key={i} msg={msg} agentColor={agent.color} />
              ))}

              {loading && (
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    animation: "fadeSlideIn 0.2s ease-out",
                  }}
                >
                  <div
                    style={{
                      background: "#f8fafc",
                      border: "1px solid #e2e8f0",
                      borderRadius: "18px 18px 18px 4px",
                    }}
                  >
                    <TypingIndicator color={agent.color} />
                  </div>
                  <span style={{ color: "#334155", fontSize: 12 }}>
                    Searching the web & analysing…
                  </span>
                </div>
              )}

              {error && <ErrorBanner message={error} />}
              <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div
              style={{
                padding: "14px 24px",
                borderTop: "1px solid rgba(255,255,255,0.07)",
                background: "rgba(0,0,0,0.25)",
                flexShrink: 0,
              }}
            >
              <div
                style={{
                  display: "flex",
                  gap: 10,
                  alignItems: "flex-end",
                  background: "rgba(255,255,255,0.06)",
                  border: "1.5px solid rgba(255,255,255,0.1)",
                  borderRadius: 16,
                  padding: "10px 14px",
                  transition: "border-color 0.15s",
                }}
                onFocus={() => {}}
              >
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder={`Ask the ${agent.name}…  (Enter to send, Shift+Enter for newline)`}
                  disabled={loading}
                  rows={1}
                  style={{
                    flex: 1,
                    background: "transparent",
                    border: "none",
                    color: "white",
                    fontSize: 14,
                    resize: "none",
                    lineHeight: 1.6,
                    maxHeight: 120,
                    overflowY: "auto",
                    fontFamily: "inherit",
                  }}
                />
                <button
                  onClick={() => sendMessage()}
                  disabled={loading || !input.trim()}
                  style={{
                    width: 36,
                    height: 36,
                    borderRadius: 10,
                    background:
                      loading || !input.trim() ? "rgba(255,255,255,0.08)" : agent.color,
                    border: "none",
                    cursor: loading || !input.trim() ? "not-allowed" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    transition: "all 0.15s",
                    fontSize: 17,
                    flexShrink: 0,
                    boxShadow:
                      loading || !input.trim() ? "none" : `0 2px 12px ${agent.color}55`,
                  }}
                  title="Send (Enter)"
                >
                  {loading ? "⏳" : "↑"}
                </button>
              </div>
              <p
                style={{
                  color: "#1e293b",
                  fontSize: 11,
                  textAlign: "center",
                  marginTop: 8,
                }}
              >
                Agents use live web search · Claude Sonnet 4 · Your API key stays on your server
              </p>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}
