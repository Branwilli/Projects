import React from "react";
import ReactMarkdown from "react-markdown";

// ── Typing indicator (animated dots) ─────────────────────────────────────────
export function TypingIndicator({ color }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "12px 16px" }}>
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          style={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: color,
            animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
    </div>
  );
}

// ── Single chat bubble ────────────────────────────────────────────────────────
export function MessageBubble({ msg, agentColor }) {
  const isUser = msg.role === "user";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        marginBottom: 16,
        animation: "fadeSlideIn 0.25s ease-out",
      }}
    >
      {!isUser && (
        <div
          style={{
            width: 8,
            borderLeft: `3px solid ${agentColor}`,
            borderRadius: "2px 0 0 2px",
            marginRight: 10,
            flexShrink: 0,
            alignSelf: "stretch",
          }}
        />
      )}
      <div
        style={{
          maxWidth: isUser ? "72%" : "82%",
          padding: "12px 16px",
          borderRadius: isUser ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
          background: isUser ? agentColor : "#f8fafc",
          color: isUser ? "white" : "#1e293b",
          border: isUser ? "none" : "1px solid #e2e8f0",
          fontSize: 14,
          lineHeight: 1.75,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
          wordBreak: "break-word",
        }}
      >
        {isUser ? (
          msg.content
        ) : (
          <div className="agent-response">
            <ReactMarkdown
              components={{
                p: ({ children }) => (
                  <p style={{ margin: "0 0 10px", lineHeight: 1.75 }}>{children}</p>
                ),
                ul: ({ children }) => (
                  <ul style={{ paddingLeft: 20, margin: "6px 0 10px" }}>{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol style={{ paddingLeft: 20, margin: "6px 0 10px" }}>{children}</ol>
                ),
                li: ({ children }) => (
                  <li style={{ marginBottom: 4, lineHeight: 1.6 }}>{children}</li>
                ),
                strong: ({ children }) => (
                  <strong style={{ color: agentColor, fontWeight: 700 }}>{children}</strong>
                ),
                table: ({ children }) => (
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, margin: "8px 0" }}>
                    {children}
                  </table>
                ),
                th: ({ children }) => (
                  <th style={{ background: "#f1f5f9", padding: "6px 10px", border: "1px solid #e2e8f0", textAlign: "left", fontWeight: 600 }}>
                    {children}
                  </th>
                ),
                td: ({ children }) => (
                  <td style={{ padding: "6px 10px", border: "1px solid #e2e8f0" }}>{children}</td>
                ),
              }}
            >
              {msg.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Agent selector card ───────────────────────────────────────────────────────
export function AgentCard({ agent, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      style={{
        background: selected ? agent.color : "white",
        border: `2px solid ${selected ? agent.color : "#e2e8f0"}`,
        borderRadius: 14,
        padding: "14px 16px",
        cursor: "pointer",
        textAlign: "left",
        transition: "all 0.18s ease",
        transform: selected ? "scale(1.02)" : "scale(1)",
        boxShadow: selected
          ? `0 4px 20px ${agent.color}40`
          : "0 1px 3px rgba(0,0,0,0.05)",
        width: "100%",
      }}
    >
      <div style={{ fontSize: 22, marginBottom: 4 }}>{agent.icon}</div>
      <div
        style={{
          fontWeight: 700,
          fontSize: 13,
          color: selected ? "white" : "#1e293b",
          marginBottom: 4,
          letterSpacing: "0.01em",
        }}
      >
        {agent.name}
      </div>
      <div
        style={{
          fontSize: 11,
          color: selected ? "rgba(255,255,255,0.8)" : "#64748b",
          lineHeight: 1.4,
        }}
      >
        {agent.description}
      </div>
    </button>
  );
}

// ── Quick prompt button ───────────────────────────────────────────────────────
export function QuickPromptButton({ text, agentColor, onClick }) {
  const [hovered, setHovered] = React.useState(false);
  return (
    <button
      onClick={onClick}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        background: hovered ? `${agentColor}15` : "rgba(255,255,255,0.04)",
        border: `1px solid ${hovered ? agentColor + "50" : "rgba(255,255,255,0.1)"}`,
        borderRadius: 12,
        padding: "12px 16px",
        color: hovered ? "white" : "#cbd5e1",
        fontSize: 13,
        cursor: "pointer",
        textAlign: "left",
        transition: "all 0.15s ease",
        lineHeight: 1.4,
        width: "100%",
      }}
    >
      {text}
    </button>
  );
}

// ── Error banner ──────────────────────────────────────────────────────────────
export function ErrorBanner({ message }) {
  return (
    <div
      style={{
        background: "#fef2f2",
        border: "1px solid #fecaca",
        borderRadius: 12,
        padding: "12px 16px",
        color: "#dc2626",
        fontSize: 13,
        marginTop: 8,
        display: "flex",
        alignItems: "flex-start",
        gap: 8,
      }}
    >
      <span>⚠️</span>
      <span>{message}</span>
    </div>
  );
}

// ── Status badge ──────────────────────────────────────────────────────────────
export function StatusBadge({ online }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        background: online ? "rgba(34,197,94,0.12)" : "rgba(239,68,68,0.12)",
        border: `1px solid ${online ? "rgba(34,197,94,0.3)" : "rgba(239,68,68,0.3)"}`,
        color: online ? "#22c55e" : "#ef4444",
        padding: "3px 10px",
        borderRadius: 20,
        fontSize: 11,
        fontWeight: 600,
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: online ? "#22c55e" : "#ef4444",
          animation: online ? "pulseGlow 2s ease-in-out infinite" : "none",
        }}
      />
      {online ? "Server connected" : "Server offline"}
    </span>
  );
}
