const API_BASE = "/api";

/**
 * Send a chat message to the selected agent via the backend proxy.
 * @param {Object} opts
 * @param {string}   opts.agentId      - One of: researcher | comms | analyst | ops
 * @param {Array}    opts.messages     - Full conversation history [{role, content}]
 * @returns {Promise<{content: string, usage: object}>}
 */
export async function sendAgentMessage({ agentId, messages }) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ agentId, messages }),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `Server error ${res.status}`);
  }

  return res.json(); // { content, usage, stop_reason }
}

/**
 * Check if the backend server is reachable.
 * @returns {Promise<boolean>}
 */
export async function checkServerHealth() {
  try {
    const res = await fetch(`${API_BASE.replace("/api", "")}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    return res.ok;
  } catch {
    return false;
  }
}
