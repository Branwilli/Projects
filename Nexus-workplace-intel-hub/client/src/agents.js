export const AGENT_PERSONAS = [
  {
    id: "researcher",
    name: "Research Agent",
    icon: "⚡",
    color: "#6366f1",
    bgLight: "#eef2ff",
    description: "Surfaces real-time market intel, competitor moves, and industry signals",
  },
  {
    id: "comms",
    name: "Comms Agent",
    icon: "✉️",
    color: "#0ea5e9",
    bgLight: "#f0f9ff",
    description: "Drafts emails, Slack messages, and announcements with the right tone",
  },
  {
    id: "analyst",
    name: "Data Analyst Agent",
    icon: "📊",
    color: "#10b981",
    bgLight: "#ecfdf5",
    description: "Interprets metrics, benchmarks performance, and spots anomalies",
  },
  {
    id: "ops",
    name: "Ops Automation Agent",
    icon: "⚙️",
    color: "#f59e0b",
    bgLight: "#fffbeb",
    description: "Maps workflows, identifies bottlenecks, and proposes automation paths",
  },
];

export const QUICK_PROMPTS = {
  researcher: [
    "What are the latest trends in AI-powered workplace tools in 2026?",
    "Summarize recent news about remote work productivity research",
    "What are top competitors doing with AI automation right now?",
  ],
  comms: [
    "Draft a Slack message announcing a new AI tool rollout to my team",
    "Write an email to stakeholders about Q2 performance exceeding targets by 15%",
    "Create an all-hands announcement about a new hybrid work policy",
  ],
  analyst: [
    "Our team's ticket resolution time is 4.2 hours — how does that benchmark?",
    "We have a 73% employee engagement score. What does that mean?",
    "Interpret: 12% MoM growth but churn rose from 2% to 3.1%",
  ],
  ops: [
    "Our onboarding takes 3 weeks and involves 7 tools — help streamline it",
    "Map and optimize a weekly reporting workflow that takes 4 hours",
    "How can we automate our expense approval process end-to-end?",
  ],
};
