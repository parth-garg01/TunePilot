#!/usr/bin/env python3
"""
TunePilot & Claude Code Live Mobile Monitor Server
Single-file, zero-dependency HTTP server with SSE/polling for live monitoring on mobile phones.
"""

from __future__ import annotations

import glob
import http.server
import json
import os
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = 8765
PROJECT_DIR = Path(r"C:\Users\parth\Desktop\New folder")
CLAUDE_PROJECTS_DIR = Path(os.path.expanduser(r"~/.claude/projects/C--Users-parth-Desktop-New-folder"))


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def get_latest_claude_session() -> tuple[Path | None, list[dict[str, Any]], dict[str, Any]]:
    if not CLAUDE_PROJECTS_DIR.exists():
        return None, [], {}

    files = sorted(CLAUDE_PROJECTS_DIR.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not files:
        return None, [], {}

    target = files[0]
    events = []
    stats = {
        "sessionId": target.stem,
        "lastUpdated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(target.stat().st_mtime)),
        "totalEvents": 0,
        "filesCreated": [],
        "currentActivity": "Idle / Waiting",
        "model": "claude-opus-4-7",
        "tokens": {"input": 0, "output": 0, "cache_read": 0},
    }

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    stats["totalEvents"] += 1

                    # Extract Assistant actions
                    if obj.get("type") == "assistant":
                        msg = obj.get("message", {})
                        if "model" in msg:
                            stats["model"] = msg["model"]
                        usage = msg.get("usage", {})
                        if usage:
                            stats["tokens"]["input"] = usage.get("input_tokens", stats["tokens"]["input"])
                            stats["tokens"]["output"] += usage.get("output_tokens", 0)
                            stats["tokens"]["cache_read"] = usage.get("cache_read_input_tokens", stats["tokens"]["cache_read"])

                        for c in msg.get("content", []):
                            if c.get("type") == "tool_use":
                                tool_name = c.get("name")
                                inp = c.get("input", {})
                                act = f"Calling tool {tool_name}"
                                target_path = inp.get("file_path") or inp.get("command") or ""
                                if tool_name == "Write":
                                    fname = Path(target_path).name if target_path else "file"
                                    act = f"Writing {fname}"
                                    if target_path and target_path not in stats["filesCreated"]:
                                        stats["filesCreated"].append(target_path)
                                elif tool_name == "Read":
                                    fname = Path(target_path).name if target_path else "file"
                                    act = f"Reading {fname}"
                                elif tool_name == "Bash":
                                    cmd = (inp.get("command") or "")[:40]
                                    act = f"Running: {cmd}"

                                stats["currentActivity"] = act
                                events.append({
                                    "time": obj.get("timestamp", ""),
                                    "kind": "tool_use",
                                    "tool": tool_name,
                                    "detail": act,
                                    "target": target_path,
                                    "uuid": obj.get("uuid", ""),
                                })
                            elif c.get("type") == "text" and c.get("text"):
                                txt = c.get("text", "").strip()
                                stats["currentActivity"] = txt[:80]
                                events.append({
                                    "time": obj.get("timestamp", ""),
                                    "kind": "assistant_text",
                                    "text": txt,
                                    "uuid": obj.get("uuid", ""),
                                })

                    elif obj.get("type") == "user":
                        msg = obj.get("message", {})
                        if isinstance(msg.get("content"), str):
                            events.append({
                                "time": obj.get("timestamp", ""),
                                "kind": "user_prompt",
                                "text": msg.get("content"),
                            })
                except Exception:
                    continue
    except Exception as e:
        stats["error"] = str(e)

    # Get recent 50 events in reverse chronological order
    recent_events = events[-50:]
    recent_events.reverse()
    return target, recent_events, stats


def get_workspace_tree() -> list[dict[str, Any]]:
    files = []
    for root, dirs, filenames in os.walk(PROJECT_DIR):
        if any(ignored in root for ignored in [".git", "__pycache__", ".pytest_cache", ".venv", "venv", ".idea"]):
            continue
        for f in filenames:
            p = Path(root) / f
            try:
                rel = p.relative_to(PROJECT_DIR).as_posix()
                size = p.stat().st_size
                mtime = time.strftime("%H:%M:%S", time.localtime(p.stat().st_mtime))
                files.append({"path": rel, "size": size, "mtime": mtime})
            except Exception:
                continue
    return sorted(files, key=lambda x: x["path"])


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>TunePilot Live Monitor</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #131b2e;
      --card-border: #1e293b;
      --accent: #38bdf8;
      --accent-glow: rgba(56, 189, 248, 0.2);
      --success: #10b981;
      --warning: #f59e0b;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: var(--font);
      padding: 16px;
      -webkit-font-smoothing: antialiased;
      padding-bottom: 80px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .brand h1 {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(135deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .badge-live {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      padding: 4px 10px;
      border-radius: 9999px;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .pulse-dot {
      width: 7px;
      height: 7px;
      background: #10b981;
      border-radius: 50%;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
      70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 14px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .card.full { grid-column: span 2; }
    .card-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 4px;
    }
    .card-value {
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--text-main);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .activity-banner {
      background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(99, 102, 241, 0.1));
      border: 1px solid rgba(56, 189, 248, 0.3);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 20px;
    }
    .activity-title {
      font-size: 0.8rem;
      color: var(--accent);
      font-weight: 600;
      margin-bottom: 6px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .activity-content {
      font-size: 1.05rem;
      font-weight: 600;
      color: #fff;
      word-break: break-word;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 20px 0 10px 0;
    }
    .section-title {
      font-size: 0.95rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .feed {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .event-item {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 0.88rem;
      transition: all 0.2s ease;
    }
    .event-top {
      display: flex;
      justify-content: space-between;
      margin-bottom: 4px;
      font-size: 0.75rem;
      color: var(--text-muted);
    }
    .tag {
      padding: 2px 6px;
      border-radius: 4px;
      font-weight: 600;
      font-size: 0.7rem;
    }
    .tag.write { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
    .tag.read { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .tag.bash { background: rgba(168, 85, 247, 0.2); color: #c084fc; }
    .tag.text { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .event-text {
      color: var(--text-main);
      font-weight: 500;
      word-break: break-all;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.82rem;
    }
    .files-list {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      overflow: hidden;
    }
    .file-row {
      display: flex;
      justify-content: space-between;
      padding: 10px 14px;
      border-bottom: 1px solid var(--card-border);
      font-size: 0.82rem;
      font-family: ui-monospace, monospace;
    }
    .file-row:last-child { border-bottom: none; }
    .file-name { color: #e2e8f0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 70%; }
    .file-meta { color: var(--text-muted); font-size: 0.75rem; }
    .tabs {
      display: flex;
      background: var(--card-bg);
      border-radius: 10px;
      padding: 4px;
      margin-bottom: 16px;
      border: 1px solid var(--card-border);
    }
    .tab {
      flex: 1;
      text-align: center;
      padding: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-muted);
      border-radius: 8px;
      cursor: pointer;
    }
    .tab.active {
      background: #1e293b;
      color: #fff;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <span style="font-size: 1.4rem;">⚡</span>
      <h1>TunePilot Monitor</h1>
    </div>
    <div class="badge-live">
      <div class="pulse-dot"></div>
      <span>LIVE</span>
    </div>
  </header>

  <div class="activity-banner">
    <div class="activity-title">CURRENT CLAUDE ACTIVITY</div>
    <div id="current-activity" class="activity-content">Connecting...</div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-label">Model</div>
      <div id="model-name" class="card-value">-</div>
    </div>
    <div class="card">
      <div class="card-label">Files Created</div>
      <div id="files-count" class="card-value">0</div>
    </div>
    <div class="card">
      <div class="card-label">Total Steps</div>
      <div id="steps-count" class="card-value">0</div>
    </div>
    <div class="card">
      <div class="card-label">Output Tokens</div>
      <div id="tokens-count" class="card-value">0</div>
    </div>
  </div>

  <div class="tabs">
    <div id="tab-events" class="tab active" onclick="switchTab('events')">Live Activity</div>
    <div id="tab-files" class="tab" onclick="switchTab('files')">Workspace Files</div>
  </div>

  <div id="view-events">
    <div class="section-header">
      <div class="section-title">Session Feed</div>
      <div id="last-updated" style="font-size:0.75rem; color:var(--text-muted)">-</div>
    </div>
    <div id="events-feed" class="feed">
      <!-- Events injected here -->
    </div>
  </div>

  <div id="view-files" style="display: none;">
    <div class="section-header">
      <div class="section-title">Project Tree</div>
      <div id="total-files" style="font-size:0.75rem; color:var(--text-muted)">-</div>
    </div>
    <div id="files-feed" class="files-list">
      <!-- Files injected here -->
    </div>
  </div>

  <script>
    function switchTab(t) {
      if (t === 'events') {
        document.getElementById('view-events').style.display = 'block';
        document.getElementById('view-files').style.display = 'none';
        document.getElementById('tab-events').classList.add('active');
        document.getElementById('tab-files').classList.remove('active');
      } else {
        document.getElementById('view-events').style.display = 'none';
        document.getElementById('view-files').style.display = 'block';
        document.getElementById('tab-events').classList.remove('active');
        document.getElementById('tab-files').classList.add('active');
      }
    }

    async function pollData() {
      try {
        const res = await fetch('/api/state');
        if (!res.ok) return;
        const data = await res.json();
        
        document.getElementById('current-activity').innerText = data.stats.currentActivity || 'Active';
        document.getElementById('model-name').innerText = (data.stats.model || 'Claude').replace('claude-', '');
        document.getElementById('files-count').innerText = (data.stats.filesCreated || []).length;
        document.getElementById('steps-count').innerText = data.stats.totalEvents || 0;
        document.getElementById('tokens-count').innerText = (data.stats.tokens.output || 0).toLocaleString();
        document.getElementById('last-updated').innerText = data.stats.lastUpdated || '';

        // Render Events
        const feed = document.getElementById('events-feed');
        feed.innerHTML = (data.events || []).map(ev => {
          let tagClass = 'text';
          let tagText = 'TEXT';
          let body = ev.text || ev.detail || '';
          if (ev.tool === 'Write') { tagClass = 'write'; tagText = 'WRITE'; }
          else if (ev.tool === 'Read') { tagClass = 'read'; tagText = 'READ'; }
          else if (ev.tool === 'Bash') { tagClass = 'bash'; tagText = 'BASH'; }
          else if (ev.kind === 'user_prompt') { tagClass = 'read'; tagText = 'USER'; }

          return `
            <div class="event-item">
              <div class="event-top">
                <span class="tag ${tagClass}">${tagText}</span>
                <span>${(ev.time || '').substring(11, 19)}</span>
              </div>
              <div class="event-text">${escapeHtml(body)}</div>
            </div>
          `;
        }).join('');

        // Render Files
        const filesFeed = document.getElementById('files-feed');
        document.getElementById('total-files').innerText = `${data.files.length} files`;
        filesFeed.innerHTML = (data.files || []).map(f => `
          <div class="file-row">
            <span class="file-name">${escapeHtml(f.path)}</span>
            <span class="file-meta">${formatBytes(f.size)} • ${f.mtime}</span>
          </div>
        `).join('');

      } catch (err) {
        console.error(err);
      }
    }

    function escapeHtml(str) {
      if (!str) return '';
      return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function formatBytes(bytes) {
      if (!bytes) return '0 B';
      const k = 1024;
      const dm = 1;
      const sizes = ['B', 'KB', 'MB', 'GB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    setInterval(pollData, 1200);
    pollData();
  </script>
</body>
</html>
"""


class MonitorHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif self.path == "/api/state":
            _, events, stats = get_latest_claude_session()
            files = get_workspace_tree()
            payload = json.dumps({"stats": stats, "events": events, "files": files})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return


def main():
    ip = get_local_ip()
    server = socketserver.TCPServer(("0.0.0.0", PORT), MonitorHandler)
    server.allow_reuse_address = True
    print("=" * 60)
    print(" TunePilot Live Mobile Monitor is RUNNING")
    print("=" * 60)
    print(f" Open this URL on your phone browser:")
    print(f" http://{ip}:{PORT}")
    print(f" Or locally: http://localhost:{PORT}")
    print("=" * 60)
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
