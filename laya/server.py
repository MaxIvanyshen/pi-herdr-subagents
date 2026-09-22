"""Laya sidecar: decides whether an idle subagent is done or waiting for the user.

  uv run --python 3.12 --with laya python laya/server.py          # serve on 127.0.0.1:8771
  uv run --python 3.12 --with laya python laya/server.py --eval   # held-out accuracy

POST /decide {"text"}          -> {"p_done", "threshold", "done"}
POST /label  {"text", "done"}  -> appends to ~/.pi/laya/labels.jsonl, refits threshold

Training data: laya/examples.jsonl seeds + your labels + past subagent sessions replayed
from ~/.pi/agent/sessions. A wrong exit is cheap (the parent can resume the subagent);
a wrong keep-open stalls the parent — the threshold fit weighs that by KEEP_OPEN_COST.
"""
import json, os, random, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
import laya

PORT = int(os.environ.get("LAYA_PORT", "8771"))
KEEP_OPEN_COST = float(os.environ.get("LAYA_KEEP_OPEN_COST", "5"))
SEEDS = Path(__file__).with_name("examples.jsonl")
LABELS = Path.home() / ".pi" / "laya" / "labels.jsonl"
SESSIONS = Path.home() / ".pi" / "agent" / "sessions"
# Ask what state the text describes, not what to do about it: "does it need the reader?"
# wordings scored inverted in the sweep (AUC 0.32); this one scored AUC 0.95.
QUESTION = {
    "state": {
        "type": "choice",
        "instructions": "What state is the author's work in at the end of this message?",
        "criteria": {
            "finished": "the work is complete; it may list caveats, findings or recommendations for later",
            "blocked": "the work stopped, waiting for the reader's answer, decision, approval or missing information",
            "unsure": "the author is not confident the result is correct and wants the reader to check it",
        },
    }
}

agent = laya.load("convaiinnovations/laya")


def p_done(text: str) -> float:
    # English ckpt reads 512 tokens and cuts from the end; the verdict lives in the tail.
    return agent.predict(text[-1500:], QUESTION)["answers"]["state"]["probabilities"]["finished"]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def replay(root: Path) -> list[dict]:
    """Past subagent sessions: a reply that ended the session (or called subagent_done) was done;
    a caller_ping was a question. A user reply after a report is NOT counted as needs-input —
    in practice those were follow-up requests after finished work."""
    out = []
    for f in root.rglob("*.jsonl"):
        lines = f.read_text().splitlines()
        if not lines or "parentSession" not in lines[0]:
            continue  # only subagent sessions
        msgs = [e["message"] for e in map(json.loads, lines) if e.get("type") == "message"]
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant" or not isinstance(m.get("content"), list):
                continue
            text = "\n".join(c["text"] for c in m["content"] if c.get("type") == "text").strip()
            calls = {c.get("name"): c for c in m["content"] if c.get("type") == "toolCall"}
            if "caller_ping" in calls:
                text, done = calls["caller_ping"]["arguments"].get("message", ""), False
            elif "subagent_done" in calls or (m.get("stopReason") == "stop" and i + 1 == len(msgs)):
                done = True
            else:
                continue
            if text:
                out.append({"text": text, "done": done, "src": "replay"})
    return out


def fit(scored: list[tuple[float, bool]]) -> float:
    """Threshold minimising cost of `p >= t -> exit`; ties go to the lower (exit-happier) t."""
    # ponytail: brute force over midpoints, O(n^2); fine until labels reach the thousands.
    ps = sorted({p for p, _ in scored})
    cands = sorted([ps[0] - 1e-9, ps[-1] + 1e-9] + [(a + b) / 2 for a, b in zip(ps, ps[1:])])
    cost = lambda t: sum(KEEP_OPEN_COST if d else 1 for p, d in scored if (p >= t) != d)
    return min(cands, key=cost)


examples = load(SEEDS) + load(LABELS) + replay(SESSIONS)
scored = [(p_done(e["text"]), e["done"]) for e in examples]

if "--eval" in sys.argv:
    idx = list(range(len(scored)))
    random.Random(0).shuffle(idx)
    cut = int(len(idx) * 0.7)
    t = fit([scored[i] for i in idx[:cut]])
    test = [(scored[i], examples[i]) for i in idx[cut:]]
    done = [(p, e) for (p, d), e in test if d]
    need = [(p, e) for (p, d), e in test if not d]
    print(f"train={cut} test={len(test)} (done={len(done)} needs-input={len(need)}) threshold={t:.3f}")
    print(f"  done kept open (bad, stalls parent): {sum(p < t for p, _ in done)}/{len(done)}")
    print(f"  needs-input exited (cheap, resumable): {sum(p >= t for p, _ in need)}/{len(need)}")
    for p, e in sorted(done + need, key=lambda x: x[0]):
        if (p >= t) != e["done"]:
            print(f"  miss p={p:.3f} {'done' if e['done'] else 'needs-input'}: {e['text'][-100:]!r}")
    sys.exit(0)

threshold = fit(scored)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        global threshold
        if self.path not in ("/decide", "/label"):
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        p = p_done(body["text"])
        if self.path == "/decide":
            out = {"p_done": p, "threshold": threshold, "done": p >= threshold}
        else:
            LABELS.parent.mkdir(parents=True, exist_ok=True)
            with LABELS.open("a") as f:
                f.write(json.dumps({"text": body["text"], "done": bool(body["done"])}) + "\n")
            scored.append((p, bool(body["done"])))
            threshold = fit(scored)
            out = {"threshold": threshold}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_):
        pass


print(f"laya sidecar on 127.0.0.1:{PORT} threshold={threshold:.3f} (n={len(scored)})", flush=True)
HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
