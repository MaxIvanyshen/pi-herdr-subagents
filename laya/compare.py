"""Laya (local) vs TypeSafe Jev (api.typesafe.ai) on the exit-or-keep-open decision,
across three input variants: final reply only / + the task brief / + a last-turn tool summary.

  TYPESAFE_API_KEY=... uv run --python 3.12 --with laya python laya/compare.py [--no-replay]

Same data as `server.py --eval` (seeds + your labels + replayed subagent sessions), same 70/30
split and cost-weighted threshold fit for every model x variant. Seeds tagged "pair" share a
final message but differ in brief or last turn; the pair score says whether a variant uses that.
--no-replay sends only laya/examples.jsonl + labels to Jev, not your past session output.
Per-example scores are written to $COMPARE_OUT (default /tmp/laya-vs-jev.jsonl).
"""
import json, os, ssl, statistics, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

import certifi  # comes with laya's deps; python.org builds ship without a CA bundle

from server import LABELS, QUESTION, SEEDS, SESSIONS, agent, fit, load, replay, split

KEY = os.environ.get("TYPESAFE_API_KEY") or sys.exit("set TYPESAFE_API_KEY")
JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-latest")
TLS = ssl.create_default_context(cafile=certifi.where())
# Seeds without a recorded last turn get a neutral one, so its presence can't leak the label.
NEUTRAL_TURN = "The final turn ran 2 tool calls and none failed. The last tool call (read) succeeded."

Q_REPLY = QUESTION["state"]  # what server.py ships
Q_CTX = {
    "type": "choice",
    "instructions": "The author was given the task below. What state is their work in at the end of their final message?",
    "criteria": {
        "finished": "the task as given is done, including tasks that only asked for a report, review or options",
        "blocked": "the work stopped, waiting for the reader's answer, decision, approval or missing information",
        "unsure": "the result may be wrong or incomplete and the reader should check it",
    },
}
# Laya cuts over-long state from the END, so the final message goes first.
VARIANTS = {
    "reply": (Q_REPLY, lambda e: e["text"][-1500:]),
    "+task": (Q_CTX, lambda e: {"final_message": e["text"][-1500:], "task": e.get("task", "")[:1000]}),
    "+task+turn": (Q_CTX, lambda e: {"final_message": e["text"][-1500:], "last_turn": e.get("last_turn", NEUTRAL_TURN),
                                     "task": e.get("task", "")[:1000]}),
}


def laya_score(q: dict, state) -> tuple[float, float]:
    t0 = time.perf_counter()
    p = agent.predict(state, {"q": q})["answers"]["q"]["probabilities"]["finished"]
    return p, time.perf_counter() - t0


def jev_score(q: dict, state) -> tuple[float, float]:
    body = json.dumps({"model": JEV_MODEL, "state": state, "questions": {"q": q}}).encode()
    for attempt in range(5):
        req = urllib.request.Request(JEV_URL, body, {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30, context=TLS) as r:
                p = json.load(r)["answers"]["q"]["probabilities"]["finished"]
            return p, time.perf_counter() - t0
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 4:
                sys.exit(f"jev HTTP {e.code}: {e.read().decode()[:500]}")
            time.sleep(2 ** attempt)


def auc(scores: list[float], labels: list[bool]) -> float:
    """P(a random done example scores above a random needs-input one)."""
    pos = [s for s, d in zip(scores, labels) if d]
    neg = [s for s, d in zip(scores, labels) if not d]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


examples = load(SEEDS) + load(LABELS) + ([] if "--no-replay" in sys.argv else replay(SESSIONS))
labels = [e["done"] for e in examples]
pairs = {}
for i, e in enumerate(examples):
    if "pair" in e:
        pairs.setdefault(e["pair"], {})[e["done"]] = i
print(f"n={len(examples)} done={sum(labels)} needs-input={len(labels) - sum(labels)} pairs={len(pairs)}", flush=True)
q, f = VARIANTS["+task+turn"]
print("jev probe ...", jev_score(q, f(examples[0]))[0], flush=True)  # fail fast on key/schema

scores = {}  # (model, variant) -> [(p, latency)]
for v, (q, f) in VARIANTS.items():
    print(f"laya {v} ...", flush=True)
    scores["laya", v] = [laya_score(q, f(e)) for e in examples]
    print(f"jev  {v} ...", flush=True)
    with ThreadPoolExecutor(8) as pool:  # network-bound, unlike laya's single GPU
        scores["jev", v] = list(pool.map(lambda e: jev_score(q, f(e)), examples))

train, test = split(len(examples))
n_done = sum(labels[i] for i in test)
n_need = len(test) - n_done
print(f"\ntest split: done={n_done} needs-input={n_need}  (threshold fit on train, keep-open error costs 5x)")
print("pairs: same final message, brief/last turn flips the label; ok = done member scored higher\n")
print(f"{'model':5} {'variant':11} {'AUC(all)':>8} {'thresh':>7} {'done kept open':>15} {'needs-input caught':>19} {'pairs ok':>9} {'median ms':>10}")
for (m, v), res in scores.items():
    s = [p for p, _ in res]
    t = fit([(s[i], labels[i]) for i in train])
    kept = sum(s[i] < t for i in test if labels[i])
    caught = sum(s[i] < t for i in test if not labels[i])
    ok = sum(s[p[True]] > s[p[False]] for p in pairs.values())
    ms = statistics.median(lat for _, lat in res) * 1000
    print(f"{m:5} {v:11} {auc(s, labels):8.3f} {t:7.3f} {kept:>9}/{n_done:<5} {caught:>12}/{n_need:<6} {ok:>6}/{len(pairs)} {ms:>10.0f}")

out = os.environ.get("COMPARE_OUT", "/tmp/laya-vs-jev.jsonl")
with open(out, "w") as fh:
    for i, e in enumerate(examples):
        row = {"done": e["done"], "pair": e.get("pair"), "text": e["text"][-300:], "task": e.get("task", "")[:200]}
        row.update({f"{m}{v}": round(scores[m, v][i][0], 4) for m, v in scores})
        fh.write(json.dumps(row) + "\n")
print(f"\nper-example scores -> {out}")
