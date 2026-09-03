#!/usr/bin/env python3
"""Reproduce the hybrid-GDN recurrent-state corruption (loops) on a warm slot.

Trigger: with a thinking template the server generates reasoning_content that the
client does NOT echo back. So the next turn's LCP ends where the thinking block
began, the recurrent tail sits past it, and the warm-slot truncation has to roll
the recurrent cache back further than n_rs_seq. If the checkpoint-recovery gate
is inverted, that rollback silently drops the recurrent state.

Positive control = looping build. Expect DEGENERATE verdicts.

Usage: [LLAMA_HOST=<host>] ./looptest.py [port]
"""
import collections, json, os, re, sys, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8064
HOST = os.environ.get("LLAMA_HOST", "127.0.0.1")

# ~30k tokens of unique-ish context so long-range recurrent state actually matters
DOC = "".join(
    f"Record {i:05d}: service svc-{i%97} in region r{i%13} owns shard {i%211}, "
    f"retry budget {i%7}, backpressure threshold {i%29}, owner team-{i%41}.\n"
    for i in range(2600)
)

# questions whose answers are recoverable from DOC, plus an arithmetic anchor
QUESTIONS = [
    ("Which team owns record 00100? Answer with just the team name.", "team-18"),
    ("What is 7*8? Answer with digits only.",                          "56"),
    ("Which region does record 00013 belong to? Answer like r<N>.",    "r0"),
    ("What is 12*12? Answer with digits only.",                        "144"),
    ("Which shard does record 00211 own? Answer with the number only.","0"),
]

def degenerate(text):
    """crude but effective: heavy 8-gram repetition, or a wall with no answer"""
    words = re.findall(r"\S+", text)
    if len(words) < 24:
        return False, 0.0
    grams = collections.Counter(" ".join(words[i:i+8]) for i in range(len(words)-7))
    top = grams.most_common(1)[0][1]
    ratio = top / max(1, len(words) - 7)
    return ratio > 0.15, ratio

def chat(messages, n=320):
    body = {"model": "x", "messages": messages, "max_tokens": n, "temperature": 0.0}
    rq = urllib.request.Request(f"http://{HOST}:{PORT}/v1/chat/completions",
                                data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(rq, timeout=1200) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    t = d.get("timings", {})
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), \
           d["choices"][0]["finish_reason"], t

hist = [{"role": "user", "content": DOC + "\nAcknowledge with just: READY"}]
c, r, f, t = chat(hist, 16)
print(f"  setup : prompt_n={t.get('prompt_n')} cache_n={t.get('cache_n')} -> {c.strip()[:24]!r}")
# IMPORTANT: append only the visible content, never the reasoning - this is what a
# normal chat client does and it is what creates the rollback every turn.
hist.append({"role": "assistant", "content": c})

bad = 0
for i, (q, expect) in enumerate(QUESTIONS):
    hist.append({"role": "user", "content": q})
    c, r, f, t = chat(hist)
    deg, ratio = degenerate(c + r)
    wrong = expect.lower() not in (c or "").lower()
    flag = "DEGENERATE" if deg else ("no-answer" if wrong else "ok")
    if deg or wrong:
        bad += 1
    print(f"  turn {i}: cache_n={str(t.get('cache_n')):>6} prompt_n={str(t.get('prompt_n')):>6} "
          f"finish={f:6} rep={ratio:.2f} reas={len(r):5d} {flag:11} content={c.strip()[:44]!r}")
    hist.append({"role": "assistant", "content": c})

print(f"  => {bad}/{len(QUESTIONS)} turns bad")
sys.exit(1 if bad else 0)
