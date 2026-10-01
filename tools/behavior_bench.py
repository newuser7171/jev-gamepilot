"""Score the Fastino GLiNER tier on respanai/behavior-benchmark.

Row = (trace in OpenAI chat format, plain-language behavior definition, label).
Predict present/absent/not_observable via POST /v1/chat/completions with
schema.classifications. Reports F1 on `present` per the dataset card protocol
(per-split, bootstrap over task_id) plus per-behavior-type breakdown.

  python tools/behavior_bench.py --selftest
  python tools/behavior_bench.py --split core --n 0 --seed 1 --out bench.json
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

LABELS = ["present", "absent", "not_observable"]
DEFAULT_DATA = os.path.join(os.environ.get("TEMP", "."), "opencode", "behavior-benchmark")


def load_dotenv(repo_root):
    path = os.path.join(repo_root, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v


def render_trace(row, per_msg=400, total_cap=12000):
    parts = []
    for m in row.get("input") or []:
        role = m.get("role") or "?"
        content = (m.get("content") or "").strip().replace("\n", " ")
        seg = f"{role}: {content[:per_msg]}"
        for tc in m.get("tool_calls") or []:
            if isinstance(tc, dict):
                args = str(tc.get("arguments") or "")[:300]
                seg += f" [tool {tc.get('name')}({args})]"
        parts.append(seg)
    out = row.get("output") or {}
    parts.append("OUTPUT: " + (out.get("content") or "").strip().replace("\n", " ")[:per_msg])
    md = row.get("metadata") or {}
    parts.append(
        "METADATA: "
        + ", ".join(f"{k}={md.get(k)}" for k in ("model", "latency_ms", "finish_reason", "tools_defined"))
    )
    text = "\n".join(parts)
    if len(text) > total_cap:
        half = total_cap // 2
        text = text[:half] + "\n...\n" + text[-half:]
    return text


def extract_label(content):
    parsed = content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except ValueError:
            return None, None
    if not isinstance(parsed, dict):
        return None, None
    node = parsed
    if "classifications" in parsed and isinstance(parsed["classifications"], dict):
        node = parsed["classifications"]
    raw = node.get("shows_behavior")
    conf = None
    if isinstance(raw, dict):
        for key in ("confidence", "score", "prob", "probability"):
            if isinstance(raw.get(key), (int, float)) and not isinstance(raw.get(key), bool):
                conf = float(raw[key])
                if conf > 1.0 and conf <= 100.0:
                    conf /= 100.0
                break
        for key in ("label", "value", "class", "choice", "text", "answer"):
            if isinstance(raw.get(key), str):
                raw = raw[key]
                break
        else:
            probs = raw.get("probabilities")
            if isinstance(probs, dict) and probs:
                best = max(probs.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else -1)
                raw, conf = str(best[0]), float(best[1])
    if isinstance(raw, str) and raw.strip() in LABELS:
        return raw.strip(), conf
    return None, None


def predict_fastino(base_url, key, model, trace_text, behavior_def, timeout, opener=None):
    prompt = (
        "Classify whether the following LLM agent trace shows the behavior.\n\n"
        f"TRACE:\n{trace_text}\n\nBEHAVIOR: {behavior_def}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "schema": {"classifications": [{"task": "shows_behavior", "labels": LABELS}]},
        "include_confidence": True,
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key, "User-Agent": "jev-gamepilot/2.0"},
    )
    open_fn = opener or urllib.request.urlopen
    with open_fn(req, timeout=timeout) as resp:
        data = json.load(resp)
    content = (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
    return extract_label(content)


def present_f1(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "present" and p == "present")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != "present" and p == "present")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "present" and p != "present")
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0


def macro_f1(y_true, y_pred):
    scores = []
    for c in LABELS:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / len(scores)


def compute_metrics(rows, reps=1000, seed=7):
    y_true = [r["label"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    n = len(rows)
    confusion = {t: {p: 0 for p in LABELS} for t in LABELS}
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1
    per_type = {}
    for btype in sorted({r["behavior_type"] for r in rows}):
        sub = [r for r in rows if r["behavior_type"] == btype]
        per_type[btype] = {
            "n": len(sub),
            "present_f1": round(present_f1([r["label"] for r in sub], [r["pred"] for r in sub]), 4),
        }
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    boot = []
    for _ in range(reps):
        draw = [rng.choice(task_ids) for _ in task_ids]
        sample = [r for tid in draw for r in by_task[tid]]
        boot.append(present_f1([r["label"] for r in sample], [r["pred"] for r in sample]))
    boot.sort()
    lo = boot[int(0.025 * (reps - 1))]
    hi = boot[int(0.975 * (reps - 1))]
    return {
        "n": n,
        "present_f1": round(present_f1(y_true, y_pred), 4),
        "present_f1_ci95_task_bootstrap": [round(lo, 4), round(hi, 4)],
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "accuracy": round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / n, 4) if n else 0.0,
        "confusion": confusion,
        "per_behavior_type": per_type,
    }


def load_rows(data_dir, split, n, seed):
    import pyarrow.parquet as pq

    path = os.path.join(data_dir, f"{split}.parquet")
    if not os.path.isfile(path):
        sys.exit(f"missing {path} — run: hf download respanai/behavior-benchmark --type dataset --local-dir {data_dir}")
    table = pq.read_table(path)
    rows = table.to_pylist()
    if n and n < len(rows):
        rng = random.Random(seed)
        rows = rng.sample(rows, n)
    return rows


def selftest():
    rng = random.Random(3)
    types = ["agentic_process", "surface_style", "quantitative_metadata"]
    rows = []
    for i in range(60):
        true = LABELS[i % 3]
        pred = true if rng.random() > 0.25 else rng.choice(LABELS)
        rows.append({
            "task_id": f"t{i // 3}",
            "label": true,
            "pred": pred,
            "behavior_type": types[i % 3],
        })
    m = compute_metrics(rows, reps=200)
    assert 0.0 <= m["present_f1"] <= 1.0 and 0.0 <= m["macro_f1"] <= 1.0
    assert sum(sum(v.values()) for v in m["confusion"].values()) == 60
    trace = render_trace({"input": [{"role": "user", "content": "hi", "tool_calls": None}],
                          "output": {"content": "ok"}, "metadata": {"model": "m", "latency_ms": 1}})
    assert "user: hi" in trace and "OUTPUT: ok" in trace
    lab, conf = extract_label('{"shows_behavior": {"label": "present", "confidence": 0.91}}')
    assert (lab, round(conf, 2)) == ("present", 0.91)
    lab2, _ = extract_label('{"shows_behavior": "absent"}')
    assert lab2 == "absent"
    print("SELFTEST_OK", json.dumps(m, sort_keys=True)[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="core", choices=["core", "multilingual"])
    ap.add_argument("--n", type=int, default=0, help="0 = all rows")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=6.0)
    ap.add_argument("--gap", type=float, default=0.25, help="min seconds between calls")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(repo_root)
    key = os.environ.get("FASTINO_API_KEY", "").strip()
    if not key:
        sys.exit("FASTINO_API_KEY missing (set in .env or environment)")
    base_url = os.environ.get("FASTINO_URL", "https://api.fastino.ai/v1")
    model = os.environ.get("FASTINO_GLINER_MODEL", "fastino/gliner2.5-multi-v1")

    rows = load_rows(args.data, args.split, args.n, args.seed)
    print(f"rows={len(rows)} split={args.split} model={model}")

    scored, errors = [], []
    for i, r in enumerate(rows):
        try:
            lab, conf = predict_fastino(
                base_url, key, model, render_trace(r), r.get("behavior_definition") or "", args.timeout
            )
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = (e.read() or b"")[:200].decode("utf-8", "replace")
            except Exception:
                pass
            code = getattr(e, "code", None)
            print(f"HTTP {code} at row {i}: {body}")
            if code in (401, 403, 404):
                sys.exit("auth/model error — aborting")
            if code in (402, 503):
                sys.exit("billing blocked (agent.pioneer.ai/billing) — aborting; rerun after fix")
            errors.append(code)
            if len(errors) >= 3:
                sys.exit("3 consecutive HTTP failures — aborting")
            time.sleep(1.0)
            continue
        except Exception as e:
            print(f"row {i} failed: {e}")
            errors.append(str(e))
            if len(errors) >= 3:
                sys.exit("3 consecutive failures — aborting")
            time.sleep(1.0)
            continue
        errors.clear()
        if lab is None:
            lab = "not_observable"
        scored.append({
            "task_id": r.get("task_id"),
            "def_id": r.get("def_id"),
            "behavior_type": r.get("behavior_type"),
            "label": r.get("label"),
            "pred": lab,
            "confidence": conf,
        })
        if (i + 1) % 25 == 0:
            print(f"scored {i + 1}/{len(rows)}")
        time.sleep(args.gap)

    if not scored:
        sys.exit("no rows scored")
    metrics = compute_metrics(scored)
    result = {"split": args.split, "n_requested": len(rows), "model": model, "metrics": metrics}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
