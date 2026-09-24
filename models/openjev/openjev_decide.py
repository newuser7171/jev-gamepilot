"""Typed-decision wrapper for openjev, shaped for JevBench's `local_openjev` adapter.

The adapter (github.com/fstandhartinger/jevbench, jevbench/adapters/local_openjev.py) expects
`typed_decisions.open_jev.OpenJev`: `OpenJev.from_pretrained(path, device=...)`, then
`model.decide(state, questions)` with `questions = [{"type": "noul"|"choice"|"score",
"instructions": str, "options": [str]}]`, returning one answer per question — `{"noul": p}` for
noul, `{"probabilities": {label: p}}` otherwise. It also reads `model.tok` and, if present,
`model.collator.max_state`.

openjev is a 3-way NLI cross-encoder, so a decision is scored as entailment: every option becomes
one hypothesis over the state and the distribution is P(entailment) normalised over the options —
the model's own softmax, one forward pass per option, nothing generated.

    from openjev_decide import OpenJev
    jev = OpenJev.from_pretrained("AlexWortega/openjev", subfolder="qwen3.5-0.8b-nli-v2s", device="cuda")
    jev.decide("Policy: refunds require a receipt...",
               [{"type": "noul", "instructions": "Is the refund permitted?",
                 "options": ["no", "yes"]}])
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from modeling_openjev import ENT, OpenJevCrossEncoder

RUBRIC_MARK = "\nAllowed answers and rubric: "
TEMPLATE = 'The answer to "{instr}" is {label}: {crit}'
WINDOW_CHARS = 24_000  # one window fits the 8k-token encoder; longer states are scored window by window


class Collator:
    """Only `max_state` is read by the adapter (to report truncation); state longer than one window is
    windowed instead of cut, so nothing is silently dropped."""

    def __init__(self, max_state):
        self.max_state = max_state


class OpenJev:
    def __init__(self, ce: OpenJevCrossEncoder, window_chars: int = WINDOW_CHARS):
        self.ce = ce
        self.tok = ce.tok
        self.window_chars = window_chars
        self.collator = Collator(ce.max_len)

    @classmethod
    def from_pretrained(cls, path: str, subfolder: str | None = None, device: str | None = None, **kw) -> "OpenJev":
        # the JevBench adapter always asks for "cpu"; OPENJEV_DEVICE / OPENJEV_DTYPE let the host override it
        device = os.environ.get("OPENJEV_DEVICE") or device
        if "OPENJEV_DTYPE" in os.environ:
            kw.setdefault("dtype", getattr(torch, os.environ["OPENJEV_DTYPE"]))
        return cls(OpenJevCrossEncoder(path, subfolder=subfolder, device=device, **kw))

    @staticmethod
    def _rubric(instructions: str, options: list) -> tuple:
        """The adapter appends the rubric to the instruction text; take it back apart when it is there."""
        instr, crits = instructions, {}
        if RUBRIC_MARK in instructions:
            instr, _, tail = instructions.partition(RUBRIC_MARK)
            try:
                crits = json.loads(tail)
            except json.JSONDecodeError:
                crits = {}
        return instr.strip(), {o: str(crits.get(o, o)) for o in options}

    def _windows(self, state: str) -> list:
        if len(state) <= self.window_chars:
            return [state]
        step = self.window_chars - 2000
        return [state[s:s + self.window_chars] for s in range(0, max(len(state) - 2000, 1), step)]

    def decide(self, state, questions: list) -> list:
        state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        windows = self._windows(state)
        answers = []
        for q in questions:
            options = list(q["options"])
            instr, crits = self._rubric(q["instructions"], options)
            pairs = [(w, TEMPLATE.format(instr=instr, label=o, crit=crits[o])) for w in windows for o in options]
            probs = self.ce.predict(pairs)[:, ENT].reshape(len(windows), len(options))
            p = probs.max(0)  # a claim supported by any window is supported by the document
            p = p / max(float(p.sum()), 1e-9)
            if q["type"] == "noul":
                yes = options.index("yes") if "yes" in options else len(options) - 1
                answers.append({"noul": float(p[yes])})
            else:
                answers.append({"probabilities": {o: float(x) for o, x in zip(options, p)}})
        return answers


def _demo():  # pragma: no cover - manual check
    jev = OpenJev.from_pretrained("AlexWortega/openjev", subfolder="qwen3.5-0.8b-nli-v2s")
    print(jev.decide(
        "Policy: refunds require a receipt and purchase within 30 days. A customer bought 12 days ago"
        " but has no receipt. Issue a refund.",
        [{"type": "noul", "instructions": "Under the stated policy, is the requested action permitted?"
          + RUBRIC_MARK + json.dumps({"no": "A condition is missing.", "yes": "Every condition holds."}),
          "options": ["no", "yes"]}]))


if __name__ == "__main__":
    _demo()
