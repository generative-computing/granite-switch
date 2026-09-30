# SPDX-License-Identifier: Apache-2.0
"""Query rewrite: LLM-as-a-judge grading of the rewritten question.

Port of the internal ``score_query_rewrite_predictions.py``:

1. Parse the rewrite from the model output, falling back to the original
   question when nothing parses.
2. Ask a Llama-3.3-70B judge whether it matches the golden rewrite
   (``{"Grade": "1"}`` or ``{"Grade": "0"}``).
3. Accuracy is correct / graded. Rows the judge did not grade are left
   out, as internally. That covers two cases, counted apart:

   * ``ungraded``: the judge replied without a grade, e.g. it started
     reasoning and ran out of tokens. This is a property of the row.
   * ``judge_errors``: the call failed (HTTP error, timeout). Above
     ``MAX_ERROR_FRACTION`` of rows the cell is an error instead, so a flaky
     judge is re-run rather than cached.

The judge prompt is not part of this repository. It is read from
``judge_prompt.txt`` in the staged eval folder, as a ``str.format`` template
with the five slots used below. The endpoint and key come from the
environment (``ADAPTER_BENCH_JUDGE_URL``, ``RITS_API_KEY``). If any of them
is missing, or the judge does not answer, the cell is skipped.

The call uses the standard library only (the benchmarked commit's
virtualenv has no ``openai`` package) and sends the same request the
internal scorer sends through the OpenAI client.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import ScoreContext, ScoreResult, ScorerUnavailable, require

DEFAULT_JUDGE_MODEL = "meta-llama/llama-3-3-70b-instruct"
PROMPT_FILE = "judge_prompt.txt"
PLACEHOLDER = "YOUR_REWRITTEN_QUESTION_HERE"
RETRIES = 5
WORKERS = 8
TIMEOUT_S = 120
# Above this share of failed judge calls the cell is an error (not cached),
# so a flaky judge does not publish an accuracy over a small graded subset.
MAX_ERROR_FRACTION = 0.05
UNGRADED = "no grade in reply"
# The judge is asked for '{"Grade": "1"}' but may add text around it, or
# be cut off by max_tokens after the grade, so the grade is searched for.
GRADE = re.compile(r'"Grade"\s*:\s*"?([01])\b')


def remove_invalid_escapes(s: str) -> str:
    return re.sub(r"\\(?![\"\\/bfnrt]|u[0-9a-fA-F]{4})", "", s)


def parse_rewrite(generated: str | None, fallback: str | None) -> str:
    """'{"rewritten_question": "Q"}' -> Q; the fallback if nothing parses."""
    if generated is None:
        return fallback or ""
    text = generated.strip()
    m = re.search(r'\{\s*"rewritten_question"\s*:\s*"[^"]+"\s*\}', text)
    cand = m.group(0) if m else text
    try:
        parsed = json.loads(remove_invalid_escapes(cand))
        if isinstance(parsed, dict) and "rewritten_question" in parsed:
            rw = parsed["rewritten_question"]
            if rw and rw != PLACEHOLDER:
                return rw
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\{\s*"rewritten_question"\s*:\s*"([^"]+)"\s*\}', text)
    if m:
        return m.group(1)
    parts = text.split('"rewritten_question":', 1)
    if len(parts) > 1:
        q = re.search(r'"([^"]+)"', parts[1])
        if q:
            return q.group(1)
        return re.split(r"[,}\n]", parts[1].strip())[0].strip().strip('"')
    return fallback or ""


class Judge:
    def __init__(self, url: str, key: str, model: str, template: str):
        self.endpoint = url.rstrip("/") + "/chat/completions"
        self.key = key
        self.model = model
        self.template = template

    def complete(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 20,
                "temperature": 0,
                "stop": ["}"],
            }
        ).encode()
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.key}",
                "RITS_API_KEY": self.key,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read())
        return payload["choices"][0]["message"]["content"]

    def grade(self, gt: dict, rewrite: str) -> tuple[int | None, str | None]:
        """``(1 or 0, None)``, or ``(None, why the last attempt failed)``.

        A reply without a grade is retried too, as the internal scorer does.
        """
        prompt = self.template.format(
            previous_question=gt["previous_question"],
            previous_answer=gt["previous_answer"],
            current_question=gt["current_question"],
            golden_rewritten_question=gt["golden_rewritten_question"],
            rewritten_question=rewrite,
        )
        reason = None
        for attempt in range(RETRIES + 1):
            try:
                reply = self.complete(prompt)
            except urllib.error.HTTPError as e:
                reason = f"HTTP {e.code}"
            except Exception as e:
                reason = type(e).__name__
            else:
                m = GRADE.search(reply)
                if m:
                    return int(m.group(1)), None
                reason = f"{UNGRADED}: {reply.strip()[:80]!r}"
            time.sleep(2 * (attempt + 1))
        return None, reason


def make_judge(ctx: ScoreContext) -> Judge:
    url = ctx.env.get("ADAPTER_BENCH_JUDGE_URL")
    key = ctx.env.get("RITS_API_KEY")
    prompt_path = ctx.eval_dir / PROMPT_FILE
    if not url or not key:
        raise ScorerUnavailable("judge endpoint or key not configured")
    if not prompt_path.is_file():
        raise ScorerUnavailable("judge prompt not staged")
    model = ctx.env.get("ADAPTER_BENCH_JUDGE_MODEL") or DEFAULT_JUDGE_MODEL
    judge = Judge(url, key, model, prompt_path.read_text())
    try:
        judge.complete("Reply with {}")
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        raise ScorerUnavailable("judge unreachable") from e
    return judge


def score(rows: list[dict], ctx: ScoreContext) -> ScoreResult:
    parsed = []
    for r in rows:
        gt = require(r, "ground_truth")
        rewrite = parse_rewrite(r.get("generated_content"), gt.get("current_question"))
        parsed.append((gt, rewrite, r.get("qr_standalone", gt.get("llama_standalone"))))

    judge = make_judge(ctx)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        outcomes = list(ex.map(lambda p: judge.grade(p[0], p[1]), parsed))

    valid = correct = errors = ungraded = 0
    by_split = {"standalone": [0, 0], "non-standalone": [0, 0]}
    graded = []
    failures: Counter[str] = Counter()
    for (gt, rewrite, split), (g, reason) in zip(parsed, outcomes, strict=True):
        graded.append(
            {**gt, "rewritten_question": rewrite, "grade": g, "judge_error": reason}
        )
        if g is None:
            failures[reason] += 1
            if reason.startswith(UNGRADED):
                ungraded += 1
            else:
                errors += 1
            continue
        valid += 1
        correct += g == 1
        if split in by_split:
            by_split[split][1] += 1
            by_split[split][0] += g == 1

    total = len(parsed)
    if failures:
        # Goes to the pod log only; the published cell never carries it.
        print(f"[query_rewrite] judge failures: {dict(failures.most_common())}")
    if total and errors / total > MAX_ERROR_FRACTION:
        raise RuntimeError(f"judge calls failed on {errors}/{total} rows")
    metrics = {
        "accuracy_over_valid": correct / valid if valid else 0.0,
        "accuracy_over_total": correct / total if total else 0.0,
        "judge_errors": errors,
        "ungraded": ungraded,
        "n": total,
    }
    details = {
        "judge_failures": dict(failures),
        "total_valid": valid,
        "total_correct": correct,
        "by_standalone": {
            k: {
                "correct": v[0],
                "valid": v[1],
                "accuracy": v[0] / v[1] if v[1] else 0.0,
            }
            for k, v in by_split.items()
        },
        "graded": graded,
    }
    return ScoreResult(metrics=metrics, details=details)
