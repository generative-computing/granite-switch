# SPDX-License-Identifier: Apache-2.0
"""Write a narrator's training lines with Mellea's instruct-validate-repair loop.

A thinking model (Granite 4.2 30B, served by vLLM's OpenAI-compatible server)
writes one spoken line per speaking moment of a recorded match (``talk.py
moments``), in order within the match, so it sees what it already said. Each line
must pass checks in code (length, openings, PG-13, not a known film line, not a
near-repeat of the match's recent lines) and checks the same model judges with
thinking off (reacts to this moment; in the voice; witty). A failed check's
reason goes back to the writer, which repairs its line (MultiTurnStrategy).

Runs in an environment with Mellea (``pip install mellea``), not the demo's::

    python narrate_ivr.py --moments data/narr/moments.jsonl --out data/narr/crime.jsonl \\
        --base-url http://HOST:PORT/v1 --model granite-4.2-30b

``--judge`` scores lines written elsewhere (a trained narrator's, the base
model's) with the same checks, once, without repair::

    python narrate_ivr.py --judge runs/narr/narrator/heldout_gen.jsonl \\
        --out runs/narr/narrator/judged.jsonl --base-url http://HOST:PORT/v1
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PERSONA = (
    "You write the lines of a character in a Doom deathmatch against bots: a calm, "
    "wisecracking professional out of a 1990s crime movie, talking to himself while "
    "he works. He is wry and snarky, deadpan with a bite, casual about the violence. "
    "He mixes it up: a dry observation, an understatement, a quiet threat, a "
    "rhetorical question, a callback to something he said, and only now and then a "
    "comparison to ordinary life. He never recites numbers. Every line is original: "
    "never quote or paraphrase any film. Mild language at most."
)
TASK = """{persona}

What is happening right now:
{brief}
The last moments of the game log (most recent last):
{recent}
Lines he already said this match (most recent last):
{prev}

Write the one line he says out loud now: 5 to 15 words, reacting to this exact
moment. Output only the line."""

OPENINGS = ("another", "time", "situation")
STRONG = ("fuck", "shit", "cunt", "motherf", "bitch", "nigg", "fag", "retard", "whore")
# Distinctive phrases of well-known film dialogue: a line containing one is refused.
FILM = (
    "royale",
    "say what again",
    "ezekiel",
    "do you speak it",
    "big mac",
    "zed's dead",
    "tasty burger",
    "cornerstone of any nutritious",
    "path of the righteous",
    "get medieval",
    "pretty please with sugar",
    "that's a bingo",
    "winston wolf",
    "i'll be back",
    "hasta la vista",
    "feel lucky, punk",
    "make my day",
    "say hello to my little",
)
JUDGE = {
    "grounded": "Does this line react to something that is actually happening in "
    "the moment described (a frag, a death, damage taken, health, the weapon, a "
    "pickup, a bot in view, or a lull)?",
    "voice": "Does this line sound like a wry, snarky, calm professional from a "
    "1990s crime movie talking to himself (not a soldier, not a mission report, "
    "not a sports announcer)?",
    "witty": "Is this line witty or striking (a turn of phrase, a dry joke or a "
    "vivid comparison) rather than generic?",
}


def clean(text: str) -> str:
    """The spoken line: no reasoning, no quote marks, one line."""
    text = re.sub(r"(?s)<think>.*?</think>", "", str(text))
    text = text.split("</think>")[-1]
    lines = [x.strip().strip('"').strip("“”").strip() for x in text.splitlines()]
    return next((x for x in lines if x), "")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def code_fns(prev: list[str]):
    """Requirements checked in code, as (description, fn); each fn returns
    (ok, reason for the repair)."""

    def length(x):
        n = len(clean(x).split())
        return (5 <= n <= 15), f"The line has {n} words; it must have 5 to 15."

    def opening(x):
        w = words(clean(x))
        ok = bool(w) and w[0] not in OPENINGS
        return ok, f'Do not start with "{w[0] if w else ""}"; start differently.'

    def plain(x):
        c = clean(x)
        ok = not re.search(r"[\[\]*#\"]|reload", c, re.I)
        return ok, "No brackets, asterisks or quote marks, and Doom has no reloading."

    def clean_language(x):
        c = clean(x).lower()
        bad = [s for s in STRONG if s in c]
        return not bad, "Keep the language mild: no strong swearing or slurs."

    def original(x):
        c = clean(x).lower()
        hit = [s for s in FILM if s in c]
        return not hit, "That echoes a famous film line; write an original one."

    def fresh(x):
        w = set(words(clean(x)))
        for p in prev[-5:]:
            q = set(words(p))
            if w and q and len(w & q) / len(w | q) > 0.5:
                return (
                    False,
                    f'Too close to an earlier line ("{p}"); say something new.',
                )
        return True, ""

    def varied(x):
        like = re.compile(r"\blike (a|an|the|my|some)\b|\bfeels like\b", re.I)
        n = sum(bool(like.search(p)) for p in prev[-4:]) + bool(like.search(clean(x)))
        return n <= 2, (
            "Too many of your lines are comparisons (like a ...). Use another form: "
            "a dry observation, an understatement, a quiet threat or a question."
        )

    return [
        ("5 to 15 words", length),
        ("A varied form", varied),
        ("A varied opening", opening),
        ("Plain spoken text", plain),
        ("Mild language", clean_language),
        ("Original, not a film quote", original),
        ("Not a repeat of recent lines", fresh),
    ]


def code_checks(prev: list[str]):
    from mellea.stdlib.requirements import req, simple_validate

    return [req(d, validation_fn=simple_validate(f)) for d, f in code_fns(prev)]


def judge_fn(judge, moment_text: str):
    """The three model-judged requirements in one call (thinking off): the line
    passes only if every verdict is YES; the verdicts are the repair feedback."""
    from mellea.backends import ModelOption

    questions = "\n".join(f"{k}: {q}" for k, q in JUDGE.items())

    def fn(x):
        q = (
            f"Judge a spoken line.\n\nThe moment:\n{moment_text}\n\nThe line: "
            f'"{clean(x)}"\n\nQuestions:\n{questions}\n\nAnswer with exactly '
            "three lines, one per question, each as `name: YES` or `name: NO - "
            "short reason`."
        )
        judge.reset()
        a = str(
            judge.instruct(
                q,
                strategy=None,
                model_options={
                    ModelOption.THINKING: False,
                    ModelOption.TEMPERATURE: 0.0,
                    ModelOption.MAX_NEW_TOKENS: 120,
                },
            )
        )
        verdicts = {k: re.search(rf"{k}\s*:\s*(YES|NO)", a, re.I) for k in JUDGE}
        bad = [k for k, v in verdicts.items() if not v or v.group(1).upper() != "YES"]
        return not bad, (a.strip() if bad else "")

    return fn


def judged_checks(judge, moment_text: str):
    from mellea.stdlib.requirements import req, simple_validate

    fn = judge_fn(judge, moment_text)
    return [req("Grounded, in the voice, witty", validation_fn=simple_validate(fn))]


def moment_text(brief: str, recent: list[str]) -> str:
    """The moment as the judge reads it."""
    return f"{brief}\nLog: " + " / ".join(recent)


def judge_file(args) -> None:
    """Score lines already written (``--judge``: rows with ``brief``, ``recent``,
    ``prev`` and one line per ``--keys`` column, e.g. train_alora.py's
    heldout_gen.jsonl) with the same checks, once each, without repair."""
    from collections import Counter

    from mellea import start_session

    rows = [json.loads(x) for x in open(args.judge)]
    keys = args.keys.split(",")
    urls = args.base_url.split(",")

    def one(ir):
        i, r = ir
        judge = start_session(
            "openai", model_id=args.model, base_url=urls[i % len(urls)], api_key="none"
        )
        mt = moment_text(r["brief"], r["recent"])
        out = {}
        for k in keys:
            fails = [d for d, f in code_fns(r["prev"]) if not f(r[k])[0]]
            ok, why = judge_fn(judge, mt)(r[k])
            if not ok:
                fails.append("judge")
            no = [q for q in JUDGE if re.search(rf"{q}\s*:\s*NO", why, re.I)]
            out[k] = {"ok": not fails, "fails": fails, "judge_no": no, "why": why}
        return {**r, "judged": out}

    with ThreadPoolExecutor(args.concurrency) as pool:
        judged = list(pool.map(one, enumerate(rows)))
    with open(args.out, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")
    n = len(judged)
    for k in keys:
        ok = sum(r["judged"][k]["ok"] for r in judged)
        fails = Counter(x for r in judged for x in r["judged"][k]["fails"])
        no = Counter(x for r in judged for x in r["judged"][k]["judge_no"])
        print(
            f"{k:<8} pass every check {100 * ok / n:5.1f}% (n={n}); failed: "
            + ", ".join(f"{d} {100 * c / n:.1f}%" for d, c in fails.most_common())
            + "; judge said NO to: "
            + ", ".join(f"{q} {100 * c / n:.1f}%" for q, c in no.most_common())
        )
    print(f"-> {args.out}")


def run_match(match: dict, args, write) -> int:
    from mellea import start_session
    from mellea.backends import ModelOption
    from mellea.stdlib.context import ChatContext
    from mellea.stdlib.sampling import MultiTurnStrategy

    urls = args.base_url.split(",")
    kw = {
        "base_url": urls[hash((match["data"], match["ep"])) % len(urls)],
        "api_key": "none",
    }
    writer = start_session(
        "openai",
        model_id=args.model,
        ctx=ChatContext(),
        model_options={
            # Granite 4.2 at "low" effort reasons briefly and closes </think>; at
            # "medium" it deliberated past 2000 tokens without answering.
            ModelOption.THINKING: args.effort,
            ModelOption.TEMPERATURE: args.temperature,
            ModelOption.MAX_NEW_TOKENS: args.max_tokens,
        },
        **kw,
    )
    judge = start_session("openai", model_id=args.model, **kw)
    prev: list[str] = []
    for m in match["moments"]:
        moment = moment_text(m["brief"], m["recent"])
        task = TASK.format(
            persona=PERSONA,
            brief=m["brief"],
            recent="\n".join(m["recent"]),
            prev="\n".join(prev[-5:]) or "(nothing yet)",
        )
        t0 = time.time()
        writer.reset()
        res = writer.instruct(
            task,
            requirements=code_checks(prev) + judged_checks(judge, moment),
            strategy=MultiTurnStrategy(loop_budget=args.loop_budget),
            return_sampling_results=True,
        )
        line = clean(res.result.value if hasattr(res.result, "value") else res.result)
        fails = [
            (r.description, str(v.reason or ""))
            for r, v in (res.result_validations or [])
            if not v.as_bool()
        ]
        write(
            {
                "data": match["data"],
                "style": match["style"],
                "ep": match["ep"],
                "t": m["t"],
                "hist_n": m["hist_n"],
                "line": line,
                "ok": bool(getattr(res, "success", not fails)),
                "attempts": len(getattr(res, "sample_generations", None) or []) or 1,
                "fails": fails,
                "prev": prev[-5:],
                "s": round(time.time() - t0, 1),
            }
        )
        if line:
            prev.append(line)  # what it said, pass or not: the next moment follows it
    return len(match["moments"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moments", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--judge",
        type=Path,
        help="Score lines already written instead (see judge_file); --out gets "
        "the verdicts",
    )
    ap.add_argument(
        "--keys", default="adapter,base", help="--judge: the columns holding lines"
    )
    ap.add_argument("--base-url", required=True, help="One or more, comma-separated")
    ap.add_argument("--model", default="granite-4.2-30b")
    ap.add_argument("--matches", type=int, default=0, help="0: all")
    ap.add_argument("--per-match", type=int, default=0, help="0: all moments")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--loop-budget", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--effort", default="low", help="Writer reasoning effort")
    args = ap.parse_args()
    if args.judge:
        judge_file(args)
        return
    if args.moments is None:
        raise SystemExit("--moments is required (or --judge)")

    matches = [json.loads(x) for x in open(args.moments)]
    if args.matches:
        matches = matches[: args.matches]
    if args.per_match:
        for m in matches:
            m["moments"] = m["moments"][: args.per_match]
    done = set()
    if args.out.exists():  # resumable: a match already written is skipped
        for x in open(args.out):
            r = json.loads(x)
            done.add((r["data"], r["ep"]))
    todo = [m for m in matches if (m["data"], m["ep"]) not in done]
    lock, n_ok, n_all = threading.Lock(), [0], [0]
    t0 = time.time()
    f = open(args.out, "a")

    def write(row):
        with lock:
            f.write(json.dumps(row) + "\n")
            f.flush()
            n_all[0] += 1
            n_ok[0] += row["ok"]
            if n_all[0] % 25 == 0:
                print(
                    f"{n_all[0]} lines, {100 * n_ok[0] / n_all[0]:.0f}% pass, "
                    f"{n_all[0] / (time.time() - t0):.2f}/s  last: {row['line']!r}",
                    flush=True,
                )

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(lambda m: run_match(m, args, write), todo))
    f.close()
    print(f"done: {n_all[0]} lines, {n_ok[0]} pass every check -> {args.out}")


if __name__ == "__main__":
    main()
