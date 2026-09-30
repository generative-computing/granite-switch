# SPDX-License-Identifier: Apache-2.0
"""Write the partner dataset: the player's replies and remarks, with Mellea IVR.

The player is a calm, dry professional out of a 1990s crime movie, and the
person watching is his partner, sitting next to him. At about half the speaking
moments of a recorded match (``talk.py moments``) the partner says something
first: lines written for that moment by the same model (backseat driving,
praise, teasing, worry, a question, small talk, a request to play differently,
or a reaction to his last reply), one of them kept and rendered the way speech
recognition hears it (sometimes with a misheard word). The player replies, and
the joke is the angle the reply takes on the partner's words: one of
:data:`MOVES`, shown with a model exchange drawn from :data:`EXAMPLES`. A
request is acknowledged and fended off; he keeps playing his way. At the other
moments he says a line of his own, often answering what a bot just did as if
it had said something to him.

The writer (Granite 4.2 30B, thinking low) sees what the trained narrator will
see: the moment's brief, the last log lines, and the exchanges still inside
the 10 s history window. Checks in code: length, no numbers, no status-report
opening, no stock phrase, calm punctuation, mild language, not a film line, not
a repeat. Checks judged by gpt-oss-120b: true to the moment, consistent with
what was said, in the voice, funny; for a reply, that it answers the partner,
and a swap test (shown the partner's line and the two other lines written for
that moment, the judge must pick the one the reply answers). A failed check's
reason goes back to the writer, which repairs its line (MultiTurnStrategy).

Runs in an environment with Mellea (``pip install mellea``)::

    python partner_ivr.py --moments data/narr/moments6.jsonl --out data/narr/partner.jsonl \\
        --base-url http://WRITER:PORT/v1 --judge-url http://JUDGE:PORT/v1
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from narrate_ivr import FILM, STRONG, clean, words

TIC_HZ = 35
MAX_ENTRIES = 50  # history.History: 10 s of 5 Hz entries, then the older half goes


def window_len(n: int) -> int:
    """How many of a conversation's first ``n`` entries the live history still
    holds (``history.History.append``: grow to MAX_ENTRIES, then keep half)."""
    if n <= MAX_ENTRIES:
        return n
    keep = MAX_ENTRIES // 2
    return keep + 1 + (n - MAX_ENTRIES - 1) % keep


PERSONA = (
    "You write the lines of a character in a Doom deathmatch against bots: a calm, "
    "dry professional out of a 1990s crime movie. His partner sits next to him "
    "watching the screen, and the two of them talk like partners on a long job: "
    "they bicker, needle each other and never get flustered. He is deadpan, "
    "unbothered and quick, and the humor is in how he takes what was just said. He "
    "never says numbers. Every line is original: never quote or paraphrase any "
    "film. Mild language at most."
)
# What the partner says: (weight, instruction to the partner's voice).
UTTERANCES = {
    "backseat": (3, "Tell him what to do right now, like a backseat driver."),
    "praise": (2, "React to something good he just did."),
    "tease": (2, "Tease him or trash-talk his play, the way a friend would."),
    "worry": (2, "Get nervous about what is about to happen to him."),
    "question": (3, "Ask him something about what is going on in the game."),
    "odd": (1, "Ask him an odd, idle question about the game world or the bots."),
    "smalltalk": (1, "Bring up something from outside the game."),
    "request": (
        2,
        "Ask him to change how he plays for a while: play safer, collect stuff, "
        "be more aggressive, stop hiding.",
    ),
}
FOLLOW_UP = (
    "React to what he just said to you: push back, laugh at it, needle him, or ask "
    "what he meant."
)
MOVES = {
    "echo": "Repeat one of their words and bend its meaning.",
    "correct": "Object to how they put it: quibble with their word choice, "
    "pedantic under fire.",
    "behind": "Answer the question behind what they said, not the one they asked.",
    "understate": "Understate it: treat what they said as much smaller than it is.",
    "tangent": "Drift to something ordinary in the middle of the fight: {topic}.",
    "pride": "Take it as a slight on your professional pride and defend your record.",
    "theory": "Offer a small theory nobody asked for.",
    "menace": "Stay polite, with a quiet menace toward the bots.",
    "bicker": "Bicker like an old partner who has heard it all before.",
    "misheard": "Speech recognition may have misheard a word: play off the odd "
    "word, then answer what they meant.",
    "callback": "Call back to something that was said earlier in the conversation.",
}
REQUEST_MOVES = ("echo", "correct", "behind", "understate", "pride", "theory", "bicker")
# For the tangent move: something ordinary, drawn at random (left to itself the
# writer drifted to coffee every time).
TANGENTS = (
    *("a parking ticket", "his dry cleaning", "a sandwich he left in the car"),
    *("the dentist", "his landlord", "a song stuck in his head", "the weather"),
    *("a cousin's wedding", "his car's check-engine light", "pie at a diner"),
    *("the neighbor's dog", "his taxes", "a bad haircut", "a crossword clue"),
    *("the laundromat", "the price of gas", "his mother's cooking", "a lost umbrella"),
)
# One exchange per move shown to the writer as a model of the move (a situation,
# what the partner said, the reply), drawn from three; ``Not a copy`` keeps the
# reply from reusing it.
EXAMPLES = {
    "echo": (
        (
            "his health is low and he was just hit",
            "you good",
            "Good is a big word. I'm upright. Let's start there.",
        ),
        (
            "he is waiting behind a wall",
            "you're hiding",
            "Hiding. I prefer waiting with intent.",
        ),
        (
            "bots all around him",
            "this is crazy",
            "Crazy is when you don't have a plan. This is just busy.",
        ),
    ),
    "correct": (
        (
            "he just fragged a bot",
            "you murdered that guy",
            "Murder? He walked into my work area. That's a safety violation.",
        ),
        (
            "he is backing away from a fight",
            "why are you running away",
            "Running is a strong word. I'm relocating with urgency.",
        ),
        (
            "he just fragged a bot",
            "you got lucky",
            "Luck is what amateurs call preparation they didn't see.",
        ),
    ),
    "behind": (
        (
            "he just got killed",
            "what happened",
            "What happened is a rocket had an opinion, and I didn't.",
        ),
        (
            "a quiet stretch",
            "are you even trying",
            "You want to know if I care. I care in private.",
        ),
        ("his health is low", "should we be worried", "You should be. I'm busy."),
    ),
    "understate": (
        (
            "several frags in a few seconds",
            "dude you're on fire",
            "I'm having a reasonable afternoon. Don't jinx it.",
        ),
        ("he took heavy damage", "that looked bad", "It was a firm handshake. Mostly."),
        (
            "two bots shooting at him",
            "they're everywhere",
            "It's a crowd. I've worked crowds.",
        ),
    ),
    "tangent": (
        (
            "two bots shooting at him",
            "what are you thinking about",
            "Honestly? Whether anybody here ever mops these floors.",
        ),
        (
            "his health is low",
            "you're gonna die",
            "Maybe. Did I leave the stove on this morning?",
        ),
        (
            "he just fragged a bot",
            "nice shot",
            "Thanks. Reminds me, the car's due for an oil change.",
        ),
    ),
    "pride": (
        (
            "he just got killed",
            "lol you suck",
            "I don't suck. I had a disagreement with physics, and physics cheats.",
        ),
        (
            "he keeps missing",
            "you missed",
            "I was establishing a pattern. The next one's a statement.",
        ),
        ("a bot just hit him", "he got you", "He got a piece of me. He'll be billed."),
    ),
    "theory": (
        (
            "a quiet stretch",
            "why do they keep coming back",
            "Same reason as everybody. Nobody ever tells them no.",
        ),
        (
            "a bot backing away",
            "why is he running",
            "Guilty conscience. They all have one. They just don't know it.",
        ),
        (
            "he just respawned",
            "how do you keep coming back",
            "Stubbornness. It's cheaper than armor.",
        ),
    ),
    "menace": (
        (
            "a bot in view",
            "there's one right there",
            "I see him. Give him a moment to make his mistake.",
        ),
        ("a bot coming at him", "he's coming for you", "Good. Saves me the walk."),
        (
            "a bot behind a pillar",
            "he's behind the pillar",
            "Let him. Pillars are temporary.",
        ),
    ),
    "bicker": (
        (
            "his partner shouting directions",
            "left left go left",
            "I heard you. The bot heard you. Everybody heard you.",
        ),
        (
            "a medikit on screen",
            "grab the health",
            "You say that like I've never seen a medikit.",
        ),
        (
            "he just got killed",
            "you should have used the rocket launcher",
            "And you should have brought snacks. We all have regrets.",
        ),
    ),
    "misheard": (
        (
            "a rocket launcher on the floor",
            "go get the rocket lunch",
            "Rocket lunch. I'll assume the launcher, not a sandwich.",
        ),
        (
            "his health is low",
            "you need some kelp",
            "Kelp. I'm fine on seaweed. Health, maybe.",
        ),
        (
            "a bot behind him",
            "there's a bot behind the whale",
            "Behind the whale. The wall, I hope. Either way, I'm turning.",
        ),
    ),
    "callback": (
        (
            "his partner told him to be careful, then he fragged a bot",
            "nice",
            "See? Careful. Exactly like you ordered.",
        ),
        (
            "he said he would take it slow, now he is charging",
            "i thought you were taking it slow",
            "I am. This is slow for me.",
        ),
        (
            "his partner said he sucked, now he fragged a bot",
            "ok that was good",
            "So I've gone from suck to good. That's progress.",
        ),
    ),
    "request": (
        (
            "a quiet stretch",
            "play it safe for a while",
            "Safe. Noted. I'll take it under advisement after this guy.",
        ),
        (
            "his armor is low",
            "go get some armor",
            "Armor's a lovely idea. I'll put it on my list, under later.",
        ),
        (
            "he is waiting behind a wall",
            "stop hiding and fight",
            "I heard you. I'm going to keep doing it my way, but I heard you.",
        ),
    ),
}
REMARKS = {
    "bots": "Answer what a bot just did as if it had said something rude to you.",
    "aside": "Make a dry aside to your partner about this moment.",
    "observe": "Make a dry observation about this exact moment.",
    "understate": "Understate what just happened.",
    "threat": "Say a quiet, polite threat to the bots.",
    "complain": "Complain about something, deadpan.",
}
REMARK_EXAMPLES = {
    "bots": (
        (
            "a bot hit him from behind",
            "Rude. I was clearly in the middle of something.",
        ),
        (
            "a bot backing off",
            "Leaving already? We were just getting to know each other.",
        ),
        ("he just fragged a bot", "You should have called first."),
    ),
    "aside": (
        ("a quiet stretch", "You notice they never clean up after themselves?"),
        ("he just fragged a bot", "Write that one down. I won't do it twice."),
    ),
    "observe": (
        ("a quiet stretch", "Every room in this place is the same room."),
        ("he just respawned", "Back again. The place hasn't missed me."),
    ),
    "understate": (
        ("he took heavy damage", "That stung a little."),
        ("several frags in a row", "Busy few seconds."),
    ),
    "threat": (
        ("a bot in view", "Take your time. I'm not going anywhere."),
        ("a quiet stretch", "Whoever's next, I've cleared my afternoon."),
    ),
    "complain": (
        ("he just got killed", "Nobody here has any manners."),
        ("a quiet stretch", "All this walking, and not one decent chair."),
    ),
}
BOT_EVENTS = ("you fragged", "you took", "you got killed", "in view, nearest")

CONTEXT = """What is happening right now:
{brief}
The last moments of the game log (most recent last):
{recent}
What was said in the last few seconds (oldest first):
{conv}"""
REPLY_TASK = """{persona}

{context}

Your partner just said to you: "{player}"
{how}
Write what you say back: 4 to 16 words, one to three short sentences. Output only \
the line."""
REMARK_TASK = """{persona}

{context}

Say one line out loud now. {how}
Say it in 3 to 10 words. Output only the line."""
UTTER_TASK = """You and your partner are on a long job together. He is playing a Doom \
deathmatch against bots and you sit next to him watching the screen. You talk to him \
the way you always do: casual, reactive, a little cheeky, with feeling. He is the one \
playing; you only watch, so never talk as if you were in the game yourself. You never \
read numbers or stats off the screen. {how}

What is happening right now:
{brief}
What was said in the last few seconds (oldest first):
{conv}

Write four different things you might say to him now, one per line, each 2 to 12 \
words of casual spoken English. They must differ in meaning. Output only the four \
lines."""
MISHEAR = """A speech recognizer heard this sentence and got exactly one word wrong: \
it swapped it for a similar-sounding real word, and the sentence came out a little \
funny. For example, "go get the rocket launcher" heard as "go get the rocket lunch".
"{x}"
Write the sentence as it was heard, with the same number of words. Output only the \
sentence."""

JUDGE_ALL = {
    "true": "Does the line avoid saying anything false about the game at this moment "
    "(kills, deaths, damage, pickups, weapons, health, ammo, enemies in view)? A line "
    "with no game facts at all is YES; jokes, opinions, plans and flavor details (a "
    "sticky floor, the coffee) are fine.",
    "consistent": "Is it consistent with what he said earlier (no contradicting his "
    "own earlier lines; a callback only to something actually said)? Disagreeing "
    "with his partner is fine.",
    "voice": "Does it sound like a calm, dry, deadpan professional from a 1990s crime "
    "movie talking to his partner (not a soldier, a sports announcer or a "
    "cheerleader)?",
    "funny": "Is it genuinely funny or sharp (a real turn of phrase or angle), not "
    "a generic quip?",
}
JUDGE_REPLY = {
    "answers": "Does it respond to what the partner actually said, playing off their "
    "words or their point?"
}
JUDGE_REQUEST = {
    "fends_off": "Does he acknowledge the partner's request but turn it down or put "
    "it off? Agreeing to do it is a NO."
}

NUMBER = re.compile(
    r"\d|\b(zero|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|dozen)\b",
    re.I,
)
# A status report's first word ("Health's low...", "BFG's warm..."): the habit of
# the first dataset.
STATUS = {
    *("health", "health's", "armor", "armor's", "ammo", "bfg", "bfg's", "pistol"),
    *("pistol's", "chaingun", "chaingun's", "shotgun", "shotgun's", "plasma"),
    *("rocket", "rockets", "another", "time", "situation"),
}
# Phrases the first dataset repeated hundreds of times.
STOCK = (
    "let's see",
    "about to learn",
    "feels like",
    "quiet now",
    "humming",
    "meters",
    "job well done",
    "just the way i like",
)


def heard(text: str) -> str:
    """As speech recognition writes it: lower case, no punctuation."""
    text = text.lower().replace("\u2019", "'")  # a curly apostrophe
    return " ".join(re.sub(r"[^\w\s']", " ", text).split())


def one_word_off(a: str, b: str) -> bool:
    """``b`` is ``a`` with exactly one word swapped (a mishearing)."""
    x, y = a.split(), b.split()
    return len(x) == len(y) and sum(p != q for p, q in zip(x, y)) == 1


def code_fns(kind: str, prev: list[str], player: str | None = None, shown=()):
    """Requirements checked in code, as (description, fn -> (ok, reason)).
    ``player``: the partner's words (a reply may open by echoing one);
    ``shown``: the example lines the writer was shown, not to be copied."""
    lo, hi = (4, 16) if kind == "reply" else (3, 10)
    echo = set(words(player or ""))

    def length(x):
        n = len(clean(x).split())
        return lo <= n <= hi, f"The line has {n} words; it must have {lo} to {hi}."

    def sentences(x):
        n = len([s for s in re.split(r"[.?!]+", clean(x)) if s.strip()])
        return n <= 3, "At most three short sentences."

    def numbers(x):
        m = NUMBER.search(clean(x))
        said = m.group(0) if m else ""
        return (
            m is None,
            f'No numbers in digits or words ("{said}"): he never says them.',
        )

    def opening(x):
        w = words(clean(x))
        first = w[0] if w else ""
        ok = bool(w) and (first not in STATUS or first.split("'")[0] in echo)
        return ok, (
            f'Do not open with "{first}": no status report; open with his angle on it.'
        )

    def not_copy(x):
        w = set(words(clean(x)))
        for e in shown:
            q = set(words(e))
            if w and q and len(w & q) / len(w | q) > 0.4:
                return False, "Too close to the example; write your own line."
        return True, ""

    def stock(x):
        hit = [s for s in STOCK if s in clean(x).lower()]
        said = hit[0] if hit else ""
        return (
            not hit,
            f'"{said}" is a stock phrase in these lines; put it another way.',
        )

    def calm(x):
        c = clean(x)
        dashes = "\u2014\u2013"  # em and en dash
        dash = any(d in c for d in dashes) and any(
            d in p for p in prev[-2:] for d in dashes
        )
        return "!" not in c and not dash, (
            "Calm punctuation: no exclamation marks, and no dash (a recent line had "
            "one); use a period or a comma."
        )

    def plain(x):
        ok = not re.search(r"[\[\]*#\"]|reload", clean(x), re.I)
        return ok, "No brackets, asterisks or quote marks, and Doom has no reloading."

    def mild(x):
        bad = [s for s in STRONG if s in clean(x).lower()]
        return not bad, "Keep the language mild: no strong swearing or slurs."

    def original(x):
        hit = [s for s in FILM if s in clean(x).lower()]
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

    fns = [
        (f"{lo} to {hi} words", length),
        ("No numbers", numbers),
        ("No status-report opening", opening),
        ("No stock phrase", stock),
        ("Calm punctuation", calm),
        ("Plain spoken text", plain),
        ("Mild language", mild),
        ("Original, not a film quote", original),
        ("Not a repeat of recent lines", fresh),
        ("Not a copy of the example", not_copy),
    ]
    if kind == "reply":
        fns.insert(1, ("At most three sentences", sentences))
    return fns


def judge_opts(args) -> dict:
    from mellea.backends import ModelOption

    return {
        ModelOption.THINKING: args.judge_effort,
        ModelOption.TEMPERATURE: 0.0,
        ModelOption.MAX_NEW_TOKENS: 2000,  # reasoning included
    }


def judge_fn(judge, context: str, questions: dict, args):
    """The judged requirements in one call: the line passes only if every
    verdict is YES; the verdicts are the repair feedback."""
    listing = "\n".join(f"{k}: {q}" for k, q in questions.items())

    def fn(x):
        q = (
            f"Judge one spoken line from a Doom deathmatch.\n\n{context}\n\n"
            f'The line he says: "{clean(x)}"\n\nQuestions:\n{listing}\n\n'
            f"Answer with exactly {len(questions)} lines, one per question, each as "
            "`name: YES` or `name: NO - short reason`."
        )
        judge.reset()
        a = str(judge.instruct(q, strategy=None, model_options=judge_opts(args)))
        verdicts = {k: re.search(rf"{k}\W*?:\W*(YES|NO)", a, re.I) for k in questions}
        bad = [k for k, v in verdicts.items() if not v or v.group(1).upper() != "YES"]
        return not bad, (a.strip() if bad else "")

    return fn


def swap_fn(judge, player: str, others: list[str], rng: random.Random, args):
    """The swap test: shown the partner's line among the two others written for
    the same moment, the judge must pick the one the reply answers."""
    opts = [player, *others]
    rng.shuffle(opts)
    letters = "ABC"[: len(opts)]

    def fn(x):
        q = (
            "In a Doom deathmatch, a player said this to his partner, in reply to "
            f'something the partner had just said:\n"{clean(x)}"\n\nWhich of these had '
            "the partner just said?\n"
            + "\n".join(f"{c}) {o}" for c, o in zip(letters, opts))
            + "\n\nAnswer with one letter."
        )
        judge.reset()
        a = str(judge.instruct(q, strategy=None, model_options=judge_opts(args)))
        m = re.search(r"\b([ABC])\b", a.strip())
        ok = bool(m) and opts[letters.index(m.group(1))] == player
        return ok, (
            "The reply could follow many things the partner might say; make it turn "
            f'on what they actually said: "{player}".'
        )

    return fn


def render_conv(turns: list[dict], now: int) -> str:
    out = []
    for tr in turns:
        when = f"{max(0, round((now - tr['t']) / TIC_HZ))} s ago"
        if tr.get("player"):
            out.append(f'{when}, your partner: "{tr["player"]}"')
        out.append(f'{when}, you: "{tr["line"]}"')
    return "\n".join(out) or "(nothing yet)"


def partner_lines(voice, how: str, brief: str, conv: str) -> list[str]:
    """Three things the partner might say at this moment."""
    from mellea.backends import ModelOption

    a = str(
        voice.instruct(
            UTTER_TASK.format(how=how, brief=brief, conv=conv),
            strategy=None,
            model_options={
                ModelOption.THINKING: False,
                ModelOption.TEMPERATURE: 1.0,
                ModelOption.MAX_NEW_TOKENS: 160,
            },
        )
    )
    out = []
    for x in clean_lines(a):
        # Numbers read off the screen made the partner sound like a HUD.
        if (
            2 <= len(x.split()) <= 14
            and not NUMBER.search(x)
            and heard(x) not in map(heard, out)
        ):
            out.append(x)
    return out[:3]


def clean_lines(text: str) -> list[str]:
    text = re.sub(r"(?s)<think>.*?</think>", "", str(text)).split("</think>")[-1]
    out = []
    for x in text.splitlines():
        x = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", x).strip().strip('"“”').strip()
        if x:
            out.append(x)
    return out


def run_match(match: dict, args, write) -> int:
    from mellea import start_session
    from mellea.backends import ModelOption
    from mellea.stdlib.context import ChatContext
    from mellea.stdlib.requirements import req, simple_validate
    from mellea.stdlib.sampling import MultiTurnStrategy

    key = (match["data"], match["ep"])
    rng = random.Random(zlib.crc32(repr((*key, args.seed)).encode()))
    wurls, jurls = args.base_url.split(","), args.judge_url.split(",")
    h = zlib.crc32(repr(key).encode())
    wkw = {"base_url": wurls[h % len(wurls)], "api_key": "none"}
    jkw = {"base_url": jurls[h % len(jurls)], "api_key": "none"}
    writer = start_session(
        "openai",
        model_id=args.model,
        ctx=ChatContext(),
        model_options={
            # Granite 4.2 at "low" effort reasons briefly and closes </think>.
            ModelOption.THINKING: args.effort,
            ModelOption.TEMPERATURE: args.temperature,
            ModelOption.MAX_NEW_TOKENS: args.max_tokens,
        },
        **wkw,
    )
    vurls = (args.voice_url or args.base_url).split(",")
    voice = start_session(  # the partner
        "openai",
        model_id=args.voice_model or args.model,
        base_url=vurls[h % len(vurls)],
        api_key="none",
    )
    judge = start_session("openai", model_id=args.judge_model, **jkw)
    names = list(UTTERANCES)
    weights = [UTTERANCES[k][0] for k in names]
    turns: list[dict] = []  # what was said: conversation position, tick, words
    prev: list[str] = []
    for m in match["moments"]:
        n = m["hist_n"] + len(turns)  # conversation entries before this turn
        win = [tr for tr in turns if tr["pos"] >= n - window_len(n)]
        conv = render_conv(win, m["t"])
        kind, utype, player, said, others, moves = "remark", None, None, None, [], []
        last = win[-1] if win else None
        if last and last.get("player") and rng.random() < args.follow_rate:
            utype = "followup"
        elif rng.random() < args.reply_rate:
            utype = rng.choices(names, weights)[0]
        if utype:
            how = FOLLOW_UP if utype == "followup" else UTTERANCES[utype][1]
            cands = partner_lines(voice, how, m["brief"], conv)
            if len(cands) == 3:
                kind = "reply"
                i = rng.randrange(3)
                said = cands[i]
                player = heard(said)
                others = [heard(c) for j, c in enumerate(cands) if j != i]
                if rng.random() < args.misheard_rate:
                    mis = clean_lines(
                        voice.instruct(
                            MISHEAR.format(x=player),
                            strategy=None,
                            model_options={
                                ModelOption.THINKING: False,
                                ModelOption.TEMPERATURE: 0.8,
                                ModelOption.MAX_NEW_TOKENS: 60,
                            },
                        )
                    )
                    if mis and one_word_off(player, heard(mis[0])):
                        player, moves = heard(mis[0]), ["misheard"]
            else:
                utype = None
        context = CONTEXT.format(
            brief=m["brief"], recent="\n".join(m["recent"]), conv=conv
        )
        if kind == "reply":
            if not moves:
                pool = (
                    REQUEST_MOVES
                    if utype == "request"
                    else [
                        k for k in MOVES if k != "misheard" and (k != "callback" or win)
                    ]
                )
                moves = [rng.choice(list(pool))]
            desc = MOVES[moves[0]].format(topic=rng.choice(TANGENTS))
            sit, ex_said, ex_line = rng.choice(
                EXAMPLES["request" if utype == "request" else moves[0]]
            )
            how = f"Move: {desc}"
            if utype == "request":
                how = (
                    "It is a request to change how you play. Acknowledge it, then fend "
                    f"it off: you keep playing your way. {how}"
                )
            how += (
                f' For example, when {sit} and the partner said "{ex_said}", he '
                f'said: "{ex_line}" Write your own line; do not reuse that one.'
            )
            task = REPLY_TASK.format(
                persona=PERSONA, context=context, player=player, how=how
            )
            questions = {**JUDGE_ALL, **JUDGE_REPLY}
            if utype == "request":
                questions.update(JUDGE_REQUEST)
            jctx = f'{context}\nHis partner just said: "{player}"'
        else:
            modes = [
                k
                for k in REMARKS
                if k != "bots" or any(e in m["brief"] for e in BOT_EVENTS)
            ]
            moves = [rng.choice(modes)]
            sit, ex_line = rng.choice(REMARK_EXAMPLES[moves[0]])
            how = (
                f'{REMARKS[moves[0]]} For example, when {sit}: "{ex_line}" Write your '
                "own line; do not reuse that one."
            )
            task = REMARK_TASK.format(persona=PERSONA, context=context, how=how)
            questions = dict(JUDGE_ALL)
            jctx = f"{context}\nHis partner said nothing."
        reqs = [
            req(d, validation_fn=simple_validate(f))
            for d, f in code_fns(kind, prev, player, (ex_line,))
        ]
        reqs.append(
            req(
                "Judged: " + ", ".join(questions),
                validation_fn=simple_validate(judge_fn(judge, jctx, questions, args)),
            )
        )
        if kind == "reply":
            reqs.append(
                req(
                    "Answers the partner's words (swap test)",
                    validation_fn=simple_validate(
                        swap_fn(judge, player, others, rng, args)
                    ),
                )
            )
        t0 = time.time()
        writer.reset()
        res = writer.instruct(
            task,
            requirements=reqs,
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
                "kind": kind,
                "utype": utype,
                "player": player,
                "said": said,
                "others": others,
                "moves": moves,
                "line": line,
                "ok": bool(getattr(res, "success", not fails)),
                "attempts": len(getattr(res, "sample_generations", None) or []) or 1,
                "fails": fails,
                "window": [[tr.get("player"), tr["line"]] for tr in win],
                "brief": m["brief"],
                "recent": m["recent"],
                "prev": prev[-5:],
                "s": round(time.time() - t0, 1),
            }
        )
        if line:  # what he said, pass or not: the next moment follows it
            prev.append(line)
            turns.append({"pos": n, "t": m["t"], "player": player, "line": line})
    return len(match["moments"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--base-url", required=True, help="Writer server(s), comma-separated"
    )
    ap.add_argument(
        "--judge-url", required=True, help="Judge server(s), comma-separated"
    )
    ap.add_argument("--model", default="granite-4.2-30b")
    ap.add_argument("--judge-model", default="gpt-oss-120b")
    ap.add_argument(
        "--shard",
        default="0/1",
        help="K/N: every N-th match from the K-th (e.g. one writer per shard)",
    )
    ap.add_argument("--voice-model", help="The partner's words (default: --model)")
    ap.add_argument("--voice-url", help="Its server(s) (default: --base-url)")
    ap.add_argument("--judge-effort", default="low", help="Judge reasoning effort")
    ap.add_argument("--matches", type=int, default=0, help="0: all")
    ap.add_argument("--per-match", type=int, default=0, help="0: all moments")
    ap.add_argument(
        "--reply-rate", type=float, default=0.5, help="Moments the partner speaks at"
    )
    ap.add_argument(
        "--follow-rate", type=float, default=0.3, help="Follow-ups to a reply"
    )
    ap.add_argument("--misheard-rate", type=float, default=0.12)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--loop-budget", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--effort", default="low", help="Writer reasoning effort")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    matches = [json.loads(x) for x in open(args.moments)]
    if args.matches:
        matches = matches[: args.matches]
    k, n = map(int, args.shard.split("/"))
    matches = matches[k::n]
    if args.per_match:
        for m in matches:
            m["moments"] = m["moments"][: args.per_match]
    done = set()
    if args.out.exists():
        # Resumable: a whole match already written is skipped; one cut short (a
        # server went away) is dropped and written again from its start.
        want = {(m["data"], m["ep"]): len(m["moments"]) for m in matches}
        rows = [json.loads(x) for x in open(args.out)]
        count = Counter((r["data"], r["ep"]) for r in rows)
        done = {k for k, c in count.items() if c >= want.get(k, 0)}
        kept = [r for r in rows if (r["data"], r["ep"]) in done]
        if len(kept) < len(rows):
            with open(args.out, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in kept)
            print(f"dropped {len(rows) - len(kept)} rows of unfinished matches")
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

    failed = []

    def one(m):
        try:
            return run_match(m, args, write)
        except Exception as e:  # a server went away: the others carry on
            failed.append((m["data"], m["ep"]))
            print(f"match {m['ep']} failed: {type(e).__name__}: {e}", flush=True)
            return 0

    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(one, todo))
    f.close()
    print(f"done: {n_all[0]} lines, {n_ok[0]} pass every check -> {args.out}")
    if failed:
        raise SystemExit(f"{len(failed)} matches failed; run again to write them")


if __name__ == "__main__":
    main()
