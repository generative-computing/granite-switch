# SPDX-License-Identifier: Apache-2.0
"""Policies for the Doom demo, plus direct prompt-id assembly.

Every policy answers ``decide(obs, adapter, history) -> Decision``,
``decide_many(obs, adapters, history)`` (several adapters on one state, one
engine step) and ``route(instruction) -> Route``, so rollouts, the latency bench
and the live server can use any of them:

* :class:`ExpertPolicy`: the scripted player, its weapon planner, a rule-based
  danger estimate and a keyword router. Needs no model, so the whole demo runs
  on a laptop.
* :class:`VLLMPolicy`: the composed Granite Switch checkpoint served in-process
  by vLLM. One engine step per tic prefills the fresh tokens and emits one
  token per adapter asked.

The game prompt, identical for every game adapter up to its query suffix::

    <|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>
    <|start_of_role|>user<|end_of_role|>{history}{pad}now t14.2 | {state}{pad}<|end_of_text|>
    <|adapter|>assistant<|end_of_role|>            -> one output token

``{history}`` is :class:`history.History`: append-only 5 Hz entries, so the
engine's prefix cache holds it from one tic to the next. ``{pad}`` is newlines
up to the next KV block boundary (``BLOCK`` tokens). The first pad changes only
when the history grows (every 0.2 s); the second makes the state end on a block
boundary. Together they leave only the adapter suffix as per-adapter work when
several adapters read the same state in one step.

A checkpoint composed from **LoRA** adapters (the baseline that aLoRA is
compared against) puts the control token at position 0 instead, in place of the
first ``<|start_of_role|>``, as the composed chat template does. Every adapter
then has its own KV from the first token on, and nothing is shared. A **Shadow
Residual** checkpoint puts it last: it replaces the final ``<|end_of_role|>``,
so the adapter stream runs on that one position and reads only base K/V.

Prompt ids are assembled directly each tick, which skips Jinja rendering.
``python policy.py --check-template <model_dir>`` confirms the ids are identical
to ``apply_chat_template(adapter_name=...)`` (without the block padding, which
has no chat-template form).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field

from doom_env import ACTIONS, WEAPON_SLOTS, Observation
from expert import BEHAVIORS, Expert
from history import History, now_prefix

ARMS = "arms"  # weapon planner: one slot digit
CRITIC = "critic"  # danger of taking damage or dying within 1 s
ROUTER = "router"
GAME_ADAPTERS: tuple[str, ...] = (*BEHAVIORS, ARMS, CRITIC)
ADAPTERS: tuple[str, ...] = (*GAME_ADAPTERS, ROUTER)
WEAPON_TOKENS: tuple[str, ...] = tuple(str(s) for s in WEAPON_SLOTS)
DANGER_LEVELS: tuple[str, ...] = ("low", "mid", "high")
OUTPUTS: dict[str, tuple[str, ...]] = {
    **{b: ACTIONS for b in BEHAVIORS},
    ARMS: WEAPON_TOKENS,
    CRITIC: DANGER_LEVELS,
    ROUTER: BEHAVIORS,
}

SYSTEM_PROMPT = (
    "You play Doom deathmatch against bots. First comes a history of the last 10 "
    "seconds, one line per 0.2 s: match time, health, heading in degrees, the "
    "enemies in view, your main action, and events. Then the current state: "
    "health, armor, the selected weapon and its ammo, owned weapon slots with ammo; "
    "objects on screen as name, bearing in degrees (negative is left) and distance; "
    "wall clearance left, front, right and behind in metres; damage taken in the "
    "last second; where an enemy was last seen; your last two actions. Reply with "
    "what is asked: one action ("
    + " ".join(ACTIONS)
    + "), a weapon slot (1-7), or the danger of being hit soon (low mid high)."
)
# The base model's spoken line: a user turn after the shared game context, so a
# talk request reuses the prefix every game adapter has just prefilled.
TALK_INSTRUCTION = (
    "You are this player: a cocky Doom marine talking to yourself as you play. Say "
    "one short line out loud, at most 12 words. React with feeling to what just "
    "happened, or tell yourself what to do next and why. Doom has no reloading. "
    'Examples: "Got him!" "Ouch, I need a medikit." "Dammit, low on ammo, let\'s '
    'grab some." "Two on the left, BFG time." Do not repeat your last line.'
)
ROUTER_SYSTEM_PROMPT = (
    "Pick the Doom deathmatch play style that best follows the player's instruction: "
    "fighter (hunt and frag the bots), cautious (avoid damage, fight only up close, "
    "heal), collector (collect items, armor and weapons). Reply with the style name."
)

_SOR, _EOR, _EOT = "<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"
BLOCK = 16  # vLLM's default KV block size, in tokens
PAD = "\n"

# CUDA-graph capture sizes, in scheduled tokens per engine step. A tic prefills
# the fresh state (~90 tokens) plus a suffix per adapter; a history window
# reset re-prefills ~700 tokens, and the LoRA baseline re-prefills the whole
# ~1.6k-token prompt when it switches adapter. vLLM otherwise caps capture at
# 2 x max_num_seqs, and a larger step runs eagerly and launch-bound: 6 ms -> 17 ms
# on an H100 for one 45-token decision.
CAPTURE_SIZES = [
    *(1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256),
    *(320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048),
]


@dataclass
class Decision:
    action: str  # the output word: an action, a slot digit or a danger level
    top3: list[tuple[str, float]]
    ms: float  # state available -> output available, wall clock
    fresh_tokens: int = 0
    cached_tokens: int = 0
    build_ms: float = 0.0  # prompt-id assembly (tokenizing the state)
    engine_ms: float = 0.0  # vLLM: submit -> output (prefill + one token)
    probs: dict[str, float] = field(default_factory=dict)  # the whole vocabulary


@dataclass
class Route:
    adapter: str
    prob: float
    ms: float
    probs: dict[str, float] = field(default_factory=dict)


# ── Prompt-id assembly ─────────────────────────────────────────────────────────
def control_token(adapter: str) -> str:
    return f"<|{adapter}|>"


def single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is {len(ids)} tokens {ids}, expected exactly 1")
    return ids[0]


def output_token_ids(tokenizer, words: tuple[str, ...]) -> dict[str, int]:
    """Map each output word to its single token id; fail loudly if any is not one
    token or two share one."""
    ids = {w: single_token_id(tokenizer, w) for w in words}
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"output tokens collide: {ids}")
    return ids


def action_token_ids(tokenizer) -> dict[str, int]:
    return output_token_ids(tokenizer, ACTIONS)


def weapon_token_ids(tokenizer) -> dict[str, int]:
    return output_token_ids(tokenizer, WEAPON_TOKENS)


def danger_token_ids(tokenizer) -> dict[str, int]:
    return output_token_ids(tokenizer, DANGER_LEVELS)


def route_token_ids(tokenizer) -> dict[str, int]:
    """Map each behavior to the first token of its name (the router's one output token)."""
    ids = {b: tokenizer.encode(b, add_special_tokens=False)[0] for b in BEHAVIORS}
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"behavior first tokens collide: {ids}")
    return ids


def vocab_ids(tokenizer, adapter: str) -> dict[str, int]:
    """Output word -> token id for an adapter's one output token."""
    if adapter == ROUTER:
        return route_token_ids(tokenizer)
    return output_token_ids(tokenizer, OUTPUTS[adapter])


def state_text(obs: Observation) -> str:
    """The current-state part of the game prompt: match time, then the state line."""
    return now_prefix(obs.tick) + obs.text


class PromptBuilder:
    """Pre-tokenized prompt pieces, joined with the fresh text each tic.

    ``adapter=None`` gives the base-model prompt (what PEFT trains on). The
    control token's place follows the composed chat template (``placement``):

    * ``alora``: it takes the place of the ``<|start_of_role|>`` that opens the
      assistant header (an aLoRA whose invocation sequence is
      ``<|start_of_role|>assistant<|end_of_role|>``);
    * ``lora``: it takes the place of the first ``<|start_of_role|>``, at
      position 0;
    * ``sr``: it takes the place of the final ``<|end_of_role|>`` (Shadow
      Residual's ``last_context_token``).

    Either way the switch gives the control token the displaced token's
    embedding at runtime, so the model sees the sequence it was trained on.

    ``align=False`` drops the block padding (for the chat-template check).
    """

    def __init__(
        self,
        tokenizer,
        system: str,
        adapters: tuple[str, ...] = (),
        *,
        placement: str = "alora",
        align: bool = True,
        block: int = BLOCK,
    ):
        self.tok = tokenizer
        self.placement = placement
        self.lora = placement == "lora"
        self.block = block if align else 0
        self.pad_id = single_token_id(tokenizer, PAD)
        body = self._enc(f"system{_EOR}{system}{_EOT}\n{_SOR}user{_EOR}")
        base_suffix = self._enc(f"{_EOT}\n{_SOR}assistant{_EOR}")
        self.head = {None: self._enc(_SOR) + body}
        self.suffix = {None: base_suffix}
        for a in adapters:
            ctl = self._enc(control_token(a))
            if placement == "lora":
                self.head[a] = ctl + body
                self.suffix[a] = base_suffix
            elif placement == "sr":
                self.head[a] = self.head[None]
                self.suffix[a] = self._enc(f"{_EOT}\n{_SOR}assistant") + ctl
            else:
                self.head[a] = self.head[None]
                self.suffix[a] = self._enc(f"{_EOT}\n{control_token(a)}assistant{_EOR}")
        self.talk_suffix = self._enc(
            f"{_EOT}\n{_SOR}user{_EOR}{TALK_INSTRUCTION}{_EOT}\n{_SOR}assistant{_EOR}"
        )
        self.system = system

    def _enc(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def _pad(self, n: int) -> list[int]:
        return [self.pad_id] * (-n % self.block) if self.block else []

    def ids(self, text: str, adapter: str | None) -> list[int]:
        """A prompt without history (the router's)."""
        return self.head[adapter] + self._enc(text) + self.suffix[adapter]

    def game_ids(
        self, history_ids: list[int], state: str, adapters: list[str | None]
    ) -> list[list[int]]:
        """One game prompt per adapter, sharing everything before the suffix."""
        body = self._body(history_ids, state)
        return [self.head[a] + body + self.suffix[a] for a in adapters]

    def talk_ids(
        self, history_ids: list[int], state: str, brief: str = ""
    ) -> list[int]:
        """The base model's talk prompt: the base game prompt up to its suffix,
        then a user turn asking for one spoken line, after ``brief`` (what just
        happened, in plain words: :func:`talk.brief`)."""
        suffix = self.talk_suffix
        if brief:
            suffix = self._enc(
                f"{_EOT}\n{_SOR}user{_EOR}{brief}\n{TALK_INSTRUCTION}{_EOT}\n"
                f"{_SOR}assistant{_EOR}"
            )
        return self.head[None] + self._body(history_ids, state) + suffix

    def _body(self, history_ids: list[int], state: str) -> list[int]:
        n_head = len(self.head[None])  # every head has the same length
        pre = history_ids + self._pad(n_head + len(history_ids))
        body = pre + self._enc(state)
        return body + self._pad(n_head + len(body))

    def n_fixed(self) -> int:
        return len(self.head[None])


def alora_invocation_ids(tokenizer) -> list[int]:
    return tokenizer.encode(f"{_SOR}assistant{_EOR}", add_special_tokens=False)


def is_dual_stream(model_dir: str) -> bool:
    """Whether a checkpoint is a Shadow Residual (dual-stream) composition."""
    cfg = os.path.join(model_dir, "config.json")
    return os.path.exists(cfg) and bool(json.load(open(cfg)).get("dual_stream"))


def adapter_placement(tokenizer) -> str:
    """Where the composed chat template puts control tokens: ``alora`` (before
    the assistant header), ``lora`` (position 0), ``sr`` (last token), or
    ``base`` for a tokenizer without them."""
    ctl = control_token(BEHAVIORS[0])
    if ctl not in tokenizer.get_vocab():
        return "base"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        add_generation_prompt=True,
        tokenize=False,
        adapter_name=BEHAVIORS[0],
    )
    if rendered.startswith(ctl):
        return "lora"
    if rendered.endswith(ctl):
        return "sr"
    return "alora"


def check_template(model_dir: str, games: list[tuple[str, str]], routes: list[str]):
    """Assert direct id assembly == ``apply_chat_template`` for every adapter.

    ``games`` holds (history text, state text) pairs from real play.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    composed = control_token(BEHAVIORS[0]) in tok.get_vocab()
    placement = adapter_placement(tok) if composed else "alora"
    game_names = (*GAME_ADAPTERS, None) if composed else (None,)
    route_names = (ROUTER, None) if composed else (None,)
    adapters = ADAPTERS if composed else ()
    pb = PromptBuilder(tok, SYSTEM_PROMPT, adapters, placement=placement, align=False)
    rb = PromptBuilder(
        tok, ROUTER_SYSTEM_PROMPT, adapters, placement=placement, align=False
    )

    def rendered_ids(system: str, user: str, a: str | None) -> tuple[list[int], str]:
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kw = {"adapter_name": a} if a else {}
        r = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, **kw
        )
        return tok(r, add_special_tokens=False).input_ids, r

    for hist_text, state in games:
        hist_ids = tok.encode(hist_text, add_special_tokens=False)
        got = pb.game_ids(hist_ids, state, list(game_names))
        for a, g in zip(game_names, got):
            ref, r = rendered_ids(SYSTEM_PROMPT, hist_text + state, a)
            assert g == ref, f"adapter={a}\n got={g}\n ref={ref}\n{r!r}"
    for text in routes:
        for a in route_names:
            ref, r = rendered_ids(ROUTER_SYSTEM_PROMPT, text, a)
            assert rb.ids(text, a) == ref, f"adapter={a}\n{r!r}"
    check_output_tokens(tok)
    kind = {"lora": "LoRA", "sr": "Shadow Residual", "alora": "aLoRA"}[placement]
    what = f"{kind} adapters {', '.join(ADAPTERS)} and base" if composed else "base"
    print(
        f"OK: prompt ids match apply_chat_template for {what} on {len(games)} "
        f"history+state prompts and {len(routes)} instructions."
    )


def check_output_tokens(tok) -> None:
    """Every output vocabulary is single tokens, distinct within its adapter, and
    unchanged when it directly follows ``<|end_of_role|>`` (where it is emitted)."""
    groups = {
        "actions": action_token_ids(tok),
        "weapon slots": weapon_token_ids(tok),
        "danger levels": danger_token_ids(tok),
        "route first tokens": route_token_ids(tok),
    }
    eor = tok.encode(_EOR, add_special_tokens=False)
    for name, ids in groups.items():
        if name == "route first tokens":
            continue
        for w, i in ids.items():
            got = tok.encode(_EOR + w, add_special_tokens=False)
            if got != [*eor, i]:
                raise ValueError(f"{w!r} after {_EOR} tokenizes as {got}")
    for name, ids in groups.items():
        print(f"  {name}: {len(ids)} single, distinct tokens")


# ── Keyword router (stand-in when no model is loaded) ───────────────────────────
_KEYWORDS = {
    "fighter": r"kill|hunt|fight|attack|shoot|aggress|destroy|frag|slay|murder|rampage|clear",
    "cautious": r"surviv|safe|careful|avoid|run away|flee|hide|retreat|heal|health|defen|cautious|stay alive|don.t die",
    "collector": r"collect|loot|pick|gather|item|ammo|armor|scaveng|grab|supplies|weapon",
}


_NEGATION = r"(?:stop|don.?t|do not|no|never|quit|avoid|without)\s+(?:\w+\s+){0,2}?"


def keyword_route(instruction: str) -> Route:
    t0 = time.perf_counter()
    text = instruction.lower()
    # "stop fighting" is a vote for cautious, not fighter.
    negated_fights = len(
        re.findall(_NEGATION + "(?:" + _KEYWORDS["fighter"] + ")", text)
    )
    text = re.sub(_NEGATION + "(?:" + _KEYWORDS["fighter"] + ")\\w*", " ", text)
    scores = {b: len(re.findall(p, text)) for b, p in _KEYWORDS.items()}
    scores["cautious"] += negated_fights
    total = sum(scores.values())
    if total == 0:
        probs = {b: 1 / len(BEHAVIORS) for b in BEHAVIORS}
    else:
        probs = {b: s / total for b, s in scores.items()}
    best = max(probs, key=lambda b: (probs[b], b == "fighter"))
    return Route(best, probs[best], (time.perf_counter() - t0) * 1000, probs)


def rule_danger(obs: Observation) -> str:
    """A rule-based stand-in for the critic adapter: recent damage or a close bot."""
    foes = [o for o in obs.seen if o.kind == "enemy" or o.kind == "missile"]
    if obs.hit >= 20 or (obs.hit > 0 and obs.hp < 40):
        return "high"
    if obs.hit > 0 or any(o.dist < 12 for o in foes):
        return "mid"
    return "low"


# ── Policies ───────────────────────────────────────────────────────────────────
class ExpertPolicy:
    """The scripted player behind the same interface as the model."""

    name = "expert"

    def __init__(self) -> None:
        self.expert = Expert()

    def reset(self) -> None:
        self.expert.reset()

    def decide(
        self, obs: Observation, adapter: str, history: History | None = None
    ) -> Decision:
        t0 = time.perf_counter()
        if adapter == ARMS:
            out = str(self.expert.weapon(obs))
        elif adapter == CRITIC:
            out = rule_danger(obs)
        else:
            out = self.expert.act(obs, adapter)
        ms = (time.perf_counter() - t0) * 1000
        return Decision(out, [(out, 1.0)], ms, probs={out: 1.0})

    def decide_many(
        self, obs: Observation, adapters: tuple[str, ...], history=None
    ) -> dict[str, Decision]:
        return {a: self.decide(obs, a) for a in adapters}

    def route(self, instruction: str) -> Route:
        return keyword_route(instruction)


class VLLMPolicy:
    """Composed Granite Switch checkpoint served in-process by vLLM.

    Args:
        model: Composed checkpoint directory. Whether its adapters are aLoRA or
            LoRA is read from its chat template.
        engine_loop: Drive ``LLMEngine.add_request/step`` directly instead of
            ``LLM.generate``.
        prefix_caching: Keep on: the system prompt and the history are then
            prefilled once.
        align: Pad history and state to KV block boundaries (see module doc).
        logprobs_mode: ``processed_logprobs`` so the probabilities are over the
            allowed outputs, renormalized.
        cudagraph_mode: ``FULL`` captures the whole forward, SWITCH kernels
            included, for prefill steps too; ``FULL_AND_PIECEWISE`` is vLLM's
            default.
        gc_freeze: After warmup, move every live object to the permanent
            generation so a full GC pass does not walk vLLM's object heap in the
            middle of a decision.
        async_scheduling: Passed to vLLM when set; ``None`` keeps its default.
        base_model: ``model`` is a plain base checkpoint without control tokens;
            every adapter then maps to the base prompt (the SWITCH-overhead
            baseline).
        temperature: Sampling temperature for the style adapters and the
            weapon planner (0 = greedy). Students of a stochastic RL teacher
            play better sampling, as the teacher does; the critic and router
            stay greedy.
        attention: vLLM ``attention_config``. ``None`` picks FlashAttention 2
            for a Shadow Residual checkpoint: its attention is one layer with
            twice the model's query heads, and FlashAttention 3's
            ahead-of-time schedule is sized from the model's head count, so
            under vLLM 0.19.1 it raises in eager mode and returns wrong
            outputs under CUDA graphs.
    """

    name = "vllm"

    def __init__(
        self,
        model: str,
        *,
        engine_loop: bool = False,
        prefix_caching: bool = True,
        align: bool = True,
        max_num_seqs: int = 16,
        gpu_memory_utilization: float = 0.5,
        max_model_len: int = 4096,  # 10 s of eventful history can pass 2k tokens
        logprobs_mode: str = "processed_logprobs",
        enforce_eager: bool = False,
        warmup: int = 20,
        base_model: bool = False,
        cudagraph_mode: str = "FULL",
        gc_freeze: bool = True,
        async_scheduling: bool | None = None,
        log_stats: bool = False,
        temperature: float = 0.0,
        attention: dict | None = None,
    ):
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM, SamplingParams

        if attention is None and is_dual_stream(model):
            attention = {"flash_attn_version": 2}
        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=max_model_len,
            enable_prefix_caching=prefix_caching,
            max_num_seqs=max_num_seqs,
            # Every step fits a captured graph; larger prefills (the LoRA
            # baseline re-prefilling whole prompts) are chunked across steps.
            # Eager steps above the largest capture size also fail in vLLM
            # 0.19.1 ("scheduler_metadata must have shape (metadata_size)").
            max_num_batched_tokens=max(CAPTURE_SIZES),
            gpu_memory_utilization=gpu_memory_utilization,
            max_logprobs=len(ACTIONS),
            logprobs_mode=logprobs_mode,
            enforce_eager=enforce_eager,
            disable_log_stats=not log_stats,
            compilation_config={
                "cudagraph_capture_sizes": CAPTURE_SIZES,
                "cudagraph_mode": cudagraph_mode,
            },
            **(
                {}
                if async_scheduling is None
                else {"async_scheduling": async_scheduling}
            ),
            **({} if attention is None else {"attention_config": attention}),
        )
        self.tok = self.llm.get_tokenizer()
        self.base_model = base_model
        self.placement = "base" if base_model else adapter_placement(self.tok)
        self.lora = self.placement == "lora"
        adapters = () if base_model else ADAPTERS
        place = "alora" if base_model else self.placement
        self.pb = PromptBuilder(
            self.tok, SYSTEM_PROMPT, adapters, placement=place, align=align
        )
        self.rb = PromptBuilder(
            self.tok, ROUTER_SYSTEM_PROMPT, adapters, placement=place, align=False
        )
        self.vocab = {a: vocab_ids(self.tok, a) for a in ADAPTERS}
        self.words = {a: {i: w for w, i in v.items()} for a, v in self.vocab.items()}
        self.temperature = temperature
        self.sp = {
            a: SamplingParams(
                max_tokens=1,
                temperature=temperature if a in (*BEHAVIORS, ARMS) else 0.0,
                allowed_token_ids=list(v.values()),
                logprobs=len(v),  # the full distribution, for the heatmap
            )
            for a, v in self.vocab.items()
        }
        self.engine_loop = engine_loop
        self._warmup(warmup)
        if gc_freeze:
            gc.collect()
            gc.freeze()

    def reset(self) -> None:
        pass

    # ── Engine calls ───────────────────────────────────────────────────────────
    def run(self, prompts: list[list[int]], sps: list) -> list:
        """One batch of one-token requests; returns vLLM outputs in order."""
        from vllm.inputs import TokensPrompt

        if not self.engine_loop:
            return self.llm.generate(
                [TokensPrompt(prompt_token_ids=p) for p in prompts], sps, use_tqdm=False
            )
        engine = self.llm.llm_engine
        ids = [uuid.uuid4().hex for _ in prompts]
        for rid, p, sp in zip(ids, prompts, sps):
            engine.add_request(rid, TokensPrompt(prompt_token_ids=p), sp)
        done: dict[str, object] = {}
        while len(done) < len(ids):
            for out in engine.step():
                if out.finished:
                    done[out.request_id] = out
        return [done[rid] for rid in ids]

    def _prompt_adapter(self, adapter: str) -> str | None:
        return None if self.base_model else adapter

    def _dist(self, out, adapter: str) -> list[tuple[str, float]]:
        words = self.words[adapter]
        lp = out.outputs[0].logprobs[0]
        pairs = [(words[i], math.exp(v.logprob)) for i, v in lp.items() if i in words]
        pairs.sort(key=lambda p: -p[1])
        return pairs

    def _decision(self, out, adapter: str, t0: float, t1: float, t2: float):
        dist = self._dist(out, adapter)
        n_prompt = len(out.prompt_token_ids)
        cached = out.num_cached_tokens or 0
        return Decision(
            self.words[adapter][out.outputs[0].token_ids[0]],
            dist[:3],
            (time.perf_counter() - t0) * 1000,
            n_prompt - cached,
            cached,
            build_ms=(t1 - t0) * 1000,
            engine_ms=(t2 - t1) * 1000,
            probs=dict(dist),
        )

    # ── Policy interface ───────────────────────────────────────────────────────
    def decide_games(
        self, games: list[tuple[list[int], str]], adapters: list[list[str]]
    ) -> list[dict[str, Decision]]:
        """Many games in one engine step. ``games[i]`` is (history ids, state
        text); ``adapters[i]`` the adapters that game asks this tic. ``ms`` is
        per step."""
        t0 = time.perf_counter()
        prompts, sps, index = [], [], []
        for g, ((hist, state), names) in enumerate(zip(games, adapters)):
            ps = self.pb.game_ids(hist, state, [self._prompt_adapter(a) for a in names])
            prompts += ps
            sps += [self.sp[a] for a in names]
            index += [(g, a) for a in names]
        t1 = time.perf_counter()
        outs = self.run(prompts, sps)
        t2 = time.perf_counter()
        res: list[dict[str, Decision]] = [{} for _ in games]
        for (g, a), o in zip(index, outs):
            res[g][a] = self._decision(o, a, t0, t1, t2)
        return res

    def decide_many(
        self, obs: Observation, adapters: tuple[str, ...], history: History | None
    ) -> dict[str, Decision]:
        """Several adapters on the same state, one engine step."""
        hist = history.ids if history is not None else []
        return self.decide_games([(hist, state_text(obs))], [list(adapters)])[0]

    def decide(
        self, obs: Observation, adapter: str, history: History | None = None
    ) -> Decision:
        return self.decide_many(obs, (adapter,), history)[adapter]

    def route(self, instruction: str) -> Route:
        t0 = time.perf_counter()
        prompt = self.rb.ids(instruction, self._prompt_adapter(ROUTER))
        out = self.run([prompt], [self.sp[ROUTER]])[0]
        probs = dict(self._dist(out, ROUTER))
        best = self.words[ROUTER][out.outputs[0].token_ids[0]]
        return Route(
            best, probs.get(best, 0.0), (time.perf_counter() - t0) * 1000, probs
        )

    def talk_params(self, max_tokens: int = 24, temperature: float = 0.8, **kw):
        """Sampling for one spoken line from the base model (no adapter).

        Temperature only: top-p sends sampling to a FlashInfer kernel that
        autotunes on first use per shape, a multi-second stall mid-match.
        """
        from vllm import SamplingParams

        opts = {"stop": ["\n"], **kw}
        return SamplingParams(max_tokens=max_tokens, temperature=temperature, **opts)

    def talk(
        self, games: list[tuple[list[int], str]], briefs: list[str] | None = None, **kw
    ) -> list[str]:
        """One spoken line per game (history ids, state text), in one engine call."""
        briefs = briefs or [""] * len(games)
        prompts = [self.pb.talk_ids(h, s, b) for (h, s), b in zip(games, briefs)]
        outs = self.run(prompts, [self.talk_params(**kw)] * len(prompts))
        return [o.outputs[0].text.strip().split("\n")[0] for o in outs]

    def _warmup(self, n: int) -> None:
        state = (
            "now t1.0 | hp 100 armor 0 | pistol 50 | arms 2:50 | see bot -12 8m, "
            "medikit +40 3m | wall l3 f9 r9 b2 | hit 0 | last forward forward"
        )
        entry = "t0.2 hp 100 face 90 | bot -12 8m | did forward\n"
        hist: list[int] = []
        for i in range(n):
            hist = hist + self.tok.encode(entry, add_special_tokens=False)
            self.decide_games([(hist, state)], [list(GAME_ADAPTERS[: 1 + i % 5])])
        self.route("go kill everything")
        for n_games in (1, 4, 16):  # the talk path, at a few batch sizes
            self.talk([(hist, state)] * n_games, max_tokens=4)


def make_policy(kind: str, model: str | None = None, **kw):
    if kind == "expert":
        return ExpertPolicy()
    if kind == "vllm":
        if not model:
            raise SystemExit("--model is required for --policy vllm")
        return VLLMPolicy(model, **kw)
    raise SystemExit(f"unknown policy {kind!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--check-template",
        metavar="MODEL_DIR",
        required=True,
        help="Composed model dir (or a base model id for the base-template check)",
    )
    ap.add_argument(
        "--seconds", type=float, default=40.0, help="Seconds of play to check"
    )
    args = ap.parse_args()

    from doom_env import TIC_HZ, DoomEnv
    from expert import PLAN_EVERY_TICS

    env = DoomEnv(seed=0, timeout_tics=int(args.seconds * TIC_HZ))
    ex, hist = Expert(), History()
    games = []
    obs = env.reset(seed=0)
    while not obs.done:
        a = ex.act(obs, BEHAVIORS[obs.tick // 350 % len(BEHAVIORS)])
        if not obs.dead and obs.tick % 5 == 0:
            games.append((hist.text, state_text(obs)))
        hist.observe(obs, a)
        w = ex.weapon(obs) if obs.tick % PLAN_EVERY_TICS == 0 else None
        obs = env.step(a, weapon=w)
    env.close()
    routes = ["go kill everything", "stay alive, grab health", "collect all the loot"]
    check_template(args.check_template, games[::5], routes)


if __name__ == "__main__":
    main()
