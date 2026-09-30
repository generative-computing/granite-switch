# SPDX-License-Identifier: Apache-2.0
"""ViZDoom wrapper for the Granite Switch Doom demo: deathmatch against built-in bots.

The arena is cig.wad MAP02 (full deathmatch, the ViZDoom competition map),
hosted in synchronous ``Mode.PLAYER`` with ZDoom bots from :data:`BOTS_CFG`, as
in ViZDoom's ``examples/python/cig_multiplayer_bots.py``. Two deliberate
differences from the competition rules: ``+viz_nocheat`` is off, because it
disables the labels, objects, sectors and position data every view below is
built from; and vertical autoaim is on, because the action space is 2D.

One :meth:`DoomEnv.step` is one Doom tic (1/35 s). Every :class:`Observation`
carries two views of the same moment:

* **Player-visible** (``obs.text`` and the fields it is built from, and
  :func:`features`, the same fields as numbers): HUD values including owned
  weapons, the player's map position and heading (what the automap and knowing
  the map give a human), the enemies, items and incoming missiles currently on
  screen (read from the labels buffer) as bearing and distance, four wall
  clearances, and a small memory the wrapper keeps (last two actions, damage
  taken in the last second, last-seen enemy bearing). This is all the policies
  read.
* **Privileged** (``obs.priv`` and :func:`priv_features`): every object in the
  level with world positions, every bot's position, the scoreboard. Only the
  scripted expert and the RL teacher's critic may use it.

Death is tic-accurate. While the player is dead every step holds USE and
ignores the action, so the respawn delay passes one tic at a time and
``obs.dead`` is set; ``DoomGame.respawn_player`` would instead skip the whole
delay inside one call while the bots keep playing.

Weapons are chosen from outside: ``step(action, weapon=slot)`` records the
wanted slot, and the wrapper presses it whenever it is owned, has ammo and is
not already selected. Nothing is auto-selected.

Conventions: bearings are degrees relative to the player's heading, negative to
the left and positive to the right, so ``left`` shrinks a negative bearing.
Distances are metres at 32 map units per metre.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import vizdoom as vzd
from vizdoom import Button, GameVariable

TIC_HZ = 35
TIC_MS = 1000.0 / TIC_HZ
AUDIO_HZ = 22050  # the game's sound, with DoomEnv(audio=True)
UNITS_PER_M = 32.0  # Doomguy is 56 units tall, about 1.75 m
MATCH_TICS = 10 * 60 * TIC_HZ  # a 10-minute deathmatch

# ── Arena ──────────────────────────────────────────────────────────────────────
BOTS_CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bots.cfg")
BOT_SETS: dict[str, tuple[str, ...]] = {
    "easy": ("Rookie", "Cadet", "Recruit", "Trainee", "Newbie", "Intern", "Novice"),
    # The first seven bots of the bots.cfg shipped with ViZDoom, unchanged.
    "default": (
        "Rambo",
        "McClane",
        "MacGyver",
        "Plissken",
        "Machete",
        "Anderson",
        "Leone",
    ),
    "hard": ("Viper", "Cobra", "Mamba", "Python", "Taipan", "Krait", "Adder"),
    "perfect": ("Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf"),
}
PLAYER_NAME = "AI"
_GAME_ARGS = (
    "-host 1 -deathmatch +sv_forcerespawn 1 +sv_respawnprotect 1 "
    "+sv_spawnfarthest 1 +sv_nocrouch 1 +sv_nojump 1 +sv_nofreelook 1 "
    f"+name {PLAYER_NAME} +colorset 0"
)
# Seconds a death keeps the player out. The ViZDoom competition uses 10
# (cig_multiplayer_bots.py); this demo uses 1, and every result here does
# unless it says otherwise.
RESPAWN_S = 1
# Item rules, set on the server console after each reset (ZDoom reapplies its
# deathmatch defaults at game start, so command-line cvars do not stick):
#   standard: ZDoom's deathmatch defaults, items respawn after 30 s and weapons
#             stay on the floor, so ammo is effectively unlimited
#   classic:  Doom's original "altdeath": items respawn, weapons do not stay
#   scarce:   nothing respawns and weapons do not stay; after the map's supply
#             only kills' drops and the respawn pistol are left
ITEM_RULES = {
    "standard": {"sv_itemrespawn": "true", "sv_weaponstay": "true"},
    "classic": {"sv_itemrespawn": "true", "sv_weaponstay": "false"},
    "scarce": {"sv_itemrespawn": "false", "sv_weaponstay": "false"},
}

# ── Action space ───────────────────────────────────────────────────────────────
# Each name is one token in the Granite 4.1 tokenizer, with no leading space
# (the action follows <|end_of_role|> directly). policy.check_output_tokens
# asserts this against the real tokenizer.
ACTIONS: tuple[str, ...] = (
    "forward",
    "back",
    "left",
    "right",
    "sl",
    "sr",
    "fl",
    "fr",
    "fire",
    "al",
    "ar",
    "wait",
    "charge",
    "bf",
    "cl",
    "cr",
    "dl",
    "dr",
    "ol",
    "or",
)
ACTION_LABELS = {
    "forward": "forward",
    "back": "back",
    "left": "turn left",
    "right": "turn right",
    "sl": "strafe left",
    "sr": "strafe right",
    "fl": "forward + left",
    "fr": "forward + right",
    "fire": "fire",
    "al": "fire + aim left",
    "ar": "fire + aim right",
    "wait": "wait",
    "charge": "charge (forward + fire)",
    "bf": "back + fire",
    "cl": "strafe left + fire",
    "cr": "strafe right + fire",
    "dl": "strafe-run left",
    "dr": "strafe-run right",
    "ol": "circle-strafe left",
    "or": "circle-strafe right",
}
# Row order for probability heatmaps: combat, running, turning, strafing, other.
DISPLAY_ORDER: tuple[str, ...] = (
    "fire",
    "al",
    "ar",
    "charge",
    "cl",
    "cr",
    "bf",
    "forward",
    "fl",
    "fr",
    "dl",
    "dr",
    "left",
    "right",
    "sl",
    "sr",
    "ol",
    "or",
    "back",
    "wait",
)
assert sorted(DISPLAY_ORDER) == sorted(ACTIONS)
# Heatmap row labels (short enough for a narrow gutter).
SHORT_LABELS = {
    "fire": "fire",
    "al": "fire+aim L",
    "ar": "fire+aim R",
    "charge": "charge",
    "cl": "fire+strf L",
    "cr": "fire+strf R",
    "bf": "back+fire",
    "forward": "forward",
    "fl": "fwd+left",
    "fr": "fwd+right",
    "dl": "strf-run L",
    "dr": "strf-run R",
    "left": "turn left",
    "right": "turn right",
    "sl": "strafe L",
    "sr": "strafe R",
    "ol": "circle L",
    "or": "circle R",
    "back": "back",
    "wait": "wait",
}
assert set(SHORT_LABELS) == set(ACTIONS)
ATTACKS = frozenset({"fire", "al", "ar", "charge", "bf", "cl", "cr"})

TURN_DEG = 10.0  # left / right: fast turn in place
VEER_DEG = 4.0  # fl / fr: turn while running
AIM_DEG = 2.5  # al / ar: fine aim while firing
ORBIT_DEG = 4.0  # ol / or: turn against the strafe, circling a target ~4-8 m out

# name -> (forward, back, strafe_left, strafe_right, attack, turn_delta_deg)
# A positive TURN_LEFT_RIGHT_DELTA turns right, matching the bearing sign.
_MACROS: dict[str, tuple[int, int, int, int, int, float]] = {
    "forward": (1, 0, 0, 0, 0, 0.0),
    "back": (0, 1, 0, 0, 0, 0.0),
    "left": (0, 0, 0, 0, 0, -TURN_DEG),
    "right": (0, 0, 0, 0, 0, TURN_DEG),
    "sl": (0, 0, 1, 0, 0, 0.0),
    "sr": (0, 0, 0, 1, 0, 0.0),
    "fl": (1, 0, 0, 0, 0, -VEER_DEG),
    "fr": (1, 0, 0, 0, 0, VEER_DEG),
    "fire": (0, 0, 0, 0, 1, 0.0),
    "al": (0, 0, 0, 0, 1, -AIM_DEG),
    "ar": (0, 0, 0, 0, 1, AIM_DEG),
    "wait": (0, 0, 0, 0, 0, 0.0),
    "charge": (1, 0, 0, 0, 1, 0.0),
    "bf": (0, 1, 0, 0, 1, 0.0),
    "cl": (0, 0, 1, 0, 1, 0.0),
    "cr": (0, 0, 0, 1, 1, 0.0),
    "dl": (1, 0, 1, 0, 0, 0.0),
    "dr": (1, 0, 0, 1, 0, 0.0),
    "ol": (0, 0, 1, 0, 0, ORBIT_DEG),
    "or": (0, 0, 0, 1, 0, -ORBIT_DEG),
}
assert set(_MACROS) == set(ACTIONS)

_BUTTONS = [
    Button.SPEED,  # always held: Doom players run (16.7 vs 8.3 units/tic)
    Button.ATTACK,
    Button.USE,  # held only while dead, to respawn
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.MOVE_LEFT,
    Button.MOVE_RIGHT,
    Button.TURN_LEFT_RIGHT_DELTA,
    *[getattr(Button, f"SELECT_WEAPON{i}") for i in range(1, 8)],
]
_N_MOVE_BUTTONS = 8  # everything before the weapon-select buttons
_DEAD_BUTTONS = [0, 0, 1] + [0] * (len(_BUTTONS) - 3)

# ── Weapons ────────────────────────────────────────────────────────────────────
WEAPON_SLOTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
WEAPON_NAMES = {
    1: "fist",
    2: "pistol",
    3: "shotgun",
    4: "chaingun",
    5: "launcher",
    6: "plasma",
    7: "bfg",
}
_WEAPON_SWITCH_COOLDOWN = 10  # tics between presses; one press takes effect in 1

_VARS = [
    GameVariable.HEALTH,
    GameVariable.ARMOR,
    GameVariable.SELECTED_WEAPON,
    GameVariable.SELECTED_WEAPON_AMMO,
    GameVariable.FRAGCOUNT,
    GameVariable.DEATHCOUNT,
    GameVariable.HITCOUNT,
    GameVariable.DAMAGECOUNT,
    GameVariable.DAMAGE_TAKEN,
    GameVariable.DEAD,
    GameVariable.POSITION_X,
    GameVariable.POSITION_Y,
    GameVariable.ANGLE,
    GameVariable.PLAYER_NUMBER,
    *[getattr(GameVariable, f"WEAPON{i}") for i in WEAPON_SLOTS],
    *[getattr(GameVariable, f"AMMO{i}") for i in WEAPON_SLOTS],
    *[getattr(GameVariable, f"PLAYER{i}_FRAGCOUNT") for i in range(1, 9)],
]
_V = {v: i for i, v in enumerate(_VARS)}
# Match counters, zeroed at reset.
_COUNTERS = {
    "frags": GameVariable.FRAGCOUNT,
    "deaths": GameVariable.DEATHCOUNT,
    "hits": GameVariable.HITCOUNT,
    "dealt": GameVariable.DAMAGECOUNT,
    "taken": GameVariable.DAMAGE_TAKEN,
}
_W = [_V[getattr(GameVariable, f"WEAPON{i}")] for i in WEAPON_SLOTS]
_A = [_V[getattr(GameVariable, f"AMMO{i}")] for i in WEAPON_SLOTS]
_PF = [_V[getattr(GameVariable, f"PLAYER{i}_FRAGCOUNT")] for i in range(1, 9)]

# ── Object vocabulary ──────────────────────────────────────────────────────────
ENEMY_LABEL = "bot"
# class name -> (text label, kind)
ITEMS: dict[str, tuple[str, str]] = {
    "Medikit": ("medikit", "health"),
    "Stimpack": ("stim", "health"),
    "Soulsphere": ("soulsphere", "health"),
    "Megasphere": ("megasphere", "health"),
    "Berserk": ("berserk", "health"),
    "HealthBonus": ("bonus", "bonus"),
    "ArmorBonus": ("bonus", "bonus"),
    "GreenArmor": ("armor", "armor"),
    "BlueArmor": ("bluearmor", "armor"),
    "Clip": ("bullets", "ammo"),
    "ClipBox": ("bullets", "ammo"),
    "Shell": ("shells", "ammo"),
    "ShellBox": ("shells", "ammo"),
    "RocketAmmo": ("rockets", "ammo"),
    "RocketBox": ("rockets", "ammo"),
    "Cell": ("cells", "ammo"),
    "CellPack": ("cells", "ammo"),
    "Backpack": ("backpack", "ammo"),
    "Chainsaw": ("chainsaw", "weapon"),
    "Shotgun": ("shotgun", "weapon"),
    "SuperShotgun": ("ssg", "weapon"),
    "Chaingun": ("chaingun", "weapon"),
    "RocketLauncher": ("launcher", "weapon"),
    "PlasmaRifle": ("plasma", "weapon"),
    "BFG9000": ("bfg", "weapon"),
}
ITEM_KINDS = ("health", "armor", "ammo", "weapon", "bonus")
WEAPON_ITEM_SLOT = {
    "Chainsaw": 1,
    "Shotgun": 3,
    "SuperShotgun": 3,
    "Chaingun": 4,
    "RocketLauncher": 5,
    "PlasmaRifle": 6,
    "BFG9000": 7,
}
AMMO_ITEM_SLOT = {  # the slot whose weapon the ammo feeds (bullets: chaingun)
    "Clip": 4,
    "ClipBox": 4,
    "Shell": 3,
    "ShellBox": 3,
    "RocketAmmo": 5,
    "RocketBox": 5,
    "Cell": 6,
    "CellPack": 6,
}
MISSILES = {"Rocket", "PlasmaBall", "BFGBall"}
_ALWAYS_PICKABLE_HEALTH = {"Soulsphere", "Megasphere", "Berserk"}

MAX_SEEN_ENEMIES = 3
MAX_SEEN_TOTAL = 6
MIN_SEEN_ITEMS = 2
ENEMY_MEMORY_TICS = 5 * TIC_HZ

# Wall probes: (label, bearing) with a small fan per probe so a pillar edge
# just off-centre still registers. Clearance subtracts the player radius. The
# rear probe stands in for the player knowing what is behind them.
WALL_PROBES = (("l", -35.0), ("f", 0.0), ("r", 35.0), ("b", 180.0))
_PROBE_FAN = (-8.0, 0.0, 8.0)
_PLAYER_RADIUS = 16.0
_MAX_STEP_UP = 24.0
_PLAYER_HEIGHT = 56.0
WALL_CAP_M = 9


def isolate_workdir() -> None:
    """Give this process its own working directory.

    ViZDoom writes ``_vizdoom.ini`` and ``_vizdoom/`` into the current directory,
    and instances starting at the same moment race to create the directory; the
    losers exit. Call at the top of every worker process.
    """
    os.chdir(tempfile.mkdtemp(prefix="vizdoom-"))


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class Seen:
    """One object on screen, as the player sees it."""

    id: int
    cls: str  # ViZDoom class name, e.g. "Medikit"; "DoomPlayer" for a bot
    label: str  # text label, e.g. "medikit"; "bot" for an enemy
    kind: str  # "enemy" | "missile" | "health" | "armor" | "ammo" | "weapon" | "bonus"
    bearing: float  # degrees, negative left
    dist: float  # metres

    @property
    def b(self) -> int:
        return round(self.bearing)

    @property
    def d(self) -> int:
        return round(self.dist)


@dataclass
class Privileged:
    """Ground truth the expert and the RL critic may use; the model never sees it."""

    x: float
    y: float
    angle: float  # Doom world angle, degrees, counter-clockwise from +x
    objects: list[tuple[int, str, str, float, float]]  # (id, class, category, x, y)
    enemies: list[tuple[float, float]]  # every live bot's (x, y)
    seen_all: list[Seen]  # every on-screen object, before capping
    walls_raw: tuple[float, ...]  # l, f, r, b clearance in metres
    scoreboard: list[tuple[str, int]]  # (name, frags), this player first


@dataclass
class Observation:
    tick: int
    text: str
    hp: int
    armor: int
    ammo: int
    weapon: str
    slot: int  # selected weapon slot
    arms: dict[int, int]  # owned slot -> ammo (the fist, slot 1, always owned)
    pos: tuple[int, int]  # metres east and north of the map's south-west corner
    face: int  # heading, whole degrees counter-clockwise from east, 0-359
    seen: list[Seen]  # the capped subset that appears in ``text``
    walls: tuple[int, int, int, int]  # l, f, r, b clearance, whole metres, capped
    hit: int  # damage taken in the last second
    enemy_mem: tuple[int, int] | None  # (bearing, seconds ago), only if none seen
    last: tuple[str, str]  # last two executed actions, oldest first
    frags: int
    deaths: int
    dead: bool  # respawning: the action is ignored, no decision is needed
    events: tuple[str, ...]  # this tic: frag, suicide, died, respawn, got <kind>
    counters: dict[str, float]  # raw game counters, for reward shaping
    priv: Privileged
    done: bool = False


@dataclass
class MatchStats:
    tics: int = 0
    frags: int = 0
    deaths: int = 0
    suicides: int = 0
    damage_taken: int = 0
    damage_dealt: int = 0
    hits: int = 0
    shots: int = 0  # tics with the attack button held
    pickups: dict[str, int] = field(default_factory=dict)
    threat_tics: int = 0  # tics with an enemy on screen
    threat_dist_sum: float = 0.0  # sum of nearest on-screen enemy distance
    scoreboard: list[tuple[str, int]] = field(default_factory=list)

    @property
    def pickups_total(self) -> int:
        return sum(self.pickups.values())

    @property
    def best_bot(self) -> tuple[str, int]:
        return max(self.scoreboard[1:], key=lambda s: s[1], default=("", 0))

    @property
    def margin(self) -> int:
        """Frags ahead of the best bot (negative when behind)."""
        return self.frags - self.best_bot[1]

    @property
    def rank(self) -> int:
        """1 + the number of bots strictly ahead; a tie for first is rank 1."""
        return 1 + sum(f > self.frags for _, f in self.scoreboard[1:])

    def as_dict(self) -> dict:
        minutes = max(1e-9, self.tics / TIC_HZ / 60)
        return {
            "tics": self.tics,
            "seconds": round(self.tics / TIC_HZ, 1),
            "frags": self.frags,
            "deaths": self.deaths,
            "suicides": self.suicides,
            "kd": round(self.frags / max(1, self.deaths), 2),
            "frags_per_min": round(self.frags / minutes, 2),
            "margin": self.margin,
            "rank": self.rank,
            "top": self.rank == 1,
            "best_bot": list(self.best_bot),
            "damage_taken": self.damage_taken,
            "damage_dealt": self.damage_dealt,
            "hits": self.hits,
            "shots": self.shots,
            "pickups": self.pickups_total,
            "pickups_by_kind": dict(self.pickups),
            "threat_s": round(self.threat_tics / TIC_HZ, 1),
            "threat_dist": round(self.threat_dist_sum / max(1, self.threat_tics), 2),
            "scoreboard": [list(s) for s in self.scoreboard],
        }


def serialize(
    hp: int,
    armor: int,
    ammo: int,
    weapon: str,
    arms: dict[int, int],
    pos: tuple[int, int],
    face: int,
    seen: list[Seen],
    walls: tuple[int, int, int, int],
    hit: int,
    enemy_mem: tuple[int, int] | None,
    last: tuple[str, str],
) -> str:
    """Build the player-visible state line the model reads (about 50-90 tokens)."""
    objs = ", ".join(f"{o.label} {o.b:+d} {o.d}m" for o in seen) or "nothing"
    owned = " ".join(f"{s}:{a}" for s, a in sorted(arms.items()) if s > 1) or "none"
    l, f, r, b = walls
    text = (
        f"hp {hp} armor {armor} | at {pos[0]},{pos[1]} face {face} | {weapon} {ammo} | "
        f"arms {owned} | see {objs} | wall l{l} f{f} r{r} b{b} | hit {hit}"
    )
    if enemy_mem is not None:
        text += f" | enemy {enemy_mem[0]:+d} {enemy_mem[1]}s"
    return text + f" | last {last[0]} {last[1]}"


def pickable(cls: str, kind: str, hp: int, armor: int, arms: dict[int, int]) -> bool:
    """Whether a player with this HUD would pick the item up by touching it.

    Uses only HUD values so the rule is learnable from the text. Weapons stay on
    the floor in deathmatch and can only be taken once; ammo is assumed
    pickable (being full is rare).
    """
    if kind == "health":
        return hp < 100 or cls in _ALWAYS_PICKABLE_HEALTH
    if kind == "armor":
        return armor < (200 if cls == "BlueArmor" else 100)
    if kind == "weapon":
        return WEAPON_ITEM_SLOT[cls] not in arms or cls == "Chainsaw"
    return kind in ("ammo", "bonus")


class DoomEnv:
    """Synchronous deathmatch on cig.wad MAP02 against named ZDoom bots.

    Args:
        bots: A :data:`BOT_SETS` name or an explicit tuple of bot names from
            :data:`BOTS_CFG`.
        n_bots: How many of them join (the first ``n_bots``).
        timeout_tics: Match length; :data:`MATCH_TICS` is 10 minutes.
        respawn_s: Seconds a death keeps the player out (:data:`RESPAWN_S`).
        item_rules: A key of :data:`ITEM_RULES`.
        audio: Keep the game's sound, as the player hears it: :meth:`audio`
            returns each tic's :data:`AUDIO_HZ` stereo samples (for videos).
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        resolution: str = "320X240",
        hud: bool = False,
        timeout_tics: int = MATCH_TICS,
        bots: str | tuple[str, ...] = "default",
        n_bots: int = 7,
        respawn_s: int = RESPAWN_S,
        item_rules: str = "standard",
        audio: bool = False,
    ):
        if item_rules not in ITEM_RULES:
            raise ValueError(f"item_rules must be one of {sorted(ITEM_RULES)}")
        g = vzd.DoomGame()
        g.load_config(os.path.join(vzd.scenarios_path, "cig.cfg"))
        g.set_doom_map("map02")
        g.set_mode(vzd.Mode.PLAYER)  # after load_config: cig.cfg sets ASYNC_PLAYER
        g.add_game_args(_GAME_ARGS)
        g.add_game_args(f"+viz_respawn_delay {int(respawn_s)}")
        g.add_game_args(f"+viz_bots_path {BOTS_CFG}")
        g.set_available_buttons(_BUTTONS)
        g.set_available_game_variables(_VARS)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_labels_buffer_enabled(True)
        g.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{resolution}"))
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(hud)
        g.set_render_crosshair(hud)
        g.set_render_weapon(True)
        g.set_window_visible(False)
        g.set_sound_enabled(audio)  # the audio buffer is silent without it
        if audio:  # rendered to a buffer (OpenAL loopback): no sound device needed
            g.set_audio_buffer_enabled(True)
            g.set_audio_sampling_rate(vzd.SamplingRate.SR_22050)
            g.set_audio_buffer_size(1)  # one tic: each step's own sound
        self._audio: np.ndarray | None = None
        g.set_episode_timeout(timeout_tics)
        if seed is not None:
            g.set_seed(seed)
        g.init()
        self.game = g
        self.bots = bots
        self.n_bots = n_bots
        self.item_rules = item_rules
        self._frame: np.ndarray | None = None
        self._segments: np.ndarray | None = None
        self._origin: tuple[float, float] | None = None
        self._heights: tuple | None = None
        self._segments_tick = -(10**9)
        self.stats = MatchStats()
        self.obs: Observation | None = None

    # ── Episode control ──────────────────────────────────────────────────────
    def bot_names(self, bots: str | tuple[str, ...] | None = None) -> tuple[str, ...]:
        bots = self.bots if bots is None else bots
        names = BOT_SETS[bots] if isinstance(bots, str) else tuple(bots)
        return names[: self.n_bots]

    def reset(
        self, seed: int | None = None, bots: str | tuple[str, ...] | None = None
    ) -> Observation:
        if seed is not None:
            self.game.set_seed(seed)
        if bots is not None:
            self.bots = bots
        g = self.game
        g.new_episode()
        for cvar, value in ITEM_RULES[self.item_rules].items():
            g.send_game_command(f"{cvar} {value}")
        g.send_game_command("removebots")
        names = self.bot_names()
        for name in names:
            g.send_game_command(f"addbot {name}")
        g.make_action([0] * len(_BUTTONS), 1)  # bots join on the next tic
        joined = g.get_server_state().player_count - 1
        if joined != len(names):
            raise RuntimeError(
                f"{joined} of {len(names)} bots joined; are all of {names} in {BOTS_CFG}?"
            )
        # A joining bot can spawn on top of the player (a telefrag). That is not
        # part of the match: wait out the respawn, then zero every counter here,
        # since ViZDoom also carries our counters over from the last episode.
        for _ in range(30 * TIC_HZ):
            if not g.is_player_dead():
                break
            g.make_action(_DEAD_BUTTONS, 1)
        gv = g.get_state().game_variables
        self._base = {k: gv[_V[v]] for k, v in _COUNTERS.items()}
        self._frag0 = [gv[i] for i in _PF]
        self._names = list(g.get_server_state().players_names)
        self.stats = MatchStats()
        self._tick = 0
        self._last: deque[str] = deque(["wait", "wait"], maxlen=2)
        self._dmg_hist: deque[float] = deque(maxlen=TIC_HZ + 1)
        self._enemy_seen: tuple[float, float, int] | None = None  # x, y, tick
        self._want_slot: int | None = None
        self._next_switch = 0
        self._prev: dict[str, float] | None = None
        self._was_dead = False
        self._segments_tick = -(10**9)
        self.obs = self._observe()
        return self.obs

    def step(self, action: str, weapon: int | None = None) -> Observation:
        """Execute one macro action for one tic and return the next observation.

        ``weapon`` (a slot, 1-7) replaces the wanted weapon; the wrapper keeps
        pressing it until it is selected, whenever it is owned and has ammo.
        While the player is dead the action is ignored and USE is held.
        """
        assert self.obs is not None and not self.obs.done, "call reset() first"
        if weapon is not None:
            self._want_slot = int(weapon)
        if self.obs.dead:
            buttons = list(_DEAD_BUTTONS)
            action = "wait"
        else:
            fwd, back, sl, sr, atk, turn = _MACROS[action]
            buttons = [1, atk, 0, fwd, back, sl, sr, turn] + [0] * 7
            slot = self._weapon_to_press()
            if slot is not None:
                buttons[_N_MOVE_BUTTONS + slot - 1] = 1
                self._next_switch = self._tick + _WEAPON_SWITCH_COOLDOWN
            self.stats.shots += atk
        self.game.make_action(buttons, 1)
        self._tick += 1
        self._last.append(action)
        if self.game.is_episode_finished():
            self.stats.tics = self._tick
            self._score(self.game.get_server_state().players_frags)
            self.obs.done = True
            return self.obs
        self.obs = self._observe()
        return self.obs

    def frame(self) -> np.ndarray | None:
        """Latest RGB frame (H, W, 3), or None before the first reset."""
        return self._frame

    def audio(self) -> np.ndarray | None:
        """The last tic's sound, (AUDIO_HZ / TIC_HZ, 2) int16, with ``audio=True``."""
        return self._audio

    def close(self) -> None:
        self.game.close()

    # ── Internals ────────────────────────────────────────────────────────────
    def _arms(self, gv) -> dict[int, int]:
        arms = {1: 0}
        for s, wi, ai in zip(WEAPON_SLOTS, _W, _A):
            if s > 1 and gv[wi] > 0:
                arms[s] = max(0, int(gv[ai]))
        return arms

    def _weapon_to_press(self) -> int | None:
        slot = self._want_slot
        if slot is None or self._tick < self._next_switch or slot == self.obs.slot:
            return None
        if slot not in self.obs.arms or (slot > 1 and self.obs.arms[slot] <= 0):
            return None
        return slot

    def _score(self, frags_by_player) -> None:
        me = int(self._gv[_V[GameVariable.PLAYER_NUMBER]])
        n = min(len(frags_by_player), len(self._frag0))
        board = [
            (name, int(frags_by_player[i] - self._frag0[i]))
            for i, name in enumerate(self._names[:n])
            if name and i != me
        ]
        board.sort(key=lambda s: -s[1])
        self.stats.scoreboard = [(PLAYER_NAME, self.stats.frags), *board]

    def _refresh_segments(self, sectors) -> None:
        """Collect impassable line segments as an (N, 4) array of x1, y1, x2, y2.

        A two-sided line blocks when the floor step exceeds what the player can
        climb or the opening is lower than the player. Drop-offs count as walls
        too, which keeps the teacher off ledges it could not climb back up.
        """
        owners: dict[tuple, tuple[list[bool], list]] = {}
        for sec in sectors:
            for ln in sec.lines:
                key = tuple(sorted(((ln.x1, ln.y1), (ln.x2, ln.y2))))
                flags, secs = owners.setdefault(key, ([], []))
                flags.append(ln.is_blocking)
                secs.append(sec)
        segs = []
        for (p1, p2), (flags, secs) in owners.items():
            if len(secs) < 2 or any(flags):
                block = True
            else:
                a, b = secs[0], secs[1]
                step = abs(a.floor_height - b.floor_height)
                opening = min(a.ceiling_height, b.ceiling_height) - max(
                    a.floor_height, b.floor_height
                )
                block = step > _MAX_STEP_UP or opening < _PLAYER_HEIGHT
            if block:
                segs.append((*p1, *p2))
        self._segments = np.asarray(segs, dtype=np.float64).reshape(-1, 4)
        if self._origin is None:  # the map's south-west corner, for positions
            s = self._segments
            self._origin = (float(s[:, [0, 2]].min()), float(s[:, [1, 3]].min()))

    def _raycast(self, x: float, y: float, angles_deg: np.ndarray) -> np.ndarray:
        """Distance in map units to the nearest blocking segment along each angle."""
        segs = self._segments
        th = np.radians(angles_deg)[:, None]
        dx, dy = np.cos(th), np.sin(th)
        ax, ay = segs[:, 0] - x, segs[:, 1] - y
        ex, ey = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * ey - ay * ex) / den
            u = (ax * dy - ay * dx) / den
        hit = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        t = np.where(hit, t, np.inf)
        return t.min(axis=1)

    def _events(self, gv, hp: int, armor: int, arms: dict[int, int], dead: bool):
        """Per-tic events and counter updates from HUD and scoreboard deltas."""
        cur = {k: gv[_V[v]] - self._base[k] for k, v in _COUNTERS.items()}
        cur |= {
            "hp": float(hp),
            "armor": float(armor),
            "weapons": float(len(arms)),
            "ammo": float(sum(arms.values())),
        }
        prev, self._prev = self._prev, cur
        events: list[str] = []
        if prev is None:
            return events, cur
        df = int(cur["frags"] - prev["frags"])
        events += ["frag"] * max(0, df) + ["suicide"] * max(0, -df)
        self.stats.suicides += max(0, -df)
        if dead and not self._was_dead:
            events.append("died")
        elif self._was_dead and not dead:
            events.append("respawn")
        elif not dead:  # a respawn resets the HUD; that is not a pickup
            got = []
            if cur["weapons"] > prev["weapons"]:
                got.append("weapon")
            elif cur["ammo"] > prev["ammo"]:
                got.append("ammo")
            if cur["hp"] > prev["hp"]:
                got.append("health")
            if cur["armor"] > prev["armor"]:
                got.append("armor")
            for k in got:
                self.stats.pickups[k] = self.stats.pickups.get(k, 0) + 1
            events += [f"got {k}" for k in got]
        return events, cur

    def _observe(self) -> Observation:
        st = self.game.get_state()
        gv = st.game_variables
        self._gv = gv
        self._frame = st.screen_buffer
        self._audio = st.audio_buffer
        dead = bool(gv[_V[GameVariable.DEAD]]) or self.game.is_player_dead()
        hp = max(0, int(gv[_V[GameVariable.HEALTH]]))
        armor = int(gv[_V[GameVariable.ARMOR]])
        slot = int(gv[_V[GameVariable.SELECTED_WEAPON]])
        slot = slot if slot in WEAPON_NAMES else 1
        weapon = WEAPON_NAMES[slot]
        ammo = (
            0 if slot <= 1 else max(0, int(gv[_V[GameVariable.SELECTED_WEAPON_AMMO]]))
        )
        arms = self._arms(gv)
        px, py = gv[_V[GameVariable.POSITION_X]], gv[_V[GameVariable.POSITION_Y]]
        heading = gv[_V[GameVariable.ANGLE]]

        events, counters = self._events(gv, hp, armor, arms, dead)
        if "respawn" in events:
            self._dmg_hist.clear()
            self._enemy_seen = None
            self._last.extend(["wait", "wait"])
        self._was_dead = dead
        s = self.stats
        s.frags = int(counters["frags"])
        s.deaths = int(counters["deaths"])
        s.hits = int(counters["hits"])
        s.damage_dealt = int(counters["dealt"])
        s.damage_taken = int(counters["taken"])
        s.tics = self._tick
        self._score([gv[i] for i in _PF])

        # Damage in the last second.
        self._dmg_hist.append(counters["taken"])
        hit = int(self._dmg_hist[-1] - self._dmg_hist[0])

        objects = [
            (o.id, o.name, o.category, o.position_x, o.position_y) for o in st.objects
        ]
        enemies = [(x, y) for _, _, cat, x, y in objects if cat == "Player"]

        # On-screen objects from the labels buffer.
        seen_all: list[Seen] = []
        seen_ids = set()
        for lab in st.labels:
            name, oid, cat = lab.object_name, lab.object_id, lab.object_category
            if oid in seen_ids:
                continue
            dx, dy = lab.object_position_x - px, lab.object_position_y - py
            if cat == "Player":
                label, kind = ENEMY_LABEL, "enemy"
            elif name in ITEMS:
                label, kind = ITEMS[name]
            elif name in MISSILES:
                # Only incoming ones: the player's own rockets fly away.
                closing = lab.object_velocity_x * dx + lab.object_velocity_y * dy
                if closing >= 0:
                    continue
                label, kind = "missile", "missile"
            else:
                continue
            seen_ids.add(oid)
            bearing = -_wrap180(math.degrees(math.atan2(dy, dx)) - heading)
            dist = math.hypot(dx, dy) / UNITS_PER_M
            seen_all.append(Seen(oid, name, label, kind, bearing, dist))
        seen_all.sort(key=lambda o: o.dist)

        foes = [o for o in seen_all if o.kind == "enemy"][:MAX_SEEN_ENEMIES]
        if foes and not dead:
            s.threat_tics += 1
            s.threat_dist_sum += foes[0].dist
        missiles = [o for o in seen_all if o.kind == "missile"][:1]
        items = [
            o
            for o in seen_all
            if o.kind in ITEM_KINDS and pickable(o.cls, o.kind, hp, armor, arms)
        ]
        n_fixed = len(foes) + len(missiles)
        items = items[: max(MIN_SEEN_ITEMS, MAX_SEEN_TOTAL - n_fixed)]
        seen = foes + missiles + items

        # Enemy memory: remember where the nearest enemy was last seen and
        # report its bearing from where the player stands now.
        if foes:
            m = foes[0]
            th = math.radians(heading - m.bearing)
            d = m.dist * UNITS_PER_M
            self._enemy_seen = (
                px + d * math.cos(th),
                py + d * math.sin(th),
                self._tick,
            )
            enemy_mem = None
        elif (
            self._enemy_seen is not None
            and self._tick - self._enemy_seen[2] <= ENEMY_MEMORY_TICS
        ):
            ex, ey, t0 = self._enemy_seen
            b = -_wrap180(math.degrees(math.atan2(ey - py, ex - px)) - heading)
            enemy_mem = (round(b), round((self._tick - t0) / TIC_HZ))
        else:
            enemy_mem = None

        # Wall clearances. The blocking lines change only when a lift or door
        # moves, so rebuild them only when some sector height has changed.
        if self._tick - self._segments_tick >= 10:
            heights = tuple(
                (sec.floor_height, sec.ceiling_height) for sec in st.sectors
            )
            if heights != self._heights:
                self._refresh_segments(st.sectors)
                self._heights = heights
            self._segments_tick = self._tick
        angles = np.array(
            [heading - (b + f) for _, b in WALL_PROBES for f in _PROBE_FAN]
        )
        dist = self._raycast(px, py, angles).reshape(len(WALL_PROBES), -1).min(axis=1)
        walls_raw = tuple(
            float(max(0.0, d - _PLAYER_RADIUS) / UNITS_PER_M) for d in dist
        )
        walls = tuple(min(WALL_CAP_M, int(w)) for w in walls_raw)
        pos = (
            max(0, round((px - self._origin[0]) / UNITS_PER_M)),
            max(0, round((py - self._origin[1]) / UNITS_PER_M)),
        )
        face = round(heading) % 360

        last = (self._last[0], self._last[1])
        if dead:
            text = "dead, respawning"
        else:
            text = serialize(
                hp,
                armor,
                ammo,
                weapon,
                arms,
                pos,
                face,
                seen,
                walls,
                hit,
                enemy_mem,
                last,
            )
        priv = Privileged(
            px, py, heading, objects, enemies, seen_all, walls_raw, s.scoreboard
        )
        return Observation(
            tick=self._tick,
            text=text,
            hp=hp,
            armor=armor,
            ammo=ammo,
            weapon=weapon,
            slot=slot,
            arms=arms,
            pos=pos,
            face=face,
            seen=seen,
            walls=walls,
            hit=hit,
            enemy_mem=enemy_mem,
            last=last,
            frags=s.frags,
            deaths=s.deaths,
            dead=dead,
            events=tuple(events),
            counters=counters,
            priv=priv,
        )


# ── Numeric views for the RL teacher ───────────────────────────────────────────
_ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
_N_ENEMY, _N_ITEM = MAX_SEEN_ENEMIES, 4


def _polar(o: Seen | None) -> list[float]:
    """present, bearing/180, sin, cos, dist/32 -- at the text's precision."""
    if o is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    r = math.radians(o.b)
    return [1.0, o.b / 180.0, math.sin(r), math.cos(r), min(o.d, 64) / 32.0]


def features(obs: Observation) -> np.ndarray:
    """The player-visible state as a fixed-length float32 vector.

    Holds exactly the information in ``obs.text`` (rounded the same way), so a
    policy trained on it can be imitated by one that reads the text.
    """
    f: list[float] = [obs.hp / 100.0, obs.armor / 100.0, min(obs.ammo, 200) / 100.0]
    r = math.radians(obs.face)
    f += [obs.pos[0] / 80.0, obs.pos[1] / 80.0, math.sin(r), math.cos(r)]
    f += [float(obs.slot == s) for s in WEAPON_SLOTS]
    f += [float(s in obs.arms) for s in WEAPON_SLOTS]
    f += [min(obs.arms.get(s, 0), 200) / 100.0 for s in WEAPON_SLOTS]
    foes = [o for o in obs.seen if o.kind == "enemy"]
    for i in range(_N_ENEMY):
        f += _polar(foes[i] if i < len(foes) else None)
    f += _polar(next((o for o in obs.seen if o.kind == "missile"), None))
    items = [o for o in obs.seen if o.kind in ITEM_KINDS]
    for i in range(_N_ITEM):
        o = items[i] if i < len(items) else None
        f += _polar(o)
        f += [float(o is not None and o.kind == k) for k in ITEM_KINDS]
        slot = 0
        if o is not None:
            slot = WEAPON_ITEM_SLOT.get(o.cls, AMMO_ITEM_SLOT.get(o.cls, 0))
        f.append(slot / 7.0)
    f += [w / WALL_CAP_M for w in obs.walls]
    f.append(min(obs.hit, 100) / 50.0)
    if obs.enemy_mem is None:
        f += [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        b, secs = obs.enemy_mem
        r = math.radians(b)
        f += [1.0, b / 180.0, math.sin(r), math.cos(r), secs / 5.0]
    for a in obs.last:
        onehot = [0.0] * len(ACTIONS)
        onehot[_ACTION_INDEX[a]] = 1.0
        f += onehot
    return np.asarray(f, dtype=np.float32)


FEATURE_DIM = (
    3
    + 4
    + 3 * len(WEAPON_SLOTS)
    + 5 * _N_ENEMY
    + 5
    + _N_ITEM * (5 + len(ITEM_KINDS) + 1)
    + 4
    + 1
    + 5
    + 2 * len(ACTIONS)
)
_N_PRIV_ENEMY = 8


def priv_features(obs: Observation, match_frac: float) -> np.ndarray:
    """What the asymmetric critic sees on top of :func:`features`: every bot's
    position relative to the player (nearest first), the frag race, and match
    time. ViZDoom does not expose bot health, so it is not here."""
    p = obs.priv
    rel = []
    for ex, ey in p.enemies:
        dx, dy = ex - p.x, ey - p.y
        b = -_wrap180(math.degrees(math.atan2(dy, dx)) - p.angle)
        rel.append((math.hypot(dx, dy) / UNITS_PER_M, b))
    rel.sort()
    f: list[float] = []
    for i in range(_N_PRIV_ENEMY):
        if i < len(rel):
            d, b = rel[i]
            r = math.radians(b)
            f += [1.0, b / 180.0, math.sin(r), math.cos(r), min(d, 64.0) / 32.0]
        else:
            f += [0.0] * 5
    others = sorted((fr for _, fr in p.scoreboard[1:]), reverse=True)
    best = others[0] if others else 0
    f += [
        (obs.frags - best) / 10.0,
        obs.frags / 20.0,
        obs.deaths / 20.0,
        float(obs.dead),
        match_frac,
    ]
    return np.asarray(f, dtype=np.float32)


PRIV_DIM = 5 * _N_PRIV_ENEMY + 5
