"""
Tennis Rotation Visualizer — 3-player rotation with realistic simulation.

Rules:
1. Player 1 and Player 2 start on court; Player 3 rests on the bench.
2. After every match the loser swaps with the bench player.
3. If a player wins 3 consecutive matches they are forced to the bench to cool off.
4. Each match is first-to-3 points.
5. Realistic tennis simulation with player profiles, momentum, fatigue, and
   shot-response logic.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import flet as ft
except ModuleNotFoundError as exc:
    raise SystemExit(
        "The 'flet' package is required. Install via 'pip install flet'."
    ) from exc


# ────────────────────────── helpers ──────────────────────────

def _anim(duration_ms: int, curve: str):
    cls = getattr(ft, "Animation", None)
    if cls:
        return cls(duration_ms, curve)
    mod = getattr(ft, "animation", None)
    if mod and hasattr(mod, "Animation"):
        return mod.Animation(duration_ms, curve)
    return None


def _shadow(**kw):
    cls = getattr(ft, "BoxShadow", None)
    return cls(**kw) if cls else None


# ────────────────────────── data models ──────────────────────────

PLAYER_COLORS = ("#4FC3F7", "#81C784", "#FFD54F")  # blue, green, gold


@dataclass
class PlayerProfile:
    """Unique skill attributes per player."""
    name: str
    color: str
    forehand: float = 0.5
    backhand: float = 0.5
    net_game: float = 0.5
    consistency: float = 0.5
    speed: float = 0.5
    serve: float = 0.5


DEFAULT_PROFILES = [
    PlayerProfile("Player 1", PLAYER_COLORS[0], forehand=0.7, backhand=0.4, net_game=0.3, consistency=0.6, speed=0.5, serve=0.6),
    PlayerProfile("Player 2", PLAYER_COLORS[1], forehand=0.5, backhand=0.6, net_game=0.7, consistency=0.5, speed=0.6, serve=0.4),
    PlayerProfile("Player 3", PLAYER_COLORS[2], forehand=0.4, backhand=0.5, net_game=0.5, consistency=0.7, speed=0.7, serve=0.5),
]


@dataclass
class Shot:
    name: str
    target_x: float
    target_y: float
    speed: float
    base_winner_chance: float
    category: str = "baseline"  # baseline | net | approach | serve


SHOT_LIBRARY = [
    Shot("cross_court_fh", 0.8, 0.9, 0.8, 0.05, "baseline"),
    Shot("cross_court_bh", 0.2, 0.9, 0.8, 0.05, "baseline"),
    Shot("dtl_fh", 0.9, 0.95, 0.7, 0.08, "baseline"),
    Shot("dtl_bh", 0.1, 0.95, 0.7, 0.08, "baseline"),
    Shot("deep_baseline", 0.5, 0.98, 0.85, 0.02, "baseline"),
    Shot("wide_fh", 0.95, 0.8, 0.75, 0.12, "baseline"),
    Shot("wide_bh", 0.05, 0.8, 0.75, 0.12, "baseline"),
    Shot("drop_shot", 0.5, 0.3, 1.0, 0.15, "net"),
    Shot("volley_l", 0.3, 0.2, 0.4, 0.20, "net"),
    Shot("volley_r", 0.7, 0.2, 0.4, 0.20, "net"),
    Shot("approach", 0.5, 0.5, 0.6, 0.03, "approach"),
    Shot("lob", 0.5, 0.95, 1.1, 0.10, "baseline"),
    Shot("passing_l", 0.05, 0.85, 0.5, 0.25, "baseline"),
    Shot("passing_r", 0.95, 0.85, 0.5, 0.25, "baseline"),
]

RESPONSES: Dict[str, List[str]] = {
    "net": ["passing_l", "passing_r", "lob"],
    "approach": ["passing_l", "passing_r", "dtl_fh", "dtl_bh"],
    "baseline": [],  # no special response
}


@dataclass
class MatchSnapshot:
    winner: str
    loser: str
    score: Tuple[int, int]
    active_after: Tuple[str, str]
    bench_after: str
    forced_bench: bool
    streak_after: int


@dataclass
class PlayerStats:
    wins: int = 0
    losses: int = 0
    current_streak: int = 0


# ────────────────────────── rotation logic ──────────────────────────

class TennisRotation:
    def __init__(self, profiles: List[PlayerProfile], max_streak: int = 3) -> None:
        self._profiles = {p.name: p for p in profiles}
        self._names = [p.name for p in profiles]
        self._active: List[str] = [self._names[0], self._names[1]]
        self._bench: str = self._names[2]
        self._streaks = {n: 0 for n in self._names}
        self._history: List[MatchSnapshot] = []
        self._stats: Dict[str, PlayerStats] = {n: PlayerStats() for n in self._names}
        self._max_streak = max_streak

    @property
    def current_players(self):
        return tuple(self._active)

    @property
    def bench_player(self):
        return self._bench

    @property
    def history(self):
        return list(self._history)

    @property
    def names(self):
        return list(self._names)

    @property
    def stats(self):
        return dict(self._stats)

    def profile(self, name: str) -> PlayerProfile:
        return self._profiles[name]

    def record_match(self, winner: str, score: Tuple[int, int]) -> MatchSnapshot:
        loser = self._active[0] if self._active[1] == winner else self._active[1]
        bench_before = self._bench
        self._streaks[winner] += 1
        self._streaks[loser] = 0
        self._stats[winner].wins += 1
        self._stats[winner].current_streak = self._streaks[winner]
        self._stats[loser].losses += 1
        self._stats[loser].current_streak = 0
        self._active = [winner, bench_before]
        self._bench = loser

        forced_bench = False
        if self._streaks[winner] >= self._max_streak:
            forced_bench = True
            self._streaks[winner] = 0
            self._stats[winner].current_streak = 0
            returning = self._bench
            self._active = [bench_before, returning]
            self._bench = winner

        snap = MatchSnapshot(
            winner=winner, loser=loser, score=score,
            active_after=tuple(self._active), bench_after=self._bench,
            forced_bench=forced_bench, streak_after=self._streaks.get(winner, 0),
        )
        self._history.append(snap)
        return snap

    def reset(self, profiles: List[PlayerProfile]):
        self.__init__(profiles, self._max_streak)


# ────────────────────────── visualizer ──────────────────────────

class RotationVisualizer:
    _POINTS_TO_WIN = 3

    def __init__(self, rotation: TennisRotation, interval_sec: float = 0.4) -> None:
        self.rotation = rotation
        self.interval_sec = interval_sec
        self._match_count = 0
        self._paused = False
        self._speed = 1.0

        self.page: Optional[ft.Page] = None
        self.ball: Optional[ft.Container] = None
        self._root_view: Optional[ft.Column] = None

        # Dimensions
        self._base_w, self._base_h = 300, 400
        self._scale = 1.0
        self.court_w = self._base_w
        self.court_h = self._base_h
        self._marker_sz = 24
        self._ball_r = 7
        self._pad = 16
        self._alley = 25

        # Derived
        self._net_y = self.court_h / 2
        self._play_l = self._pad + self._alley + 8
        self._play_r = self.court_w - self._pad - self._alley - 8
        self._play_w = self._play_r - self._play_l
        self._near_base = self.court_h - 45
        self._near_net = self._net_y + 35
        self._far_base = 45
        self._far_net = self._net_y - 35
        self._bench_left = self.court_w + 30
        self._bench_top = self.court_h / 2 - 50
        self._bench_slot = (self._bench_left + 20, self._bench_top + 45)

        # State
        self._positions: Dict[str, Tuple[float, float]] = {}
        self._markers: Dict[str, ft.Container] = {}
        self._labels: Dict[str, ft.Text] = {}
        self._side: Dict[str, str] = {}
        self._score_dots: Dict[str, List[ft.Container]] = {}
        self._momentum: Dict[str, float] = {}
        self._fatigue: Dict[str, float] = {}
        self._prev_shot_cat: str = "baseline"

        # Animations
        self._anim_ball = _anim(500, "easeOutCubic")
        self._anim_player = _anim(450, "easeOutQuad")
        self._anim_fade = _anim(120, "linear")
        self._ball_shadow = _shadow(spread_radius=2, blur_radius=6, color="#CCFF0066")
        self._base_shot_time = 0.55

        # UI refs
        self._score_text: Optional[ft.Text] = None
        self._match_label: Optional[ft.Text] = None
        self._bench_name: Optional[ft.Text] = None
        self._bench_reason: Optional[ft.Text] = None
        self._history_col: Optional[ft.Column] = None
        self._stats_texts: Dict[str, ft.Text] = {}
        self._pause_btn: Optional[ft.IconButton] = None
        self._speed_label: Optional[ft.Text] = None

    # ── scaling ──
    def _calc_dims(self, pw: float, ph: float):
        aw = pw - 400  # room for sidebar
        ah = ph - 140  # room for header/controls
        sw = aw / self._base_w
        sh = ah / self._base_h
        self._scale = max(0.55, min(sw, sh, 1.3))
        s = self._scale
        self.court_w = self._base_w * s
        self.court_h = self._base_h * s
        self._marker_sz = int(24 * s)
        self._ball_r = int(7 * s)
        self._pad = int(16 * s)
        self._alley = int(25 * s)
        self._net_y = self.court_h / 2
        self._play_l = self._pad + self._alley + int(8 * s)
        self._play_r = self.court_w - self._pad - self._alley - int(8 * s)
        self._play_w = self._play_r - self._play_l
        self._near_base = self.court_h - int(45 * s)
        self._near_net = self._net_y + int(35 * s)
        self._far_base = int(45 * s)
        self._far_net = self._net_y - int(35 * s)
        self._bench_left = self.court_w + int(20 * s)
        self._bench_top = self.court_h / 2 - int(50 * s)
        self._bench_slot = (self._bench_left + int(25 * s), self._bench_top + int(45 * s))

    def _court_pos(self, side: str, rx: float, ry: float):
        x = self._play_l + rx * self._play_w
        if side == "near":
            y = self._near_base - ry * (self._near_base - self._near_net)
        else:
            y = self._far_base + ry * (self._far_net - self._far_base)
        return (x, y)

    def _ball_target(self, side: str, shot: Shot):
        ts = "far" if side == "near" else "near"
        return self._court_pos(ts, shot.target_x, shot.target_y)

    # ── start ──
    async def start(self, page: ft.Page):
        self.page = page
        page.title = "🎾 Tennis Rotation Visualizer"
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.vertical_alignment = ft.MainAxisAlignment.START
        page.bgcolor = "#0A0E1A"
        page.padding = 0
        page.theme_mode = ft.ThemeMode.DARK
        page.fonts = {"mono": "monospace"}

        w = page.width or 800
        h = page.height or 700
        self._calc_dims(w, h)

        for n in self.rotation.names:
            self._momentum[n] = 0.5
            self._fatigue[n] = 0.0

        self._root_view = self._build_full_ui()
        page.add(self._root_view)
        await self._init_players(self.rotation.current_players, self.rotation.bench_player, reason="")
        if hasattr(page, "run_task"):
            page.run_task(self._loop)
        else:
            asyncio.create_task(self._loop())

    # ── main UI layout ──
    def _build_full_ui(self) -> ft.Column:
        header = self._build_header()
        controls = self._build_controls()
        court_and_sidebar = self._build_court_and_sidebar()
        return ft.Column(
            controls=[header, controls, court_and_sidebar],
            spacing=0,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _build_header(self) -> ft.Container:
        return ft.Container(
            content=ft.Column([
                ft.Text("🎾 Tennis Rotation Visualizer",
                         size=20, weight=ft.FontWeight.BOLD, color="#FFFFFF",
                         text_align=ft.TextAlign.CENTER),
                ft.Text("3-player rotation  •  Winner stays on  •  Forced rest after 3-win streak",
                         size=11, color="#888888", text_align=ft.TextAlign.CENTER),
            ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=ft.padding.only(top=16, bottom=8, left=16, right=16),
        )

    def _build_controls(self) -> ft.Container:
        self._pause_btn = ft.IconButton(
            icon=ft.Icons.PAUSE_CIRCLE_FILLED,
            icon_color="#FFFFFF",
            icon_size=28,
            tooltip="Pause/Resume",
            on_click=self._toggle_pause,
        )
        self._speed_label = ft.Text("1x", size=12, color="#CCCCCC", weight=ft.FontWeight.BOLD)
        speed_down = ft.IconButton(icon=ft.Icons.REMOVE, icon_color="#888888", icon_size=18,
                                   tooltip="Slower", on_click=self._speed_down)
        speed_up = ft.IconButton(icon=ft.Icons.ADD, icon_color="#888888", icon_size=18,
                                 tooltip="Faster", on_click=self._speed_up)
        reset_btn = ft.TextButton("Reset", on_click=self._reset,
                                   style=ft.ButtonStyle(color="#FF8A80"))

        return ft.Container(
            content=ft.Row([
                self._pause_btn,
                ft.Container(width=1, height=24, bgcolor="#333333"),
                speed_down, self._speed_label, speed_up,
                ft.Container(width=1, height=24, bgcolor="#333333"),
                reset_btn,
            ], alignment=ft.MainAxisAlignment.CENTER, spacing=4),
            bgcolor="#111827",
            border=ft.border.only(bottom=ft.BorderSide(1, "#1F2937")),
            padding=ft.padding.symmetric(vertical=6, horizontal=16),
        )

    def _build_court_and_sidebar(self) -> ft.Container:
        court_stack = self._build_court()
        sidebar = self._build_sidebar()
        return ft.Container(
            content=ft.Row(
                controls=[court_stack, sidebar],
                alignment=ft.MainAxisAlignment.CENTER,
                vertical_alignment=ft.CrossAxisAlignment.START,
                spacing=0,
            ),
            padding=ft.padding.only(top=8, left=8, right=8, bottom=16),
        )

    # ── court drawing ──
    def _build_court(self) -> ft.Container:
        els: List[ft.Control] = []
        s = self._scale

        # Court surface — dark blue-green
        els.append(ft.Container(left=0, top=0, width=self.court_w, height=self.court_h,
                                bgcolor="#0D3B4F", border_radius=ft.border_radius.all(6)))
        # Inner play area
        inner_l, inner_t = self._pad, self._pad
        inner_w = self.court_w - 2 * self._pad
        inner_h = self.court_h - 2 * self._pad
        els.append(ft.Container(left=inner_l, top=inner_t, width=inner_w, height=inner_h,
                                bgcolor="#0F4A60", border=ft.border.all(2, "#FFFFFFCC")))

        # Singles sidelines
        sl = self._pad + self._alley
        sr = self.court_w - self._pad - self._alley
        sw = sr - sl
        els.append(ft.Container(left=sl, top=self._pad, width=2, height=inner_h, bgcolor="#FFFFFFAA"))
        els.append(ft.Container(left=sr - 2, top=self._pad, width=2, height=inner_h, bgcolor="#FFFFFFAA"))

        # Net
        no = int(8 * s)
        nh = max(4, int(5 * s))
        els.append(ft.Container(left=self._pad - no, top=self._net_y - nh / 2,
                                width=inner_w + 2 * no, height=nh, bgcolor="#FFFFFFDD"))
        # Net posts
        ps = max(6, int(7 * s))
        els.append(ft.Container(left=self._pad - no - ps / 2, top=self._net_y - ps,
                                width=ps, height=ps * 2, bgcolor="#FFFFFFCC",
                                border_radius=ft.border_radius.all(2)))
        els.append(ft.Container(left=self.court_w - self._pad + no - ps / 2, top=self._net_y - ps,
                                width=ps, height=ps * 2, bgcolor="#FFFFFFCC",
                                border_radius=ft.border_radius.all(2)))

        # Service boxes
        sb = int(85 * s)
        cx = self.court_w / 2
        els.append(ft.Container(left=sl, top=self._pad + sb, width=sw, height=2, bgcolor="#FFFFFF99"))
        els.append(ft.Container(left=sl, top=self.court_h - self._pad - sb - 2, width=sw, height=2, bgcolor="#FFFFFF99"))
        els.append(ft.Container(left=cx - 1, top=self._pad + sb, width=2,
                                height=self._net_y - self._pad - sb, bgcolor="#FFFFFF99"))
        els.append(ft.Container(left=cx - 1, top=self._net_y, width=2,
                                height=self.court_h - self._net_y - self._pad - sb, bgcolor="#FFFFFF99"))

        # Center marks
        cm = int(10 * s)
        els.append(ft.Container(left=cx - 1, top=self._pad, width=2, height=cm, bgcolor="#FFFFFF99"))
        els.append(ft.Container(left=cx - 1, top=self.court_h - self._pad - cm, width=2, height=cm, bgcolor="#FFFFFF99"))

        # Score dots
        near_sy = self.court_h - int(25 * s)
        far_sy = int(25 * s)
        ds = int(14 * s)
        dd = int(9 * s)
        for slot, sy in [("near", near_sy), ("far", far_sy)]:
            dots = []
            for i in range(self._POINTS_TO_WIN):
                d = ft.Container(
                    left=cx - (self._POINTS_TO_WIN * ds) / 2 + i * ds,
                    top=sy - dd / 2, width=dd, height=dd,
                    bgcolor="#1A1A1A66", border=ft.border.all(1, "#FFFFFF33"),
                    border_radius=ft.border_radius.all(dd / 2),
                )
                dots.append(d)
                els.append(d)
            self._score_dots[slot] = dots

        # Bench area
        bw = int(70 * s)
        bh = int(110 * s)
        els.append(ft.Container(left=self._bench_left, top=self._bench_top,
                                width=bw, height=bh, bgcolor="#0A0E1ACC",
                                border=ft.border.all(1, "#1F2937"),
                                border_radius=ft.border_radius.all(8)))
        # Bench label
        els.append(ft.Container(
            left=self._bench_left, top=self._bench_top + int(3 * s), width=bw,
            content=ft.Text("🪑 BENCH", size=max(8, int(9 * s)), color="#666666",
                            text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD),
            alignment=ft.Alignment(0, 0),
        ))
        # Bench player name
        self._bench_name = ft.Text("", size=max(7, int(8 * s)), color="#AAAAAA",
                                    text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD)
        els.append(ft.Container(
            left=self._bench_left, top=self._bench_top + int(78 * s), width=bw,
            content=self._bench_name, alignment=ft.Alignment(0, 0),
        ))
        # Bench reason
        self._bench_reason = ft.Text("", size=max(6, int(7 * s)), color="#888888",
                                      text_align=ft.TextAlign.CENTER, italic=True)
        els.append(ft.Container(
            left=self._bench_left, top=self._bench_top + int(93 * s), width=bw,
            content=self._bench_reason, alignment=ft.Alignment(0, 0),
        ))

        # Ball (tennis-ball yellow)
        bkw = {}
        if self._anim_ball:
            bkw["animate_position"] = self._anim_ball
        if self._anim_fade:
            bkw["animate_opacity"] = self._anim_fade
        if self._ball_shadow:
            bkw["shadow"] = self._ball_shadow
        self.ball = ft.Container(
            left=-50, top=-50, width=self._ball_r * 2, height=self._ball_r * 2,
            bgcolor="#CCFF00", border_radius=ft.border_radius.all(self._ball_r),
            opacity=0, **bkw,
        )
        els.append(self.ball)

        # Player markers + labels
        for i, name in enumerate(self.rotation.names):
            color = self.rotation.profile(name).color
            marker = self._make_marker(i, color)
            marker.left = -100
            marker.top = -100
            if self._anim_player:
                marker.animate_position = self._anim_player
            self._markers[name] = marker
            els.append(marker)

            label = ft.Text(name, size=max(7, int(8 * self._scale)), color=color,
                            text_align=ft.TextAlign.CENTER, weight=ft.FontWeight.BOLD)
            label_c = ft.Container(left=-100, top=-100, width=int(60 * self._scale),
                                    content=label, alignment=ft.Alignment(0, 0))
            if self._anim_player:
                label_c.animate_position = self._anim_player
            self._labels[name] = label_c
            els.append(label_c)

        total_w = self._bench_left + bw + int(10 * s)
        stack = ft.Stack(width=total_w, height=self.court_h, controls=els)
        return ft.Container(content=stack, padding=ft.padding.all(max(8, int(12 * s))))

    def _make_marker(self, idx: int, color: str) -> ft.Container:
        sz = self._marker_sz
        half = sz / 2
        bw = max(2, int(3 * self._scale))
        if idx == 0:  # circle
            return ft.Container(width=sz, height=sz, bgcolor=color,
                                border_radius=ft.border_radius.all(half))
        if idx == 1:  # square
            return ft.Container(width=sz, height=sz, bgcolor="#0D3B4F",
                                border=ft.border.all(bw, color))
        # triangle
        steps = 8
        rh = sz / steps
        bars = []
        for i in range(steps):
            w = (i + 1) / steps * sz
            bars.append(ft.Container(left=(sz - w) / 2, top=i * rh,
                                     width=w, height=rh + 1, bgcolor=color))
        return ft.Stack(width=sz, height=sz, controls=bars)

    # ── sidebar ──
    def _build_sidebar(self) -> ft.Container:
        s = self._scale
        panel_w = max(160, int(200 * s))

        # Scoreboard
        self._match_label = ft.Text("Match #1", size=11, color="#888888", weight=ft.FontWeight.BOLD)
        self._score_text = ft.Text("0 - 0", size=24, color="#FFFFFF", weight=ft.FontWeight.BOLD,
                                    text_align=ft.TextAlign.CENTER)
        scoreboard = ft.Container(
            content=ft.Column([
                ft.Text("SCOREBOARD", size=10, color="#666666", weight=ft.FontWeight.BOLD,
                         letter_spacing=2),
                self._match_label,
                ft.Container(height=4),
                self._score_text,
            ], spacing=2, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            bgcolor="#111827", border=ft.border.all(1, "#1F2937"),
            border_radius=ft.border_radius.all(10), padding=16, width=panel_w,
        )

        # Stats
        stats_rows = []
        for name in self.rotation.names:
            color = self.rotation.profile(name).color
            txt = ft.Text(f"{name}: W0 L0", size=10, color=color)
            self._stats_texts[name] = txt
            stats_rows.append(txt)

        stats_panel = ft.Container(
            content=ft.Column([
                ft.Text("PLAYER STATS", size=10, color="#666666", weight=ft.FontWeight.BOLD,
                         letter_spacing=2),
                *stats_rows,
            ], spacing=4),
            bgcolor="#111827", border=ft.border.all(1, "#1F2937"),
            border_radius=ft.border_radius.all(10), padding=16, width=panel_w,
        )

        # Match history
        self._history_col = ft.Column(spacing=2, scroll=ft.ScrollMode.AUTO)
        history_panel = ft.Container(
            content=ft.Column([
                ft.Text("MATCH HISTORY", size=10, color="#666666", weight=ft.FontWeight.BOLD,
                         letter_spacing=2),
                ft.Container(content=self._history_col, height=max(120, int(160 * s))),
            ], spacing=4),
            bgcolor="#111827", border=ft.border.all(1, "#1F2937"),
            border_radius=ft.border_radius.all(10), padding=16, width=panel_w,
        )

        # Legend
        legend_items = []
        shapes = ["●", "■", "▲"]
        for i, name in enumerate(self.rotation.names):
            color = self.rotation.profile(name).color
            legend_items.append(ft.Row([
                ft.Text(shapes[i], size=14, color=color),
                ft.Text(name, size=10, color=color),
            ], spacing=6))

        legend_panel = ft.Container(
            content=ft.Column([
                ft.Text("LEGEND", size=10, color="#666666", weight=ft.FontWeight.BOLD,
                         letter_spacing=2),
                *legend_items,
            ], spacing=4),
            bgcolor="#111827", border=ft.border.all(1, "#1F2937"),
            border_radius=ft.border_radius.all(10), padding=16, width=panel_w,
        )

        return ft.Container(
            content=ft.Column([scoreboard, stats_panel, history_panel, legend_panel],
                              spacing=8, horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                              scroll=ft.ScrollMode.AUTO),
            padding=ft.padding.only(top=max(8, int(12 * s)), right=8),
            height=self.court_h + max(16, int(24 * s)),
        )

    # ── controls ──
    def _toggle_pause(self, e):
        self._paused = not self._paused
        if self._pause_btn:
            self._pause_btn.icon = ft.Icons.PLAY_CIRCLE_FILLED if self._paused else ft.Icons.PAUSE_CIRCLE_FILLED
            self._pause_btn.icon_color = "#4FC3F7" if self._paused else "#FFFFFF"
        if self.page:
            self.page.update()

    def _speed_down(self, e):
        if self._speed > 0.25:
            self._speed = max(0.25, self._speed / 2)
        self._update_speed_label()

    def _speed_up(self, e):
        if self._speed < 4:
            self._speed = min(4, self._speed * 2)
        self._update_speed_label()

    def _update_speed_label(self):
        if self._speed_label:
            if self._speed >= 1:
                self._speed_label.value = f"{int(self._speed)}x"
            else:
                self._speed_label.value = f"{self._speed:.2g}x"
        if self.page:
            self.page.update()

    def _reset(self, e):
        profiles = [self.rotation.profile(n) for n in self.rotation.names]
        self.rotation.reset(profiles)
        self._match_count = 0
        self._momentum = {n: 0.5 for n in self.rotation.names}
        self._fatigue = {n: 0.0 for n in self.rotation.names}
        if self._history_col:
            self._history_col.controls.clear()
        for n in self.rotation.names:
            s = self.rotation.stats[n]
            self._stats_texts[n].value = f"{n}: W{s.wins} L{s.losses}"
        if self._score_text:
            self._score_text.value = "0 - 0"
        if self._match_label:
            self._match_label.value = "Match #1"
        if self.page:
            self.page.update()

    # ── sleep helper ──
    async def _sleep(self, base_sec: float):
        await asyncio.sleep(base_sec / self._speed)

    # ── game loop ──
    async def _loop(self):
        while True:
            while self._paused:
                await asyncio.sleep(0.1)
            await self._play_match()
            await self._sleep(self.interval_sec)

    async def _play_match(self):
        active = self.rotation.current_players
        scores = {active[0]: 0, active[1]: 0}
        self._match_count += 1

        if self._match_label:
            self._match_label.value = f"Match #{self._match_count}"
        p0_col = self.rotation.profile(active[0]).color
        p1_col = self.rotation.profile(active[1]).color
        await self._update_score_ui(scores, active)

        while max(scores.values()) < self._POINTS_TO_WIN:
            while self._paused:
                await asyncio.sleep(0.1)
            winner = await self._play_point(active)
            scores[winner] += 1
            # Update momentum
            loser = active[0] if winner == active[1] else active[1]
            self._momentum[winner] = min(1.0, self._momentum.get(winner, 0.5) + 0.1)
            self._momentum[loser] = max(0.0, self._momentum.get(loser, 0.5) - 0.05)
            await self._update_score_ui(scores, active)
            await self._sleep(0.5)

        match_winner = active[0] if scores[active[0]] >= self._POINTS_TO_WIN else active[1]
        match_loser = active[0] if match_winner == active[1] else active[1]
        w_score = scores[match_winner]
        l_score = scores[match_loser]

        await self._flash_winner(match_winner, active)

        snap = self.rotation.record_match(match_winner, (w_score, l_score))

        # Update fatigue
        self._fatigue[snap.bench_after] = max(0.0, self._fatigue.get(snap.bench_after, 0) - 0.3)
        for p in snap.active_after:
            self._fatigue[p] = min(1.0, self._fatigue.get(p, 0) + 0.05)

        # Update UI
        self._update_stats_ui()
        self._add_history_entry(snap)

        bench_reason = "🔥 Cooling off!" if snap.forced_bench else "Lost match"
        await self._reset_score_dots()
        await self._init_players(snap.active_after, snap.bench_after, reason=bench_reason)
        await self._sleep(1.5)

    async def _play_point(self, active) -> str:
        server_idx = random.randint(0, 1)
        server = active[server_idx]
        receiver = active[1 - server_idx]
        s_side = self._side[server]
        r_side = self._side[receiver]

        # Serve positioning
        serve_pos = self._court_pos(s_side, 0.5, 0.0)
        recv_pos = self._court_pos(r_side, 0.5, 0.1)
        self._set_pos(server, serve_pos)
        self._set_pos(receiver, recv_pos)
        await self._page_update()
        await self._sleep(0.35)

        # Ball at server
        if self.ball:
            self.ball.left = serve_pos[0] - self._ball_r
            self.ball.top = serve_pos[1] - self._ball_r
            self.ball.opacity = 1
            await self._page_update()
            await self._sleep(0.25)

        # Ace chance based on serve skill
        s_prof = self.rotation.profile(server)
        if random.random() < s_prof.serve * 0.08:
            if self.ball:
                target = self._court_pos(r_side, random.choice([0.2, 0.8]), 0.95)
                self.ball.left = target[0] - self._ball_r
                self.ball.top = target[1] - self._ball_r
                await self._page_update()
                await self._sleep(0.3)
                self.ball.opacity = 0
                await self._page_update()
            return server

        current = server
        cur_side = s_side
        self._prev_shot_cat = "serve"
        max_rally = 50  # safety cap

        for rally_i in range(max_rally):
            while self._paused:
                await asyncio.sleep(0.1)

            prof = self.rotation.profile(current)
            opponent = active[0] if current == active[1] else active[1]
            opp_side = self._side[opponent]
            opp_prof = self.rotation.profile(opponent)

            h_pos = self._positions.get(current, (0, 0))
            shot = self._choose_smart_shot(cur_side, h_pos, prof, self._prev_shot_cat)
            target = self._ball_target(cur_side, shot)

            retrieve = self._get_retrieve(opp_side, target, opp_prof)
            recover = self._get_recovery(cur_side, shot)

            if self.ball:
                self.ball.left = target[0] - self._ball_r
                self.ball.top = target[1] - self._ball_r
            self._set_pos(opponent, retrieve)
            self._set_pos(current, recover)
            await self._page_update()
            await self._sleep(self._base_shot_time * shot.speed)

            # Winner chance (boosted by skill + momentum)
            skill_mult = self._skill_for_shot(prof, shot)
            mom_bonus = (self._momentum.get(current, 0.5) - 0.5) * 0.15
            fat_penalty = self._fatigue.get(current, 0) * 0.05
            winner_chance = shot.base_winner_chance * (1 + skill_mult + mom_bonus - fat_penalty)
            if random.random() < winner_chance:
                if self.ball:
                    self.ball.opacity = 0
                    await self._page_update()
                self._fatigue[current] = self._fatigue.get(current, 0) + 0.01 * (rally_i + 1)
                self._fatigue[opponent] = self._fatigue.get(opponent, 0) + 0.01 * (rally_i + 1)
                return current

            # Unforced error (affected by consistency, fatigue)
            error_base = 0.04
            consistency_mod = (1 - opp_prof.consistency) * 0.06
            fatigue_mod = self._fatigue.get(opponent, 0) * 0.08
            error_chance = error_base + consistency_mod + fatigue_mod
            if random.random() < error_chance:
                if self.ball:
                    self.ball.opacity = 0
                    await self._page_update()
                self._fatigue[current] = self._fatigue.get(current, 0) + 0.01 * (rally_i + 1)
                self._fatigue[opponent] = self._fatigue.get(opponent, 0) + 0.01 * (rally_i + 1)
                return current

            # Failed retrieve (speed-based)
            dist = math.sqrt((target[0] - self._positions.get(opponent, target)[0]) ** 2 +
                             (target[1] - self._positions.get(opponent, target)[1]) ** 2)
            max_reach = (50 + opp_prof.speed * 40) * self._scale
            if dist > max_reach and random.random() < 0.3:
                if self.ball:
                    self.ball.opacity = 0
                    await self._page_update()
                return current

            self._prev_shot_cat = shot.category
            current = opponent
            cur_side = opp_side

        # Safety: rally too long
        if self.ball:
            self.ball.opacity = 0
            await self._page_update()
        return random.choice(list(active))

    def _skill_for_shot(self, prof: PlayerProfile, shot: Shot) -> float:
        if "fh" in shot.name or "forehand" in shot.name:
            return prof.forehand * 0.3
        if "bh" in shot.name or "backhand" in shot.name:
            return prof.backhand * 0.3
        if shot.category == "net":
            return prof.net_game * 0.4
        return (prof.forehand + prof.backhand) / 2 * 0.2

    def _choose_smart_shot(self, side: str, pos, prof: PlayerProfile, prev_cat: str) -> Shot:
        # Respond to previous shot category
        responses = RESPONSES.get(prev_cat, [])
        if responses and random.random() < 0.5:
            matching = [s for s in SHOT_LIBRARY if s.name in responses]
            if matching:
                return random.choice(matching)

        rel_y = pos[1] / self.court_h
        at_net = (side == "near" and rel_y < 0.6) or (side == "far" and rel_y > 0.4)

        if at_net and prof.net_game > 0.4:
            net_shots = [s for s in SHOT_LIBRARY if s.category == "net"]
            if net_shots and random.random() < 0.4 + prof.net_game * 0.3:
                return random.choice(net_shots)

        # Approach shot if mid-court
        mid_court = (side == "near" and 0.4 < rel_y < 0.65) or (side == "far" and 0.35 < rel_y < 0.6)
        if mid_court and random.random() < 0.2:
            approaches = [s for s in SHOT_LIBRARY if s.category == "approach"]
            if approaches:
                return random.choice(approaches)

        # Position-based shot selection
        rel_x = (pos[0] - self._play_l) / self._play_w if self._play_w > 0 else 0.5
        if rel_x < 0.3:
            preferred = [s for s in SHOT_LIBRARY if s.target_x > 0.6 and s.category == "baseline"]
            if preferred and random.random() < 0.5:
                return random.choice(preferred)
        elif rel_x > 0.7:
            preferred = [s for s in SHOT_LIBRARY if s.target_x < 0.4 and s.category == "baseline"]
            if preferred and random.random() < 0.5:
                return random.choice(preferred)

        baseline = [s for s in SHOT_LIBRARY if s.category == "baseline"]
        return random.choice(baseline) if baseline else random.choice(SHOT_LIBRARY)

    def _get_retrieve(self, side: str, target, prof: PlayerProfile):
        tx = max(self._play_l, min(self._play_r, target[0]))
        speed_bonus = prof.speed * 10 * self._scale
        if side == "near":
            ty = self._near_base if target[1] > self._net_y + 80 * self._scale else max(self._near_net, target[1] + 30 * self._scale - speed_bonus)
        else:
            ty = self._far_base if target[1] < self._net_y - 80 * self._scale else min(self._far_net, target[1] - 30 * self._scale + speed_bonus)
        return (tx, ty)

    def _get_recovery(self, side: str, shot: Shot):
        cx = self._play_l + self._play_w * 0.5
        rx = cx - 15 * self._scale if shot.target_x > 0.5 else cx + 15 * self._scale
        if side == "near":
            ry = self._near_net + 15 * self._scale if shot.category in ("approach", "net") else self._near_base - 8 * self._scale
        else:
            ry = self._far_net - 15 * self._scale if shot.category in ("approach", "net") else self._far_base + 8 * self._scale
        return (rx, ry)

    def _set_pos(self, name: str, pos):
        self._positions[name] = pos
        m = self._markers.get(name)
        if m:
            half = self._marker_sz / 2
            m.left = pos[0] - half
            m.top = pos[1] - half
        lc = self._labels.get(name)
        if lc:
            lw = int(60 * self._scale)
            lc.left = pos[0] - lw / 2
            lc.top = pos[1] + self._marker_sz / 2 + 2

    async def _init_players(self, active, bench: str, reason: str = ""):
        if not self.page:
            return
        self._side.clear()
        self._positions.clear()
        half = self._marker_sz / 2

        near = self._court_pos("near", 0.5, 0.0)
        self._side[active[0]] = "near"
        self._set_pos(active[0], near)
        m0 = self._markers[active[0]]
        m0.left = near[0] - half
        m0.top = near[1] - half
        m0.opacity = 1

        far = self._court_pos("far", 0.5, 0.0)
        self._side[active[1]] = "far"
        self._set_pos(active[1], far)
        m1 = self._markers[active[1]]
        m1.left = far[0] - half
        m1.top = far[1] - half
        m1.opacity = 1

        bm = self._markers[bench]
        bm.left = self._bench_slot[0] - half
        bm.top = self._bench_slot[1] - half
        bm.opacity = 0.6

        bl = self._labels.get(bench)
        if bl:
            lw = int(60 * self._scale)
            bl.left = self._bench_slot[0] - lw / 2
            bl.top = self._bench_slot[1] + self._marker_sz / 2 + 2

        if self._bench_name:
            self._bench_name.value = bench
            self._bench_name.color = self.rotation.profile(bench).color
        if self._bench_reason:
            self._bench_reason.value = reason

        # Update scoreboard player names for new match
        if self._score_text:
            self._score_text.value = f"0 - 0"
        if self._match_label:
            c0 = self.rotation.profile(active[0]).color
            c1 = self.rotation.profile(active[1]).color
            self._match_label.value = f"Match #{self._match_count}"

        await self._page_update()

    async def _update_score_ui(self, scores, active):
        if not self.page:
            return
        s0, s1 = scores.get(active[0], 0), scores.get(active[1], 0)
        if self._score_text:
            self._score_text.value = f"{s0} - {s1}"

        for idx, slot in enumerate(("near", "far")):
            p = active[idx]
            ps = scores.get(p, 0)
            color = self.rotation.profile(p).color
            for i, dot in enumerate(self._score_dots.get(slot, [])):
                if i < ps:
                    dot.bgcolor = color
                    dot.border = ft.border.all(1, color)
                else:
                    dot.bgcolor = "#1A1A1A66"
                    dot.border = ft.border.all(1, "#FFFFFF33")
        await self._page_update()

    async def _flash_winner(self, winner: str, active):
        if not self.page:
            return
        slot = "near" if active[0] == winner else "far"
        dots = self._score_dots.get(slot, [])
        for _ in range(3):
            for d in dots:
                d.opacity = 0.3
            await self._page_update()
            await self._sleep(0.08)
            for d in dots:
                d.opacity = 1
            await self._page_update()
            await self._sleep(0.08)

    async def _reset_score_dots(self):
        if not self.page:
            return
        for slot in ("near", "far"):
            for d in self._score_dots.get(slot, []):
                d.bgcolor = "#1A1A1A66"
                d.border = ft.border.all(1, "#FFFFFF33")
                d.opacity = 1
        await self._page_update()

    def _update_stats_ui(self):
        for name in self.rotation.names:
            s = self.rotation.stats[name]
            streak = ""
            if s.current_streak >= 2:
                streak = f" 🔥{s.current_streak}"
            self._stats_texts[name].value = f"{name}: W{s.wins} L{s.losses}{streak}"

    def _add_history_entry(self, snap: MatchSnapshot):
        if not self._history_col:
            return
        color = self.rotation.profile(snap.winner).color
        forced = " 🔥" if snap.forced_bench else ""
        txt = f"✅ {snap.winner} beat {snap.loser} ({snap.score[0]}-{snap.score[1]}){forced}"
        self._history_col.controls.insert(0,
            ft.Text(txt, size=9, color=color))
        # Keep last 15
        if len(self._history_col.controls) > 15:
            self._history_col.controls.pop()

    async def _page_update(self):
        if not self.page:
            return
        au = getattr(self.page, "update_async", None)
        if au:
            await au()
        else:
            self.page.update()


# ────────────────────────── entry point ──────────────────────────

async def main(page: ft.Page):
    rotation = TennisRotation(list(DEFAULT_PROFILES))
    vis = RotationVisualizer(rotation, interval_sec=0.4)
    await vis.start(page)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Tennis Rotation on http://0.0.0.0:{port}")
    ft.app(
        target=main,
        host="0.0.0.0",
        port=port,
        view=None,
    )
