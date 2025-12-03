"""
Simple rotation model for three tennis players.

Rules implemented:
1. Player 1 and Player 2 start on court while Player 3 waits on the bench.
2. After every match the loser leaves the court and swaps with the bench player.
3. If a player reaches the configured consecutive-win streak (default: 3),
   they immediately trade places with whoever is resting on the bench so that
   they can cool off. This can result in the most recent loser returning right
   away because both rotation rules are applied in sequence.
4. Each match requires 3 points to win.
5. Realistic tennis simulation with various shots and player movement.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import flet as ft
except ModuleNotFoundError as exc:  # pragma: no cover - config guard
    raise SystemExit(
        "The 'flet' package is required for the visualizer. "
        "Install it via 'pip install flet' and rerun this script."
    ) from exc


def _make_animation(duration_ms: int, curve: str):
    """Compatibility helper to obtain an Animation object across Flet versions."""
    animation_cls = getattr(ft, "Animation", None)
    if animation_cls:
        return animation_cls(duration_ms, curve)
    animation_module = getattr(ft, "animation", None)
    if animation_module and hasattr(animation_module, "Animation"):
        return animation_module.Animation(duration_ms, curve)
    return None


def _make_shadow(**kwargs):
    """Return a BoxShadow if supported, otherwise None."""
    box_shadow_cls = getattr(ft, "BoxShadow", None)
    if box_shadow_cls:
        return box_shadow_cls(**kwargs)
    return None


@dataclass
class MatchSnapshot:
    """Immutable snapshot describing the system after a completed match."""
    winner: str
    loser: str
    active_after: Tuple[str, str]
    bench_after: str
    forced_bench: bool
    streak_after: int


@dataclass
class Shot:
    """Represents a tennis shot with target position and type."""
    name: str
    target_x: float  # relative position 0-1 (left to right)
    target_y: float  # relative position 0-1 (baseline to net)
    speed: float     # animation duration multiplier (lower = faster)
    is_winner_chance: float  # chance this shot wins the point outright


class TennisRotation:
    """Tracks which players are on court and who is on the bench."""

    def __init__(self, players: Sequence[str], max_streak: int = 3) -> None:
        if len(players) != 3:
            raise ValueError("Exactly three players are required.")
        if max_streak < 1:
            raise ValueError("max_streak must be at least 1.")

        self._players: Tuple[str, str, str] = (players[0], players[1], players[2])
        self._active: List[str] = [players[0], players[1]]
        self._bench: str = players[2]
        self._streaks = {player: 0 for player in players}
        self._history: List[MatchSnapshot] = []
        self._max_streak = max_streak

    @property
    def current_players(self) -> Tuple[str, str]:
        return tuple(self._active)

    @property
    def bench_player(self) -> str:
        return self._bench

    @property
    def history(self) -> Tuple[MatchSnapshot, ...]:
        return tuple(self._history)

    @property
    def players(self) -> Tuple[str, str, str]:
        return self._players

    def record_match(self, winner: str) -> MatchSnapshot:
        """Register the outcome of a match."""
        if winner not in self._active:
            raise ValueError(f"{winner!r} is not currently on court.")

        loser = self._active[0] if self._active[1] == winner else self._active[1]
        bench_before = self._bench

        self._streaks[winner] += 1
        self._streaks[loser] = 0

        self._active = [winner, bench_before]
        self._bench = loser

        forced_bench = False
        if self._streaks[winner] >= self._max_streak:
            forced_bench = True
            self._streaks[winner] = 0
            returning_player = self._bench
            self._active = [bench_before, returning_player]
            self._bench = winner

        snapshot = MatchSnapshot(
            winner=winner,
            loser=loser,
            active_after=self.current_players,
            bench_after=self._bench,
            forced_bench=forced_bench,
            streak_after=self._streaks[winner],
        )
        self._history.append(snapshot)
        return snapshot


class RotationVisualizer:
    """Flet-based visualizer with realistic tennis simulation."""

    _SHAPES = ("circle", "square", "triangle")
    _POINTS_TO_WIN = 3

    # Shot types - faster speeds for smoother play
    _SHOTS = [
        Shot("cross_court_forehand", 0.8, 0.9, 0.8, 0.05),
        Shot("cross_court_backhand", 0.2, 0.9, 0.8, 0.05),
        Shot("down_the_line", 0.1, 0.95, 0.7, 0.08),
        Shot("down_the_line_r", 0.9, 0.95, 0.7, 0.08),
        Shot("drop_shot", 0.5, 0.3, 1.0, 0.15),
        Shot("lob", 0.5, 0.95, 1.1, 0.10),
        Shot("approach_shot", 0.5, 0.5, 0.6, 0.03),
        Shot("volley", 0.3, 0.2, 0.4, 0.20),
        Shot("volley_r", 0.7, 0.2, 0.4, 0.20),
        Shot("passing_shot_l", 0.05, 0.85, 0.5, 0.25),
        Shot("passing_shot_r", 0.95, 0.85, 0.5, 0.25),
        Shot("deep_baseline", 0.5, 0.98, 0.85, 0.02),
        Shot("wide_forehand", 0.95, 0.8, 0.75, 0.12),
        Shot("wide_backhand", 0.05, 0.8, 0.75, 0.12),
    ]

    def __init__(self, rotation: TennisRotation, interval_sec: float = 0.4) -> None:
        self.rotation = rotation
        self.interval_sec = interval_sec
        self._match_count = 0

        self.page: Optional[ft.Page] = None
        self.ball: Optional[ft.Container] = None
        self._root_view: Optional[ft.Column] = None

        # Base court dimensions (will be scaled for mobile)
        self._base_court_width = 300
        self._base_court_height = 400
        self._scale = 1.0

        # These will be calculated based on scale
        self.court_width = self._base_court_width
        self.court_height = self._base_court_height
        self._marker_size = 24
        self._ball_radius = 7

        # Court layout
        self._padding = 16
        self._alley_width = 25
        self._net_y = self.court_height / 2

        # Playable area boundaries
        self._play_left = self._padding + self._alley_width + 8
        self._play_right = self.court_width - self._padding - self._alley_width - 8
        self._play_width = self._play_right - self._play_left

        # Player court zones (near = bottom, far = top)
        self._near_baseline_y = self.court_height - 45
        self._near_net_y = self._net_y + 35
        self._far_baseline_y = 45
        self._far_net_y = self._net_y - 35

        # Bench position - will be recalculated
        self._bench_area_left = self.court_width + 30
        self._bench_area_top = self.court_height / 2 - 50
        self._bench_slot = (self._bench_area_left + 20, self._bench_area_top + 45)

        # Current player positions (x, y) - dynamically updated
        self._player_positions: Dict[str, Tuple[float, float]] = {}

        self.player_markers: Dict[str, ft.Container] = {}
        self._player_to_side: Dict[str, str] = {}  # "near" or "far"

        # Slower, more visible animations
        self._anim_ball = _make_animation(500, "easeOutCubic")
        self._anim_player = _make_animation(450, "easeOutQuad")
        self._anim_fade = _make_animation(120, "linear")
        self._ball_shadow = _make_shadow(spread_radius=2, blur_radius=6, color="#ffffff44")

        # Score indicators
        self._score_dots: Dict[str, List[ft.Container]] = {}

        # Base timing for shots (will be multiplied by shot speed)
        self._base_shot_time = 0.55

    def _calculate_dimensions(self, page_width: float, page_height: float) -> None:
        """Calculate responsive dimensions based on screen size."""
        # Determine scale based on available space
        # Leave room for bench area (add ~100px) and padding
        available_width = page_width - 120
        available_height = page_height - 80

        # Calculate scale to fit court
        scale_w = available_width / self._base_court_width
        scale_h = available_height / self._base_court_height
        self._scale = min(scale_w, scale_h, 1.3)  # Cap at 1.3x for large screens
        self._scale = max(self._scale, 0.6)  # Min 0.6x for very small screens

        # Apply scale
        self.court_width = self._base_court_width * self._scale
        self.court_height = self._base_court_height * self._scale
        self._marker_size = int(24 * self._scale)
        self._ball_radius = int(7 * self._scale)
        self._padding = int(16 * self._scale)
        self._alley_width = int(25 * self._scale)

        # Recalculate derived values
        self._net_y = self.court_height / 2
        self._play_left = self._padding + self._alley_width + int(8 * self._scale)
        self._play_right = self.court_width - self._padding - self._alley_width - int(8 * self._scale)
        self._play_width = self._play_right - self._play_left

        self._near_baseline_y = self.court_height - int(45 * self._scale)
        self._near_net_y = self._net_y + int(35 * self._scale)
        self._far_baseline_y = int(45 * self._scale)
        self._far_net_y = self._net_y - int(35 * self._scale)

        self._bench_area_left = self.court_width + int(25 * self._scale)
        self._bench_area_top = self.court_height / 2 - int(50 * self._scale)
        self._bench_slot = (
            self._bench_area_left + int(20 * self._scale),
            self._bench_area_top + int(45 * self._scale)
        )

    def _get_court_position(self, side: str, rel_x: float, rel_y: float) -> Tuple[float, float]:
        """Convert relative position (0-1) to actual court coordinates."""
        x = self._play_left + rel_x * self._play_width

        if side == "near":
            y = self._near_baseline_y - rel_y * (self._near_baseline_y - self._near_net_y)
        else:
            y = self._far_baseline_y + rel_y * (self._far_net_y - self._far_baseline_y)

        return (x, y)

    def _get_ball_target(self, side: str, shot: Shot) -> Tuple[float, float]:
        """Get where the ball should land for a given shot."""
        target_side = "far" if side == "near" else "near"
        return self._get_court_position(target_side, shot.target_x, shot.target_y)

    async def start(self, page: ft.Page) -> None:
        """Mount the UI on the provided Flet page and start the loop."""
        self.page = page
        page.title = "Tennis Rotation"
        page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        page.vertical_alignment = ft.MainAxisAlignment.CENTER
        page.bgcolor = "#000000"
        page.padding = 10

        # Mobile-friendly settings
        page.theme_mode = ft.ThemeMode.DARK

        # Get initial dimensions
        width = page.width or 400
        height = page.height or 700
        self._calculate_dimensions(width, height)

        self._root_view = self._build_ui()
        page.add(self._root_view)
        await self._init_players(self.rotation.current_players, self.rotation.bench_player)
        if hasattr(page, "run_task"):
            page.run_task(self._loop)
        else:
            asyncio.create_task(self._loop())

    def _build_ui(self) -> ft.Column:
        all_elements: List[ft.Control] = []

        # Court surface
        court_surface = ft.Container(
            left=0, top=0,
            width=self.court_width, height=self.court_height,
            bgcolor="#1a1a1a",
            border_radius=ft.border_radius.all(4),
        )
        all_elements.append(court_surface)

        # Outer boundary
        all_elements.append(ft.Container(
            left=self._padding, top=self._padding,
            width=self.court_width - 2 * self._padding,
            height=self.court_height - 2 * self._padding,
            border=ft.border.all(3, "#ffffff"),
        ))

        # Singles sidelines
        singles_left = self._padding + self._alley_width
        singles_right = self.court_width - self._padding - self._alley_width
        singles_width = singles_right - singles_left

        all_elements.append(ft.Container(
            left=singles_left, top=self._padding,
            width=2, height=self.court_height - 2 * self._padding,
            bgcolor="#ffffff",
        ))
        all_elements.append(ft.Container(
            left=singles_right - 2, top=self._padding,
            width=2, height=self.court_height - 2 * self._padding,
            bgcolor="#ffffff",
        ))

        # Net
        net_overhang = int(8 * self._scale)
        net_height = max(4, int(5 * self._scale))
        all_elements.append(ft.Container(
            left=self._padding - net_overhang, top=self._net_y - net_height / 2,
            width=self.court_width - 2 * self._padding + 2 * net_overhang, height=net_height,
            bgcolor="#ffffff",
        ))

        # Net posts
        post_size = max(6, int(7 * self._scale))
        all_elements.append(ft.Container(
            left=self._padding - net_overhang - post_size / 2, top=self._net_y - post_size,
            width=post_size, height=post_size * 2, bgcolor="#ffffff",
            border_radius=ft.border_radius.all(2),
        ))
        all_elements.append(ft.Container(
            left=self.court_width - self._padding + net_overhang - post_size / 2, top=self._net_y - post_size,
            width=post_size, height=post_size * 2, bgcolor="#ffffff",
            border_radius=ft.border_radius.all(2),
        ))

        # Service lines
        service_box_height = int(85 * self._scale)
        center_x = self.court_width / 2

        all_elements.append(ft.Container(
            left=singles_left, top=self._padding + service_box_height,
            width=singles_width, height=2, bgcolor="#ffffff",
        ))
        all_elements.append(ft.Container(
            left=singles_left, top=self.court_height - self._padding - service_box_height - 2,
            width=singles_width, height=2, bgcolor="#ffffff",
        ))

        # Center service lines
        all_elements.append(ft.Container(
            left=center_x - 1, top=self._padding + service_box_height,
            width=2, height=self._net_y - self._padding - service_box_height,
            bgcolor="#ffffff",
        ))
        all_elements.append(ft.Container(
            left=center_x - 1, top=self._net_y,
            width=2, height=self.court_height - self._net_y - self._padding - service_box_height,
            bgcolor="#ffffff",
        ))

        # Center marks
        center_mark_height = int(10 * self._scale)
        all_elements.append(ft.Container(
            left=center_x - 1, top=self._padding, width=2, height=center_mark_height, bgcolor="#ffffff",
        ))
        all_elements.append(ft.Container(
            left=center_x - 1, top=self.court_height - self._padding - center_mark_height,
            width=2, height=center_mark_height, bgcolor="#ffffff",
        ))

        # Score dots
        near_score_y = self.court_height - int(25 * self._scale)
        far_score_y = int(25 * self._scale)
        dot_spacing = int(14 * self._scale)
        dot_size = int(9 * self._scale)

        for slot, score_y in [("near", near_score_y), ("far", far_score_y)]:
            dots = []
            for i in range(self._POINTS_TO_WIN):
                dot = ft.Container(
                    left=center_x - (self._POINTS_TO_WIN * dot_spacing) / 2 + i * dot_spacing,
                    top=score_y - dot_size / 2,
                    width=dot_size, height=dot_size,
                    bgcolor="#333333",
                    border=ft.border.all(1, "#666666"),
                    border_radius=ft.border_radius.all(dot_size / 2),
                )
                dots.append(dot)
                all_elements.append(dot)
            self._score_dots[slot] = dots

        # Bench area
        bench_width = int(65 * self._scale)
        bench_height = int(100 * self._scale)
        bench_bg = ft.Container(
            left=self._bench_area_left, top=self._bench_area_top,
            width=bench_width, height=bench_height, bgcolor="#0d0d0d",
            border=ft.border.all(2, "#333333"),
            border_radius=ft.border_radius.all(6),
        )
        all_elements.append(bench_bg)

        bench_bar_top = self._bench_area_top + int(65 * self._scale)
        for i in range(3):
            all_elements.append(ft.Container(
                left=self._bench_area_left + int(8 * self._scale),
                top=bench_bar_top + i * int(10 * self._scale),
                width=bench_width - int(16 * self._scale), height=int(3 * self._scale),
                bgcolor="#444444",
                border_radius=ft.border_radius.all(2),
            ))

        # Ball
        ball_kwargs = {}
        if self._anim_ball:
            ball_kwargs["animate_position"] = self._anim_ball
        if self._anim_fade:
            ball_kwargs["animate_opacity"] = self._anim_fade
        if self._ball_shadow:
            ball_kwargs["shadow"] = self._ball_shadow

        self.ball = ft.Container(
            left=-50, top=-50,
            width=self._ball_radius * 2, height=self._ball_radius * 2,
            bgcolor="#ffffff",
            border_radius=ft.border_radius.all(self._ball_radius),
            opacity=0,
            **ball_kwargs,
        )
        all_elements.append(self.ball)

        # Player markers
        for idx, player in enumerate(self.rotation.players):
            marker = self._create_marker(self._SHAPES[idx % len(self._SHAPES)])
            marker.left = -100
            marker.top = -100
            if self._anim_player:
                marker.animate_position = self._anim_player
            self.player_markers[player] = marker
            all_elements.append(marker)

        total_width = self._bench_area_left + bench_width + int(10 * self._scale)
        canvas = ft.Stack(width=total_width, height=self.court_height, controls=all_elements)

        canvas_padding = max(10, int(20 * self._scale))
        return ft.Column(
            controls=[ft.Container(content=canvas, padding=canvas_padding)],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

    def _create_marker(self, shape: str) -> ft.Container:
        half = self._marker_size / 2
        border_width = max(2, int(3 * self._scale))
        if shape == "circle":
            return ft.Container(
                width=self._marker_size, height=self._marker_size,
                bgcolor="#ffffff",
                border_radius=ft.border_radius.all(half),
            )
        if shape == "square":
            return ft.Container(
                width=self._marker_size, height=self._marker_size,
                bgcolor="#000000",
                border=ft.border.all(border_width, "#ffffff"),
            )
        return self._triangle_marker()

    def _triangle_marker(self) -> ft.Container:
        steps = 8
        row_height = self._marker_size / steps
        bars: List[ft.Container] = []
        for i in range(steps):
            width = (i + 1) / steps * self._marker_size
            bars.append(ft.Container(
                left=(self._marker_size - width) / 2,
                top=i * row_height,
                width=width, height=row_height + 1,
                bgcolor="#ffffff",
            ))
        return ft.Stack(width=self._marker_size, height=self._marker_size, controls=bars)

    async def _loop(self) -> None:
        while True:
            await self._play_match()
            await asyncio.sleep(self.interval_sec)

    async def _play_match(self) -> None:
        """Play a full match (first to 3 points wins)."""
        active = self.rotation.current_players
        scores = {active[0]: 0, active[1]: 0}

        await self._update_score_display(scores, active)

        while max(scores.values()) < self._POINTS_TO_WIN:
            point_winner = await self._play_point(active)
            scores[point_winner] += 1
            await self._update_score_display(scores, active)
            await asyncio.sleep(0.6)  # Pause between points

        match_winner = active[0] if scores[active[0]] >= self._POINTS_TO_WIN else active[1]
        await self._flash_winner_score(match_winner, active)

        snapshot = self.rotation.record_match(match_winner)
        self._match_count += 1

        await self._reset_score_display()
        await self._init_players(snapshot.active_after, snapshot.bench_after)
        await asyncio.sleep(1.5)  # Pause before next match

    async def _play_point(self, active: Tuple[str, str]) -> str:
        """Play a single point with realistic shots and movement."""
        rally_length = random.randint(4, 12)

        server_idx = random.randint(0, 1)
        server = active[server_idx]
        receiver = active[1 - server_idx]

        server_side = self._player_to_side[server]
        receiver_side = self._player_to_side[receiver]

        # Position players for serve (move simultaneously)
        serve_pos = self._get_court_position(server_side, 0.5, 0.0)
        receive_pos = self._get_court_position(receiver_side, 0.5, 0.1)

        self._set_player_position(server, serve_pos)
        self._set_player_position(receiver, receive_pos)
        await self._page_update()
        await asyncio.sleep(0.4)  # Players get ready

        # Ball appears at server
        if self.ball:
            self.ball.left = serve_pos[0] - self._ball_radius
            self.ball.top = serve_pos[1] - self._ball_radius
            self.ball.opacity = 1
            await self._page_update()
            await asyncio.sleep(0.3)  # Serve toss

        current_hitter = server
        current_side = server_side

        # Rally loop
        for _ in range(rally_length):
            hitter_pos = self._player_positions.get(current_hitter, (0, 0))
            shot = self._choose_shot(current_side, hitter_pos)
            ball_target = self._get_ball_target(current_side, shot)

            # Get opponent info
            opponent = active[0] if current_hitter == active[1] else active[1]
            opponent_side = self._player_to_side[opponent]

            # Calculate where opponent needs to move
            retrieve_pos = self._get_retrieve_position(opponent_side, ball_target)
            recovery_pos = self._get_recovery_position(current_side, shot)

            # Move ball AND both players simultaneously
            if self.ball:
                self.ball.left = ball_target[0] - self._ball_radius
                self.ball.top = ball_target[1] - self._ball_radius

            self._set_player_position(opponent, retrieve_pos)
            self._set_player_position(current_hitter, recovery_pos)
            await self._page_update()

            # Wait for animation (based on shot speed)
            await asyncio.sleep(self._base_shot_time * shot.speed)

            # Check for winner
            if random.random() < shot.is_winner_chance:
                if self.ball:
                    self.ball.opacity = 0
                    await self._page_update()
                return current_hitter

            # Check for unforced error
            if random.random() < 0.06:
                if self.ball:
                    self.ball.opacity = 0
                    await self._page_update()
                return opponent

            current_hitter = opponent
            current_side = opponent_side

        # Rally ends naturally
        if self.ball:
            self.ball.opacity = 0
            await self._page_update()

        point_winner = active[0] if current_hitter == active[1] else active[1]
        return point_winner

    def _set_player_position(self, player: str, pos: Tuple[float, float]) -> None:
        """Set player position (updates marker, no page update)."""
        self._player_positions[player] = pos
        marker = self.player_markers.get(player)
        if marker:
            half = self._marker_size / 2
            marker.left = pos[0] - half
            marker.top = pos[1] - half

    def _choose_shot(self, side: str, pos: Tuple[float, float]) -> Shot:
        """Choose an appropriate shot based on position."""
        rel_y = pos[1] / self.court_height

        if side == "near":
            at_net = rel_y < 0.6
        else:
            at_net = rel_y > 0.4

        if at_net:
            net_shots = [s for s in self._SHOTS if "volley" in s.name or "drop" in s.name]
            if net_shots and random.random() < 0.6:
                return random.choice(net_shots)

        rel_x = (pos[0] - self._play_left) / self._play_width
        if rel_x < 0.3:
            preferred = [s for s in self._SHOTS if s.target_x > 0.6]
            if preferred and random.random() < 0.5:
                return random.choice(preferred)
        elif rel_x > 0.7:
            preferred = [s for s in self._SHOTS if s.target_x < 0.4]
            if preferred and random.random() < 0.5:
                return random.choice(preferred)

        return random.choice(self._SHOTS)

    def _get_retrieve_position(self, side: str, ball_target: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate where player needs to move to retrieve ball."""
        target_x = ball_target[0]
        target_x = max(self._play_left, min(self._play_right, target_x))

        if side == "near":
            if ball_target[1] > self._net_y + 80:
                target_y = self._near_baseline_y
            else:
                target_y = max(self._near_net_y, ball_target[1] + 30)
        else:
            if ball_target[1] < self._net_y - 80:
                target_y = self._far_baseline_y
            else:
                target_y = min(self._far_net_y, ball_target[1] - 30)

        return (target_x, target_y)

    def _get_recovery_position(self, side: str, shot: Shot) -> Tuple[float, float]:
        """Get position to recover to after hitting."""
        center_x = self._play_left + self._play_width * 0.5

        if shot.target_x > 0.5:
            recover_x = center_x - 15
        else:
            recover_x = center_x + 15

        if side == "near":
            if "approach" in shot.name or "volley" in shot.name:
                recover_y = self._near_net_y + 15
            else:
                recover_y = self._near_baseline_y - 8
        else:
            if "approach" in shot.name or "volley" in shot.name:
                recover_y = self._far_net_y - 15
            else:
                recover_y = self._far_baseline_y + 8

        return (recover_x, recover_y)

    async def _init_players(self, active: Sequence[str], bench: str) -> None:
        """Initialize player positions at start of match."""
        if not self.page:
            return

        self._player_to_side.clear()
        self._player_positions.clear()

        half = self._marker_size / 2

        # Near player (bottom)
        near_pos = self._get_court_position("near", 0.5, 0.0)
        self._player_to_side[active[0]] = "near"
        self._player_positions[active[0]] = near_pos
        marker0 = self.player_markers[active[0]]
        marker0.left = near_pos[0] - half
        marker0.top = near_pos[1] - half
        marker0.opacity = 1

        # Far player (top)
        far_pos = self._get_court_position("far", 0.5, 0.0)
        self._player_to_side[active[1]] = "far"
        self._player_positions[active[1]] = far_pos
        marker1 = self.player_markers[active[1]]
        marker1.left = far_pos[0] - half
        marker1.top = far_pos[1] - half
        marker1.opacity = 1

        # Bench player
        bench_marker = self.player_markers[bench]
        bench_marker.left = self._bench_slot[0] - half
        bench_marker.top = self._bench_slot[1] - half
        bench_marker.opacity = 1

        await self._page_update()

    async def _update_score_display(self, scores: Dict[str, int], active: Tuple[str, str]) -> None:
        """Update the score dots."""
        if not self.page:
            return

        for idx, slot in enumerate(("near", "far")):
            player = active[idx]
            player_score = scores.get(player, 0)
            dots = self._score_dots.get(slot, [])

            for i, dot in enumerate(dots):
                if i < player_score:
                    dot.bgcolor = "#ffffff"
                    dot.border = ft.border.all(1, "#ffffff")
                else:
                    dot.bgcolor = "#333333"
                    dot.border = ft.border.all(1, "#666666")

        await self._page_update()

    async def _flash_winner_score(self, winner: str, active: Tuple[str, str]) -> None:
        """Flash the winner's score dots."""
        if not self.page:
            return

        slot = "near" if active[0] == winner else "far"
        dots = self._score_dots.get(slot, [])

        for _ in range(3):
            for dot in dots:
                dot.opacity = 0.3
            await self._page_update()
            await asyncio.sleep(0.1)
            for dot in dots:
                dot.opacity = 1
            await self._page_update()
            await asyncio.sleep(0.1)

    async def _reset_score_display(self) -> None:
        """Reset all score dots."""
        if not self.page:
            return

        for slot in ("near", "far"):
            dots = self._score_dots.get(slot, [])
            for dot in dots:
                dot.bgcolor = "#333333"
                dot.border = ft.border.all(1, "#666666")
                dot.opacity = 1

        await self._page_update()

    async def _page_update(self) -> None:
        if not self.page:
            return
        async_update = getattr(self.page, "update_async", None)
        if async_update:
            await async_update()
        else:
            self.page.update()


async def main(page: ft.Page) -> None:
    # Mobile-friendly PWA settings
    page.title = "Tennis Rotation"
    page.theme_mode = ft.ThemeMode.DARK

    rotation = TennisRotation(["Player 1", "Player 2", "Player 3"])
    visualizer = RotationVisualizer(rotation, interval_sec=0.4)
    await visualizer.start(page)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    print(f"Starting Tennis Rotation on http://localhost:{port}")
    ft.app(
        target=main,
        port=port,
        view=None,  # Don't open browser, just serve
    )
