"""Side-effect-free replay rendering and media export."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from relay.envs.visibility import visibility_mask
from relay.runtime.replay_store import REPLAY_SCHEMA_VERSION

DEFAULT_OVERLAYS = {
    "grid",
    "visibility",
    "incidents",
    "actions",
    "communications",
    "rewards",
    "hash",
}

ROLE_COLORS = {
    "scout": "#2E74B5",
    "ambulance": "#2E7D32",
    "fireman": "#C44E32",
}


class ReplayData:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.rows: list[dict[str, Any]] = pq.read_table(self.path).to_pylist()
        if not self.rows:
            raise ValueError("replay contains no rows")
        if any(row["schema_version"] != REPLAY_SCHEMA_VERSION for row in self.rows):
            raise ValueError("unsupported replay schema")
        self.environment_config: dict[str, Any] = json.loads(
            self.rows[0]["environment_config_json"]
        )
        self.snapshots = [json.loads(row["snapshot_json"]) for row in self.rows]
        self.actions = [json.loads(row["actions_json"]) for row in self.rows]
        self.infos = [json.loads(row["infos_json"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def agents(self) -> list[str]:
        return list(self.snapshots[0]["agents"])

    def inspection(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        first_agent = self.agents[0]
        info = self.infos[index][first_agent]
        snapshot = self.snapshots[index]
        unresolved = [
            incident_id
            for incident_id, incident in snapshot["incidents"].items()
            if incident["status"] != "resolved"
        ]
        return {
            "replay": str(self.path),
            "tick": row["tick"],
            "episode": row["episode"],
            "scenario_seed": row["scenario_seed"],
            "scenario_id": row["scenario_id"],
            "state_hash": row["state_hash"],
            "run_id": row.get("run_id", "unknown"),
            "checkpoint_id": row.get("checkpoint_id", "unknown"),
            "checkpoint_step": row.get("checkpoint_step", -1),
            "actions": self.actions[index],
            "reward_components": info["reward_components"],
            "recipient_attempts": info["recipient_attempts"],
            "packet_events": info["packet_events"],
            "unresolved_incidents": unresolved,
            "terminal_reason": info["terminal_reason"],
        }


@lru_cache(maxsize=8)
def _font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    candidates = (
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _perspective_mask(data: ReplayData, index: int, perspective: str) -> np.ndarray | None:
    if perspective == "global":
        return None
    snapshot = data.snapshots[index]
    if perspective not in snapshot["agents"]:
        raise ValueError(f"unknown perspective {perspective!r}")
    agent = snapshot["agents"][perspective]
    walls = np.asarray(snapshot["walls"], dtype=np.bool_)
    radius = (
        int(data.environment_config["scout_sensor_radius"])
        if agent["role"] == "scout"
        else int(data.environment_config["specialist_sensor_radius"])
    )
    return visibility_mask(walls, (int(agent["x"]), int(agent["y"])), radius)


def render_replay_frame(
    data: ReplayData,
    index: int,
    *,
    perspective: str = "global",
    overlays: set[str] | None = None,
    cell_size: int = 32,
) -> Image.Image:
    if not 0 <= index < len(data):
        raise IndexError(index)
    enabled = DEFAULT_OVERLAYS if overlays is None else overlays
    snapshot = data.snapshots[index]
    row = data.rows[index]
    walls = np.asarray(snapshot["walls"], dtype=np.bool_)
    staging = np.asarray(snapshot["staging"], dtype=np.bool_)
    height, width = walls.shape
    panel_width = 430
    margin = 20
    image = Image.new(
        "RGB",
        (width * cell_size + panel_width + margin * 3, height * cell_size + margin * 2),
        "#F4F7FB",
    )
    draw = ImageDraw.Draw(image)
    ox = oy = margin
    visibility = _perspective_mask(data, index, perspective)
    for y in range(height):
        for x in range(width):
            box = (
                ox + x * cell_size,
                oy + y * cell_size,
                ox + (x + 1) * cell_size,
                oy + (y + 1) * cell_size,
            )
            fill = "#FFFFFF"
            if staging[y, x]:
                fill = "#DCEAF7"
            if walls[y, x]:
                fill = "#26364A"
            if visibility is not None and not visibility[y, x] and "visibility" in enabled:
                fill = "#B7BEC8" if not walls[y, x] else "#556170"
            outline = "#CBD2DA" if "grid" in enabled else fill
            draw.rectangle(box, fill=fill, outline=outline)
    if "incidents" in enabled:
        for incident in snapshot["incidents"].values():
            if incident["status"] == "resolved":
                continue
            x, y = int(incident["x"]), int(incident["y"])
            if visibility is not None and not visibility[y, x]:
                continue
            cx = ox + x * cell_size + cell_size // 2
            cy = oy + y * cell_size + cell_size // 2
            radius = max(6, cell_size // 3)
            color = "#D32F2F" if incident["kind"] == "victim" else "#F57C00"
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
            label = "V" if incident["kind"] == "victim" else "F"
            draw.text((cx - 5, cy - 8), label, fill="white", font=_font(13))
            if incident["status"] == "reserved":
                draw.ellipse(
                    (cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3),
                    outline="#8A6500",
                    width=3,
                )
    positions: dict[str, tuple[int, int]] = {}
    for agent_id, agent in snapshot["agents"].items():
        x, y = int(agent["x"]), int(agent["y"])
        positions[agent_id] = (x, y)
        if visibility is not None and not visibility[y, x] and agent_id != perspective:
            continue
        cx = ox + x * cell_size + cell_size // 2
        cy = oy + y * cell_size + cell_size // 2
        radius = max(8, cell_size // 3)
        draw.rounded_rectangle(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            radius=5,
            fill=ROLE_COLORS[str(agent["role"])],
            outline="#FFD54F" if agent_id == perspective else "white",
            width=3 if agent_id == perspective else 1,
        )
        draw.text((cx - 6, cy - 8), agent_id[0].upper(), fill="white", font=_font(13))
        if int(agent["busy_remaining"]) > 0:
            draw.text(
                (cx - 10, cy + radius + 1),
                f"{agent['busy_remaining']}t",
                fill="#6A4A00",
                font=_font(11),
            )
    if "actions" in enabled:
        direction = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
        for agent_id, action in data.actions[index].items():
            physical = int(action["physical"])
            if physical not in direction or agent_id not in positions:
                continue
            x, y = positions[agent_id]
            dx, dy = direction[physical]
            cx = ox + x * cell_size + cell_size // 2
            cy = oy + y * cell_size + cell_size // 2
            draw.line(
                (cx, cy, cx + dx * cell_size * 0.7, cy + dy * cell_size * 0.7),
                fill="#111827",
                width=3,
            )
    first_agent = data.agents[0]
    info = data.infos[index][first_agent]
    if "communications" in enabled:
        event_colors = {"queued": "#7B1FA2", "delivered": "#00838F", "dropped": "#C62828"}
        for event in info["packet_events"]:
            sender = event.get("sender_id")
            recipient = event.get("recipient_id")
            if sender not in positions or recipient not in positions:
                continue
            sx, sy = positions[sender]
            rx, ry = positions[recipient]
            draw.line(
                (
                    ox + sx * cell_size + cell_size // 2,
                    oy + sy * cell_size + cell_size // 2,
                    ox + rx * cell_size + cell_size // 2,
                    oy + ry * cell_size + cell_size // 2,
                ),
                fill=event_colors.get(event["event"], "#6B7280"),
                width=2,
            )
    panel_x = ox + width * cell_size + margin
    title_font = _font(20)
    body_font = _font(13)
    draw.text((panel_x, margin), "RELAY replay", fill="#17365D", font=title_font)
    lines = [
        f"Tick: {row['tick']} / {len(data)}",
        f"Communication: {data.environment_config['communication_mode']}",
        f"Perspective: {perspective}",
        f"Scenario: {row['scenario_id']}",
        f"Seed: {row['scenario_seed']}",
        f"Run: {row.get('run_id', 'unknown')}",
        f"Checkpoint: {row.get('checkpoint_id', 'unknown')}",
        f"Attempts: {info['recipient_attempts']}",
    ]
    if "rewards" in enabled:
        lines.append("Reward components:")
        lines.extend(f"  {key}: {value:+.3f}" for key, value in info["reward_components"].items())
    if info["terminal_reason"]:
        lines.append(f"Terminal: {info['terminal_reason']}")
    y_cursor = margin + 38
    for line in lines:
        draw.text((panel_x, y_cursor), line[:58], fill="#27364A", font=body_font)
        y_cursor += 21
    if "hash" in enabled:
        y_cursor += 8
        draw.text((panel_x, y_cursor), "State hash", fill="#17365D", font=body_font)
        y_cursor += 20
        state_hash = str(row["state_hash"])
        draw.text((panel_x, y_cursor), state_hash[:32], fill="#5F6B7A", font=_font(11))
        draw.text((panel_x, y_cursor + 16), state_hash[32:], fill="#5F6B7A", font=_font(11))
    return image


def export_replay(
    data: ReplayData,
    output: str | Path,
    *,
    start: int = 0,
    end: int | None = None,
    fps: float = 4.0,
    perspective: str = "global",
    overlays: set[str] | None = None,
    stride: int = 1,
) -> Path:
    output_path = Path(output).resolve()
    final = len(data) - 1 if end is None else end
    if start < 0 or final >= len(data) or start > final:
        raise ValueError("invalid export tick range")
    if stride < 1:
        raise ValueError("replay export stride must be positive")
    frames = [
        np.asarray(render_replay_frame(data, index, perspective=perspective, overlays=overlays))
        for index in range(start, final + 1, stride)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".png":
        Image.fromarray(frames[0]).save(output_path)
    elif suffix == ".gif":
        imageio.mimsave(output_path, cast(Any, frames), duration=1000 / fps, loop=0)
    elif suffix == ".mp4":
        writer: Any = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        with writer:
            for frame in frames:
                writer.append_data(frame)
    else:
        raise ValueError("export extension must be .png, .gif, or .mp4")
    return output_path
