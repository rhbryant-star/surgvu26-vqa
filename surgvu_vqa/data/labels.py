"""Parse one SurgVU case's tools.csv / tasks.csv into typed intervals.

Times are seconds WITHIN a video part (parts are separate mp4 files named
case_NNN_video_part_PPP.mp4). A tool installed in part P and uninstalled in a
later part is represented as an open-ended interval in part P (end_s=None):
we never invent cross-part timestamps. Column names are centralized below —
verified against the real 2024-v2 labels; adjust ONLY these if the header
drifts (Task 1 Step 1 inspection).

tools.csv: HH:MM:SS.ffffff times; parts are floats (e.g. "1.0").
tasks.csv: bare float seconds; parts are integers; task column is
           groundtruth_taskname (NOT "task").
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

TOOL_COLS = {
    "install_part": "install_case_part",
    "install_time": "install_case_time",
    "uninstall_part": "uninstall_case_part",
    "uninstall_time": "uninstall_case_time",
    "commercial": "commercial_toolname",
    "groundtruth": "groundtruth_toolname",
}
TASK_COLS = {
    "task": "groundtruth_taskname",
    "start_part": "start_part",
    "start_time": "start_time",
    "stop_part": "stop_part",
    "stop_time": "stop_time",
}


def parse_time_s(text: str) -> float:
    """'HH:MM:SS.ffffff' -> seconds (tools.csv format)."""
    h, m, s = text.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


@dataclass(frozen=True)
class ToolInterval:
    part: int
    start_s: float
    end_s: float | None  # None = active until the end of this part
    groundtruth: str
    commercial: str


@dataclass(frozen=True)
class TaskInterval:
    part: int
    start_s: float
    end_s: float
    task: str


@dataclass
class CaseLabels:
    case_id: str
    tool_intervals: list[ToolInterval]
    task_intervals: list[TaskInterval]

    @classmethod
    def load(cls, case_dir: Path) -> "CaseLabels":
        case_dir = Path(case_dir)
        tools: list[ToolInterval] = []
        with open(case_dir / "tools.csv", newline="") as f:
            for row in csv.DictReader(f):
                ip = int(float(row[TOOL_COLS["install_part"]]))
                up = int(float(row[TOOL_COLS["uninstall_part"]]))
                start = parse_time_s(row[TOOL_COLS["install_time"]])
                end: float | None = parse_time_s(row[TOOL_COLS["uninstall_time"]])
                if up != ip:
                    end = None  # spans parts: open-ended within install part
                tools.append(ToolInterval(
                    part=ip, start_s=start, end_s=end,
                    groundtruth=row[TOOL_COLS["groundtruth"]].strip(),
                    commercial=row[TOOL_COLS["commercial"]].strip(),
                ))
        tasks: list[TaskInterval] = []
        with open(case_dir / "tasks.csv", newline="") as f:
            for row in csv.DictReader(f):
                sp = int(float(row[TASK_COLS["start_part"]]))
                ep = int(float(row[TASK_COLS["stop_part"]]))
                if sp != ep:
                    continue  # cross-part task segments are skipped (risk 2)
                task_name = row[TASK_COLS["task"]].strip()
                if not task_name:
                    continue  # drop empty task names (revision 4)
                tasks.append(TaskInterval(
                    part=sp,
                    start_s=float(row[TASK_COLS["start_time"]]),
                    end_s=float(row[TASK_COLS["stop_time"]]),
                    task=task_name,
                ))
        return cls(case_id=case_dir.name, tool_intervals=tools, task_intervals=tasks)

    def tools_in_window(self, part: int, start_s: float, end_s: float,
                        min_overlap_s: float = 10.0) -> list[ToolInterval]:
        """Tools whose installed interval overlaps [start_s, end_s) in `part`
        by at least min_overlap_s. Open-ended intervals extend to +inf."""
        out = []
        for t in self.tool_intervals:
            if t.part != part:
                continue
            t_end = t.end_s if t.end_s is not None else float("inf")
            overlap = min(t_end, end_s) - max(t.start_s, start_s)
            if overlap >= min_overlap_s:
                out.append(t)
        return out

    def task_for_window(self, part: int, start_s: float, end_s: float) -> str | None:
        """The task covering >=50% of the window, else None."""
        need = (end_s - start_s) / 2.0
        for a in self.task_intervals:
            if a.part != part:
                continue
            overlap = min(a.end_s, end_s) - max(a.start_s, start_s)
            if overlap >= need:
                return a.task
        return None
