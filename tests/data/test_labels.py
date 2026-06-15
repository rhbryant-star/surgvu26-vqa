from __future__ import annotations

from pathlib import Path

from surgvu_vqa.data.labels import CaseLabels, parse_time_s

# Real tools.csv format: HH:MM:SS.ffffff times, install/uninstall parts as floats.
TOOLS_CSV = """index,install_case_part,install_case_time,uninstall_case_part,uninstall_case_time,arm,commercial_toolname,groundtruth_toolname
0,1.0,00:00:10.000000,1.0,00:01:40.000000,USM1,Large Needle Driver,needle driver
1,1.0,00:00:20.500000,2.0,00:00:30.000000,USM2,Cadiere Forceps,cadiere forceps
"""

# Real tasks.csv format: bare float seconds, column groundtruth_taskname.
TASKS_CSV = """index,start_part,start_time,stop_part,stop_time,groundtruth_taskname
0,1,5.0,1,120.0,Suturing
1,2,0.0,2,60.0,Range of motion
"""


def _write_case(tmp_path: Path) -> Path:
    case = tmp_path / "case_001"
    case.mkdir()
    (case / "tools.csv").write_text(TOOLS_CSV)
    (case / "tasks.csv").write_text(TASKS_CSV)
    return case


def test_parse_time_s():
    assert parse_time_s("00:01:40.500000") == 100.5


def test_tools_parsed_per_part(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    # Tool 0 lives entirely in part 1: 10s..100s
    t0 = labels.tool_intervals[0]
    assert (t0.part, t0.start_s, t0.end_s) == (1, 10.0, 100.0)
    assert t0.groundtruth == "needle driver"
    assert t0.commercial == "Large Needle Driver"
    # Tool 1 spans parts (install part 1, uninstall part 2) and must be
    # represented without inventing cross-part timestamps:
    t1 = labels.tool_intervals[1]
    assert t1.part == 1 and t1.start_s == 20.5 and t1.end_s is None  # open until end of part


def test_tasks_parsed(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    a = labels.task_intervals[0]
    assert (a.part, a.start_s, a.end_s, a.task) == (1, 5.0, 120.0, "Suturing")


def test_tools_in_window(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    present = labels.tools_in_window(part=1, start_s=15.0, end_s=45.0, min_overlap_s=10.0)
    names = {t.groundtruth for t in present}
    assert "needle driver" in names          # overlaps 15..45 fully
    assert "cadiere forceps" in names        # 20.5..45 = 24.5s >= 10s
    none = labels.tools_in_window(part=1, start_s=101.0, end_s=131.0, min_overlap_s=10.0)
    assert {t.groundtruth for t in none} == {"cadiere forceps"}  # open-ended interval still active


def test_task_for_window(tmp_path):
    labels = CaseLabels.load(_write_case(tmp_path))
    assert labels.task_for_window(part=1, start_s=10.0, end_s=40.0) == "Suturing"
    assert labels.task_for_window(part=1, start_s=110.0, end_s=140.0) is None  # <50% covered
