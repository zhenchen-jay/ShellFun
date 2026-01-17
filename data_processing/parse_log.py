#!/usr/bin/env python3
"""
Parse simulation logs and report statistics within a frame range.

Extracts:
- Test case + model type from paths like:
  simulation_outputs/comparison_experiments/CrushTestCan/Directional_StVK_Bending_Tan_RestFlat
- Total frames (seen in the parsed range)
- Total iterations (sum of "==== Frame ... total iterations: ...")
- Global time (right-hand cumulative duration from the LAST "Global :" line within range)
- Peak memory (max "Memory usage peak: ... M" within range)
- Hessian sparsity min/max/avg (from "Sparsity: ...%" within range)

Frame-range behavior:
- You can specify a frame window [start_frame, end_frame] (inclusive).
- Default: start_frame=0, end_frame=None (parse to end).
- We only start accumulating stats once we encounter "==== Frame <start_frame> ..."
- We stop after processing "==== Frame <end_frame> ..." if end_frame is provided.

Usage:
  python parse_log_stats.py /path/to/log.txt
  python parse_log_stats.py /path/to/log.txt --start-frame 10 --end-frame 120
  python parse_log_stats.py /path/to/log.txt --start-frame 10 --end-frame 120 --json
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple


@dataclass
class Stats:
    test_case: Optional[str]
    model_type: Optional[str]

    start_frame: int
    end_frame: Optional[int]

    frames_seen_in_range: int
    frame_first_seen: Optional[int]
    frame_last_seen: Optional[int]

    total_iterations: int
    global_time_s: Optional[float]
    peak_memory_mib: Optional[int]

    sparsity_min_pct: Optional[float]
    sparsity_max_pct: Optional[float]
    sparsity_avg_pct: Optional[float]
    sparsity_samples: int


# ----------------------------
# Regex patterns
# ----------------------------

# Matches per-frame summary like:
# "==== Frame 3 total iterations: 1  time: 7.513...(s)"
RE_FRAME_TOTAL_ITERS = re.compile(
    r"====\s*Frame\s+(\d+)\s+total iterations:\s*(\d+)\s+time:\s*([0-9.]+)\s*\(s\)",
    re.IGNORECASE,
)

# Matches memory peak like:
# "Memory usage peak: 7678 M"
RE_MEMORY_PEAK = re.compile(r"Memory usage peak:\s*(\d+)\s*M\b", re.IGNORECASE)

# Matches sparsity like:
# "Direct Solver, Sparsity: 0.0104%"
RE_SPARSITY = re.compile(r"Sparsity:\s*([0-9.]+)\s*%", re.IGNORECASE)

# Matches the timer-tree global line and captures both time strings:
# "Global : 2m 25.249569s (100.0%)   1h 13m 41.416288s (100.0%)"
# Use RHS cumulative time; keep LAST occurrence inside range.
RE_GLOBAL_LINE = re.compile(
    r"Global\s*:\s*(.*?)\s*\(\s*100(?:\.0+)?\s*%\s*\)\s+(.*?)\s*\(\s*100(?:\.0+)?\s*%\s*\)",
    re.IGNORECASE,
)

# Extract test_case and model folder from paths containing "comparison_experiments/<case>/<model_folder>"
RE_CASE_MODEL = re.compile(
    r"comparison_experiments/(?P<case>[^/\s]+)/(?P<model>[^/\s]+)",
    re.IGNORECASE,
)


# ----------------------------
# Helpers
# ----------------------------

def parse_duration_to_seconds(s: str) -> Optional[float]:
    """
    Parse durations like:
      "41.2s"
      "2m 25.249569s"
      "1h 13m 41.416288s"
    into seconds. Returns None if no recognizable tokens found.
    """
    s = s.strip()
    if not s:
        return None

    tokens = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*([hms])", s)
    if not tokens:
        return None

    hours = minutes = seconds = 0.0
    for num, unit in tokens:
        val = float(num)
        if unit == "h":
            hours += val
        elif unit == "m":
            minutes += val
        elif unit == "s":
            seconds += val

    return hours * 3600.0 + minutes * 60.0 + seconds


def strip_model_suffix(model_folder: str) -> str:
    """
    Strip common configuration suffixes:
      Directional_StVK_Bending_Tan_RestFlat -> Directional_StVK_Bending_Tan
    Extend as needed.
    """
    suffixes = [
        "_RestFlat",
        "_RestCurved",
        "_Irregular",
        "_Regular",
        "_Flat",
        "_Curved",
    ]
    for suf in suffixes:
        if model_folder.endswith(suf):
            return model_folder[: -len(suf)]
    return model_folder


def parse_case_and_model_from_line(line: str) -> Tuple[Optional[str], Optional[str]]:
    m = RE_CASE_MODEL.search(line)
    if not m:
        return None, None
    case = m.group("case")
    model_folder = m.group("model")
    return case, strip_model_suffix(model_folder)


def format_seconds_hms(total_s: float) -> str:
    h = int(total_s // 3600)
    m = int((total_s - 3600 * h) // 60)
    s = total_s - 3600 * h - 60 * m
    if h > 0:
        return f"{h}h {m}m {s:.6f}s"
    if m > 0:
        return f"{m}m {s:.6f}s"
    return f"{s:.6f}s"


# ----------------------------
# Main parser
# ----------------------------

def parse_log(path: str, start_frame: int = 0, end_frame: Optional[int] = None) -> Stats:
    if start_frame < 0:
        raise ValueError("--start-frame must be >= 0")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("--end-frame must be >= --start-frame")

    # For labeling only (last seen in file; not range-specific)
    last_test_case: Optional[str] = None
    last_model_type: Optional[str] = None

    # Range accumulation
    in_range = False
    frames_seen_in_range: List[int] = []
    total_iterations = 0
    peak_memory_mib: Optional[int] = None
    sparsities: List[float] = []
    last_global_cumulative_seconds: Optional[float] = None

    # We stop after we process the end_frame summary line (inclusive)
    stop_after_end_frame = False

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if stop_after_end_frame:
                break

            # Always keep updating case/model if present (useful even if appears before range)
            case, model = parse_case_and_model_from_line(line)
            if case and model:
                last_test_case = case
                last_model_type = model

            # Frame summary line defines the current frame boundary
            m = RE_FRAME_TOTAL_ITERS.search(line)
            if m:
                try:
                    frame_idx = int(m.group(1))
                    iters = int(m.group(2))
                except ValueError:
                    continue

                # Enter range at first frame >= start_frame
                if not in_range and frame_idx >= start_frame:
                    in_range = True

                # If already in range, but we overshot end_frame, stop (do not include this frame)
                if in_range and end_frame is not None and frame_idx > end_frame:
                    stop_after_end_frame = True
                    continue

                # Accumulate if the frame is within [start_frame, end_frame]
                if in_range and (end_frame is None or frame_idx <= end_frame):
                    frames_seen_in_range.append(frame_idx)
                    total_iterations += iters

                    if end_frame is not None and frame_idx == end_frame:
                        # Stop after processing this frame boundary
                        stop_after_end_frame = True

                continue  # don't let this line be double-used below

            # Only accumulate other stats while in range
            if not in_range:
                continue

            # Memory peak (within range)
            m = RE_MEMORY_PEAK.search(line)
            if m:
                try:
                    mem = int(m.group(1))
                    peak_memory_mib = mem if peak_memory_mib is None else max(peak_memory_mib, mem)
                except ValueError:
                    pass

            # Sparsity (within range)
            m = RE_SPARSITY.search(line)
            if m:
                try:
                    sparsities.append(float(m.group(1)))
                except ValueError:
                    pass

            # Global time line (within range; take RHS cumulative, keep LAST)
            m = RE_GLOBAL_LINE.search(line)
            if m:
                rhs = m.group(2)
                secs = parse_duration_to_seconds(rhs)
                if secs is not None:
                    last_global_cumulative_seconds = secs

    if frames_seen_in_range:
        frames_seen_unique = sorted(set(frames_seen_in_range))
        frame_first_seen = frames_seen_unique[0]
        frame_last_seen = frames_seen_unique[-1]
        frames_seen_count = len(frames_seen_unique)
    else:
        frame_first_seen = None
        frame_last_seen = None
        frames_seen_count = 0

    if sparsities:
        smin = min(sparsities)
        smax = max(sparsities)
        savg = sum(sparsities) / len(sparsities)
    else:
        smin = smax = savg = None

    return Stats(
        test_case=last_test_case,
        model_type=last_model_type,
        start_frame=start_frame,
        end_frame=end_frame,
        frames_seen_in_range=frames_seen_count,
        frame_first_seen=frame_first_seen,
        frame_last_seen=frame_last_seen,
        total_iterations=total_iterations,
        global_time_s=last_global_cumulative_seconds,
        peak_memory_mib=peak_memory_mib,
        sparsity_min_pct=smin,
        sparsity_max_pct=smax,
        sparsity_avg_pct=savg,
        sparsity_samples=len(sparsities),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="Path to the log file (e.g., log.txt)")
    ap.add_argument("--start-frame", type=int, default=0, help="Start frame (inclusive). Default: 0")
    ap.add_argument("--end-frame", type=int, default=None, help="End frame (inclusive). Default: end of log")
    ap.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = ap.parse_args()

    st = parse_log(args.log, start_frame=args.start_frame, end_frame=args.end_frame)

    if args.json:
        print(json.dumps(asdict(st), indent=2))
        return

    if st.end_frame is None:
        range_str = f"[{st.start_frame}, end]"
    else:
        range_str = f"[{st.start_frame}, {st.end_frame}]"

    print("=== Parsed Log Summary ===")
    print(f"Parsed frame range:       {range_str}")
    print(f"Frames seen in range:     {st.frames_seen_in_range} "
          f"(first={st.frame_first_seen if st.frame_first_seen is not None else 'n/a'}, "
          f"last={st.frame_last_seen if st.frame_last_seen is not None else 'n/a'})")
    print(f"Test case:                {st.test_case if st.test_case else '(not found)'}")
    print(f"Model type:               {st.model_type if st.model_type else '(not found)'}")
    print(f"Total iterations:         {st.total_iterations}")

    if st.global_time_s is None:
        print("Global time:              (not found in range)")
    else:
        print(f"Global time:              {format_seconds_hms(st.global_time_s)}  ({st.global_time_s:.6f} s)")

    if st.peak_memory_mib is None:
        print("Peak memory (MiB):        (not found in range)")
    else:
        print(f"Peak memory (MiB):        {st.peak_memory_mib}")

    if st.sparsity_samples == 0:
        print("Hessian sparsity (%):     (not found in range)")
    else:
        print(
            "Hessian sparsity (%):     "
            f"min={st.sparsity_min_pct:.12g}  "
            f"max={st.sparsity_max_pct:.12g}  "
            f"avg={st.sparsity_avg_pct:.12g}  "
            f"(n={st.sparsity_samples})"
        )


if __name__ == "__main__":
    main()