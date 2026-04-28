#!/usr/bin/env python3
"""Benchmark script for comparing LuisaRender integrators across scenes.

Compares MegaPath vs WavePath integrators, with parameter sweeps
(samples_per_pass, use_ser) for MegaPath.
"""

import argparse
import csv
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BINARY = "luisa-render-cli"

# Default integrator properties
INTEGRATOR_DEFAULTS = {
    "depth": 16,
    "rr_depth": 5,
    "rr_threshold": 0.95,
}

# (use_ser, ser_hit, ser_material, ser_rr, label)
MEGAPATH_CONFIGS = [
    (False, None,  None,  None,  "ser=off"),
    (True,  False, False, False, "ser hints=none"),
    (True,  True,  False, False, "ser hint=hit"),
    (True,  False, True,  False, "ser hint=material"),
    (True,  False, False, True,  "ser hint=rr"),
    (True,  True,  True,  True,  "ser hints=all"),
]


def find_scenes(scenes_dir: Path, filter_names: list[str] | None = None) -> list[Path]:
    """Find all .luisa scene files under scenes/."""
    scene_files = sorted(scenes_dir.rglob("scene.luisa"))
    if filter_names:
        filtered = []
        for sf in scene_files:
            # Match against the scene directory name (first component under scenes/)
            rel = sf.relative_to(scenes_dir)
            scene_name = rel.parts[0]
            if scene_name in filter_names:
                filtered.append(sf)
        scene_files = filtered
    return scene_files


def scene_display_name(scenes_dir: Path, scene_file: Path) -> str:
    """Get a human-readable name for a scene."""
    return str(scene_file.relative_to(scenes_dir).parent)


def make_integrator_block(integrator_type: str, samples_per_pass: int | None = None,
                          use_ser: bool | None = None, ser_hit: bool | None = None,
                          ser_material: bool | None = None, ser_rr: bool | None = None) -> str:
    """Generate an integrator block string."""
    def b(v): return "true" if v else "false"
    lines = [f"  integrator : {integrator_type} {{"]
    lines.append("    sampler : Independent {}")
    lines.append(f"    depth {{ {INTEGRATOR_DEFAULTS['depth']} }}")
    lines.append(f"    rr_depth {{ {INTEGRATOR_DEFAULTS['rr_depth']} }}")
    lines.append(f"    rr_threshold {{ {INTEGRATOR_DEFAULTS['rr_threshold']} }}")
    if samples_per_pass is not None:
        lines.append(f"    samples_per_pass {{ {samples_per_pass} }}")
    if use_ser is not None:
        lines.append(f"    use_ser {{ {b(use_ser)} }}")
    if ser_hit is not None:
        lines.append(f"    ser_hit {{ {b(ser_hit)} }}")
    if ser_material is not None:
        lines.append(f"    ser_material {{ {b(ser_material)} }}")
    if ser_rr is not None:
        lines.append(f"    ser_rr {{ {b(ser_rr)} }}")
    lines.append("  }")
    return "\n".join(lines)


def _find_balanced_brace_end(text: str, open_pos: int) -> int:
    """Given position of an opening '{', return position of its matching '}'."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i
    return -1


def _replace_braced_block(text: str, pattern: str, replacement: str) -> str:
    """Find `pattern` followed by a braced block `{ ... }` and replace the whole thing.

    Uses brace counting to correctly handle arbitrarily nested braces.
    """
    m = re.search(pattern, text)
    if not m:
        return text
    # Find the opening brace after the match
    rest = text[m.end():]
    brace_offset = rest.find('{')
    if brace_offset == -1:
        return text
    open_pos = m.end() + brace_offset
    close_pos = _find_balanced_brace_end(text, open_pos)
    if close_pos == -1:
        return text
    # Replace from start of match through closing brace
    return text[:m.start()] + replacement + text[close_pos + 1:]


def rewrite_scene(scene_content: str, integrator_block: str) -> str:
    """Rewrite the integrator in a scene file.

    Handles two patterns:
    1. Inline: `integrator : Type { ... }` inside render block
    2. Named: `Integrator name : Type { ... }` as top-level + `integrator { @name }` in render
    """
    # Remove top-level named integrator definitions: Integrator <name> : <Type> { ... }
    scene_content = _replace_braced_block(
        scene_content,
        r'Integrator\s+\w+\s*:\s*\w+\s*',
        ''
    )

    # Replace inline integrator: integrator : Type { ... }
    replaced = _replace_braced_block(
        scene_content,
        r'integrator\s*:\s*\w+\s*',
        integrator_block.strip()
    )

    # Replace reference-style integrator: integrator { @name }
    replaced = re.sub(
        r'integrator\s*\{\s*@\w+\s*\}',
        integrator_block.strip(),
        replaced
    )

    return replaced


def rewrite_spp(scene_content: str, spp: int) -> str:
    """Override spp value in scene file."""
    # Replace spp { N } pattern
    return re.sub(r'spp\s*\{\s*\d+\s*\}', f'spp {{ {spp} }}', scene_content)


def write_temp_scene(scene_file: Path, content: str) -> Path:
    """Write modified scene to a temp file in the same directory (for relative paths)."""
    scene_dir = scene_file.parent
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    tmp_path = scene_dir / f".bench_{h}.luisa"
    tmp_path.write_text(content)
    return tmp_path


def parse_render_times(output: str) -> dict:
    """Parse timing information from renderer output.

    Returns dict with:
      render_ms: rendering time in ms (from "Rendering finished in X ms.")
      compile_ms: shader compile time in ms (from "Integrator shader compile in X ms.")
    """
    times = {}
    m = re.search(r'Rendering finished in ([\d.]+) ms', output)
    if m:
        times["render_ms"] = float(m.group(1))
    m = re.search(r'Integrator shader compile in ([\d.]+) ms', output)
    if m:
        times["compile_ms"] = float(m.group(1))
    return times


def run_render(binary: str, backend: str, scene_path: Path, device: int = 0,
               timeout: int | None = None) -> dict:
    """Run the renderer and return a result dict."""
    cmd = [binary, "-b", backend, "-d", str(device), "--scene", str(scene_path)]
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout
        )
        elapsed = time.monotonic() - start
        output = result.stdout + result.stderr
        times = parse_render_times(output)
        return {
            "elapsed": elapsed,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "cmd": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            **times,
        }
    except subprocess.TimeoutExpired as e:
        elapsed = time.monotonic() - start
        return {
            "elapsed": elapsed,
            "ok": False,
            "returncode": None,
            "cmd": cmd,
            "stdout": e.stdout or "",
            "stderr": e.stderr or "",
            "timeout": True,
        }


def print_render_failure(r: dict):
    """Print detailed failure info for a render run."""
    print(f"  Command: {' '.join(r['cmd'])}")
    if r.get("timeout"):
        print(f"  TIMEOUT after {r['elapsed']:.1f}s")
    else:
        print(f"  Exit code: {r['returncode']}")
    if r["stderr"].strip():
        print(f"  Stderr:\n    " + "\n    ".join(r["stderr"].strip().splitlines()[-30:]))
    if r["stdout"].strip():
        print(f"  Stdout:\n    " + "\n    ".join(r["stdout"].strip().splitlines()[-15:]))


def _fmt_ms(ms):
    """Format milliseconds as a human-readable string."""
    if ms is None:
        return "-"
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.1f}ms"


def print_table(results: list[dict]):
    """Print results as a formatted table."""
    if not results:
        return
    headers = ["Scene", "Integrator", "spp_per_pass", "ser_config",
               "Render", "Compile", "Wall", "Status"]
    keys = ["scene", "integrator", "samples_per_pass", "ser_config",
            "render_ms", "compile_ms", "wall_ms", "status"]
    col_widths = [max(len(headers[i]), max(len(str(r.get(k, ""))) for r in results))
                  for i, k in enumerate(keys)]

    def fmt_row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in results:
        print(fmt_row([
            r['scene'], r['integrator'],
            r.get('samples_per_pass', '-'),
            r.get('ser_config', '-'),
            _fmt_ms(r.get('render_ms')),
            _fmt_ms(r.get('compile_ms')),
            _fmt_ms(r.get('wall_ms')),
            r['status'],
        ]))
    print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark LuisaRender integrators")
    parser.add_argument("-b", "--backend", default="cuda", help="Compute backend (default: cuda)")
    parser.add_argument("-d", "--device", type=int, default=0, help="Device index (default: 0)")
    parser.add_argument("--scenes", nargs="*", help="Filter to specific scene names (e.g. bathroom staircase2)")
    parser.add_argument("--spp", type=int, default=None, help="Override SPP for all scenes")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per render in seconds")
    parser.add_argument("--output", "-o", default="benchmark_results.csv", help="Output CSV file")
    parser.add_argument("--binary", type=str, default=None, help="Path to luisa-render-cli binary")
    parser.add_argument("--no-sweep", action="store_true",
                        help="Skip MegaPath parameter sweep, only run defaults")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    binary = args.binary if args.binary else BINARY
    if not shutil.which(binary):
        print(f"Error: binary '{binary}' not found in PATH", file=sys.stderr)
        print("Install luisa-render-cli or pass --binary <path>", file=sys.stderr)
        sys.exit(1)

    scenes_dir = REPO_ROOT / "scenes"
    scene_files = find_scenes(scenes_dir, args.scenes)
    if not scene_files:
        print("No scenes found!", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(scene_files)} scenes:")
    for sf in scene_files:
        print(f"  {scene_display_name(scenes_dir, sf)}")
    print(f"Backend: {args.backend}, Device: {args.device}")
    if args.spp:
        print(f"SPP override: {args.spp}")
    print()

    results = []
    tmp_files = []

    try:
        for scene_file in scene_files:
            name = scene_display_name(scenes_dir, scene_file)
            original = scene_file.read_text()

            # --- WavePath ---
            print(f"[{name}] WavePath ...", end=" ", flush=True)
            integrator_block = make_integrator_block("WavePath")
            modified = rewrite_scene(original, integrator_block)
            if args.spp:
                modified = rewrite_spp(modified, args.spp)
            tmp = write_temp_scene(scene_file, modified)
            tmp_files.append(tmp)

            if args.dry_run:
                print(f"(dry run) cmd: {binary} -b {args.backend} --scene {tmp}")
            else:
                r = run_render(binary, args.backend, tmp, args.device, args.timeout)
                status = "ok" if r["ok"] else ("TIMEOUT" if r.get("timeout") else "FAIL")
                render_str = _fmt_ms(r.get("render_ms")) if r.get("render_ms") else f"{r['elapsed']:.1f}s (wall)"
                print(f"{render_str} [{status}]")
                if not r["ok"]:
                    print_render_failure(r)
                results.append({
                    "scene": name, "integrator": "WavePath",
                    "samples_per_pass": "-", "ser_config": "-",
                    "use_ser": "-", "ser_hit": "-", "ser_material": "-", "ser_rr": "-",
                    "render_ms": r.get("render_ms"),
                    "compile_ms": r.get("compile_ms"),
                    "wall_ms": r["elapsed"] * 1000,
                    "status": status,
                })

            # --- MegaPath runs ---
            if args.no_sweep:
                configs = [(True, True, True, True, "ser hints=all")]
            else:
                configs = MEGAPATH_CONFIGS

            for use_ser, ser_hit, ser_material, ser_rr, ser_label in configs:
                label = f"MegaPath {ser_label}"
                print(f"[{name}] {label} ...", end=" ", flush=True)
                integrator_block = make_integrator_block(
                    "MegaPath", samples_per_pass=1,
                    use_ser=use_ser, ser_hit=ser_hit, ser_material=ser_material, ser_rr=ser_rr,
                )
                modified = rewrite_scene(original, integrator_block)
                if args.spp:
                    modified = rewrite_spp(modified, args.spp)
                tmp = write_temp_scene(scene_file, modified)
                tmp_files.append(tmp)

                if args.dry_run:
                    print(f"(dry run)")
                else:
                    r = run_render(
                        binary, args.backend, tmp, args.device, args.timeout
                    )
                    status = "ok" if r["ok"] else ("TIMEOUT" if r.get("timeout") else "FAIL")
                    render_str = _fmt_ms(r.get("render_ms")) if r.get("render_ms") else f"{r['elapsed']:.1f}s (wall)"
                    print(f"{render_str} [{status}]")
                    if not r["ok"]:
                        print_render_failure(r)
                    results.append({
                        "scene": name, "integrator": "MegaPath",
                        "samples_per_pass": 1, "ser_config": ser_label,
                        "use_ser": use_ser, "ser_hit": ser_hit,
                        "ser_material": ser_material, "ser_rr": ser_rr,
                        "render_ms": r.get("render_ms"),
                        "compile_ms": r.get("compile_ms"),
                        "wall_ms": r["elapsed"] * 1000,
                        "status": status,
                    })

    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        # Clean up temp files
        for tmp in tmp_files:
            try:
                tmp.unlink()
            except OSError:
                pass

    if results:
        print_table(results)

        # Write CSV
        csv_path = Path(args.output)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "scene", "integrator", "samples_per_pass", "ser_config",
                "use_ser", "ser_hit", "ser_material", "ser_rr",
                "render_ms", "compile_ms", "wall_ms", "status",
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
