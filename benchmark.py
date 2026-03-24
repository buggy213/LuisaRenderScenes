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
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BINARY = REPO_ROOT / "cmake-build-release" / "bin" / "luisa-render-cli"

# Default integrator properties
INTEGRATOR_DEFAULTS = {
    "depth": 16,
    "rr_depth": 5,
    "rr_threshold": 0.95,
}

MEGAPATH_SPP_VALUES = [1, 2, 4, 8]
MEGAPATH_SER_VALUES = [True, False]


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
                          use_ser: bool | None = None) -> str:
    """Generate an integrator block string."""
    lines = [f"  integrator : {integrator_type} {{"]
    lines.append("    sampler : Independent {}")
    lines.append(f"    depth {{ {INTEGRATOR_DEFAULTS['depth']} }}")
    lines.append(f"    rr_depth {{ {INTEGRATOR_DEFAULTS['rr_depth']} }}")
    lines.append(f"    rr_threshold {{ {INTEGRATOR_DEFAULTS['rr_threshold']} }}")
    if samples_per_pass is not None:
        lines.append(f"    samples_per_pass {{ {samples_per_pass} }}")
    if use_ser is not None:
        lines.append(f"    use_ser {{ {'true' if use_ser else 'false'} }}")
    lines.append("  }")
    return "\n".join(lines)


def rewrite_scene(scene_content: str, integrator_block: str) -> str:
    """Rewrite the integrator in a scene file.

    Handles two patterns:
    1. Inline: `integrator : Type { ... }` inside render block
    2. Named: `Integrator name : Type { ... }` as top-level + `integrator { @name }` in render
    """
    # Remove top-level named integrator definitions
    # Pattern: Integrator <name> : <Type> { ... }
    scene_content = re.sub(
        r'Integrator\s+\w+\s*:\s*\w+\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}\s*\n?',
        '',
        scene_content
    )

    # Replace inline integrator block (with nested braces)
    # Match: integrator : Type { ... } handling one level of nesting
    replaced = re.sub(
        r'integrator\s*:\s*\w+\s*\{[^}]*(?:\{[^}]*\}[^}]*)*\}',
        integrator_block.strip(),
        scene_content
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


def run_render(binary: Path, backend: str, scene_path: Path, device: int = 0,
               timeout: int | None = None) -> tuple[float, bool, str]:
    """Run the renderer and return (elapsed_seconds, success, output)."""
    cmd = [str(binary), "-b", backend, "-d", str(device), "--scene", str(scene_path)]
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout
        )
        elapsed = time.monotonic() - start
        output = result.stdout + result.stderr
        return elapsed, result.returncode == 0, output
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return elapsed, False, "TIMEOUT"


def print_table(results: list[dict]):
    """Print results as a formatted table."""
    if not results:
        return
    headers = ["Scene", "Integrator", "spp_per_pass", "use_ser", "Time (s)", "Status"]
    col_widths = [max(len(headers[i]), max(len(str(r.get(h, ""))) for r in results))
                  for i, h in enumerate(["scene", "integrator", "samples_per_pass", "use_ser", "time", "status"])]

    def fmt_row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_widths))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in results:
        t = f"{r['time']:.1f}" if r['status'] == 'ok' else r['status']
        print(fmt_row([
            r['scene'], r['integrator'],
            r.get('samples_per_pass', '-'),
            r.get('use_ser', '-'),
            t, r['status']
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
    parser.add_argument("--spp-values", nargs="*", type=int, default=MEGAPATH_SPP_VALUES,
                        help="samples_per_pass values to sweep (default: 1 2 4 8)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    binary = Path(args.binary) if args.binary else BINARY
    if not binary.exists():
        print(f"Error: binary not found at {binary}", file=sys.stderr)
        print("Build with cmake first, or pass --binary <path>", file=sys.stderr)
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
                elapsed, ok, output = run_render(binary, args.backend, tmp, args.device, args.timeout)
                status = "ok" if ok else "FAIL"
                print(f"{elapsed:.1f}s [{status}]")
                if not ok:
                    print(f"  Output: {output[-500:]}")
                results.append({
                    "scene": name, "integrator": "WavePath",
                    "samples_per_pass": "-", "use_ser": "-",
                    "time": elapsed, "status": status,
                })

            # --- MegaPath runs ---
            if args.no_sweep:
                # Single MegaPath run with defaults
                configs = [(1, True)]
            else:
                configs = [
                    (spp_val, ser_val)
                    for spp_val in args.spp_values
                    for ser_val in MEGAPATH_SER_VALUES
                ]

            for spp_val, ser_val in configs:
                label = f"MegaPath spp_pass={spp_val} ser={ser_val}"
                print(f"[{name}] {label} ...", end=" ", flush=True)
                integrator_block = make_integrator_block(
                    "MegaPath", samples_per_pass=spp_val, use_ser=ser_val
                )
                modified = rewrite_scene(original, integrator_block)
                if args.spp:
                    modified = rewrite_spp(modified, args.spp)
                tmp = write_temp_scene(scene_file, modified)
                tmp_files.append(tmp)

                if args.dry_run:
                    print(f"(dry run)")
                else:
                    elapsed, ok, output = run_render(
                        binary, args.backend, tmp, args.device, args.timeout
                    )
                    status = "ok" if ok else "FAIL"
                    print(f"{elapsed:.1f}s [{status}]")
                    if not ok:
                        print(f"  Output: {output[-500:]}")
                    results.append({
                        "scene": name, "integrator": "MegaPath",
                        "samples_per_pass": spp_val, "use_ser": ser_val,
                        "time": elapsed, "status": status,
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
                "scene", "integrator", "samples_per_pass", "use_ser", "time", "status"
            ])
            writer.writeheader()
            writer.writerows(results)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
