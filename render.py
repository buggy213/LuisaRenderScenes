#!/usr/bin/env python3
"""One-off render script for LuisaRender scenes.

Quickly render a single scene with a chosen integrator and parameter overrides.
"""

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BINARY = "luisa-render-cli"

INTEGRATOR_DEFAULTS = {
    "depth": 16,
    "rr_depth": 5,
    "rr_threshold": 0.95,
}


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def find_scene(scenes_dir: Path, name: str) -> Path | None:
    """Resolve a scene by name or path."""
    # Direct path
    p = Path(name)
    if p.exists() and p.suffix == ".luisa":
        return p.resolve()
    # scenes/<name>/scene.luisa
    candidate = scenes_dir / name / "scene.luisa"
    if candidate.exists():
        return candidate
    # Glob fallback
    matches = list(scenes_dir.rglob(f"{name}/scene.luisa"))
    return matches[0] if matches else None


def list_scenes(scenes_dir: Path) -> list[str]:
    return sorted(sf.parent.name for sf in scenes_dir.rglob("scene.luisa"))


# ---------------------------------------------------------------------------
# Scene rewriting (shared logic with benchmark.py)
# ---------------------------------------------------------------------------

def make_integrator_block(integrator_type: str, depth: int, rr_depth: int,
                          rr_threshold: float, sampler: str = "Independent",
                          samples_per_pass: int | None = None,
                          use_ser: bool | None = None,
                          unrolled: bool | None = None,
                          material_binning: str | None = None) -> str:
    lines = [f"integrator : {integrator_type} {{"]
    lines.append(f"  sampler : {sampler} {{}}")
    lines.append(f"  depth {{ {depth} }}")
    lines.append(f"  rr_depth {{ {rr_depth} }}")
    lines.append(f"  rr_threshold {{ {rr_threshold} }}")
    if samples_per_pass is not None:
        lines.append(f"  samples_per_pass {{ {samples_per_pass} }}")
    if use_ser is not None:
        lines.append(f"  use_ser {{ {'true' if use_ser else 'false'} }}")
    if unrolled is not None:
        lines.append(f"  unrolled {{ {'true' if unrolled else 'false'} }}")
    if material_binning is not None:
        lines.append(f"  material_binning {{ \"{material_binning}\" }}")
    lines.append("}")
    return "\n".join(lines)


def _find_balanced_brace_end(text: str, open_pos: int) -> int:
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
    m = re.search(pattern, text)
    if not m:
        return text
    rest = text[m.end():]
    brace_offset = rest.find('{')
    if brace_offset == -1:
        return text
    open_pos = m.end() + brace_offset
    close_pos = _find_balanced_brace_end(text, open_pos)
    if close_pos == -1:
        return text
    return text[:m.start()] + replacement + text[close_pos + 1:]


def rewrite_integrator(scene_content: str, integrator_block: str) -> str:
    # Remove top-level named integrator definitions: Integrator <name> : <Type> { ... }
    scene_content = _replace_braced_block(
        scene_content, r'Integrator\s+\w+\s*:\s*\w+\s*', ''
    )
    # Replace inline integrator: integrator : Type { ... }
    replaced = _replace_braced_block(
        scene_content, r'integrator\s*:\s*\w+\s*', integrator_block.strip()
    )
    # Replace reference-style integrator: integrator { @name }
    replaced = re.sub(
        r'integrator\s*\{\s*@\w+\s*\}', integrator_block.strip(), replaced
    )
    return replaced


def rewrite_spp(scene_content: str, spp: int) -> str:
    return re.sub(r'spp\s*\{\s*\d+\s*\}', f'spp {{ {spp} }}', scene_content)


def rewrite_resolution(scene_content: str, width: int, height: int) -> str:
    return re.sub(
        r'resolution\s*\{\s*\d+\s*,\s*\d+\s*\}',
        f'resolution {{ {width}, {height} }}',
        scene_content,
    )


def rewrite_output_file(scene_content: str, filename: str) -> str:
    """Override the output filename in the camera block."""
    return re.sub(
        r'(file\s*\{\s*)"[^"]*"(\s*\})',
        rf'\g<1>"{filename}"\2',
        scene_content,
    )


def write_temp_scene(scene_file: Path, content: str) -> Path:
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    tmp_path = scene_file.parent / f".render_{h}.luisa"
    tmp_path.write_text(content)
    return tmp_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def parse_render_times(output: str) -> dict:
    times = {}
    m = re.search(r'Rendering finished in ([\d.]+) ms', output)
    if m:
        times["render_ms"] = float(m.group(1))
    m = re.search(r'Integrator shader compile in ([\d.]+) ms', output)
    if m:
        times["compile_ms"] = float(m.group(1))
    return times


def _fmt_ms(ms) -> str:
    if ms is None:
        return "-"
    return f"{ms / 1000:.2f}s" if ms >= 1000 else f"{ms:.1f}ms"


def run_render(binary: str, backend: str, scene_path: Path,
               device: int = 0, timeout: int | None = None) -> dict:
    cmd = [binary, "-b", backend, "-d", str(device), "--scene", str(scene_path)]
    start = time.monotonic()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed = time.monotonic() - start
        output = result.stdout + result.stderr
        return {
            "elapsed": elapsed,
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "cmd": cmd,
            "stdout": result.stdout,
            "stderr": result.stderr,
            **parse_render_times(output),
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_resolution(s: str) -> tuple[int, int]:
    sep = "x" if "x" in s.lower() else ","
    parts = s.lower().split(sep)
    if len(parts) != 2:
        raise ValueError(s)
    return int(parts[0]), int(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Render a single LuisaRender scene with a specified integrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s bathroom
  %(prog)s bedroom --integrator MegaPath --samples-per-pass 4 --use-ser
  %(prog)s staircase2 --integrator WGPath --samples-per-pass 4 --unrolled
  %(prog)s staircase2 --spp 512 --resolution 1920x1080 --output out.exr
  %(prog)s scenes/bathroom/scene.luisa -i WavePath --spp 256 --dry-run
  %(prog)s --list
        """,
    )

    parser.add_argument("scene", nargs="?", help="Scene name or path to .luisa file")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List available scenes and exit")

    # --- Integrator ---
    integ = parser.add_argument_group("integrator")
    integ.add_argument("--integrator", "-i", default="WavePath",
                       choices=["WavePath", "MegaPath", "WGPath"],
                       help="Integrator type (default: WavePath)")
    integ.add_argument("--depth", type=int, default=INTEGRATOR_DEFAULTS["depth"],
                       help=f"Ray depth (default: {INTEGRATOR_DEFAULTS['depth']})")
    integ.add_argument("--rr-depth", type=int, default=INTEGRATOR_DEFAULTS["rr_depth"],
                       help=f"Russian roulette depth (default: {INTEGRATOR_DEFAULTS['rr_depth']})")
    integ.add_argument("--rr-threshold", type=float, default=INTEGRATOR_DEFAULTS["rr_threshold"],
                       help=f"Russian roulette threshold (default: {INTEGRATOR_DEFAULTS['rr_threshold']})")
    integ.add_argument("--sampler", default="Independent",
                       choices=["Independent", "PMJ02BN"],
                       help="Sampler type (default: Independent)")

    # MegaPath / WGPath shared
    progressive = parser.add_argument_group("MegaPath / WGPath options")
    progressive.add_argument("--samples-per-pass", type=int, default=None, metavar="N",
                             help="samples_per_pass (MegaPath and WGPath)")

    # MegaPath-specific
    mega = parser.add_argument_group("MegaPath options")
    ser = mega.add_mutually_exclusive_group()
    ser.add_argument("--use-ser", dest="use_ser", action="store_true", default=None,
                     help="Enable SER")
    ser.add_argument("--no-ser", dest="use_ser", action="store_false",
                     help="Disable SER")

    # WGPath-specific
    wg = parser.add_argument_group("WGPath options")
    wg.add_argument("--unrolled", action="store_true", default=False,
                    help="Enable unrolled mode")
    wg.add_argument("--material-binning", default=None, metavar="MODE",
                    help="Material binning mode: full, none, ... (default: full)")

    # --- Scene overrides ---
    overrides = parser.add_argument_group("scene overrides")
    overrides.add_argument("--spp", type=int, default=None,
                           help="Override samples per pixel")
    overrides.add_argument("--resolution", "-r", default=None, metavar="WxH",
                           help="Override resolution (e.g. 1920x1080 or 1920,1080)")
    overrides.add_argument("--output", "-o", default=None, metavar="FILE",
                           help="Override output filename (e.g. result.exr)")

    # --- Backend ---
    backend = parser.add_argument_group("backend")
    backend.add_argument("--backend", "-b", default="cuda",
                         help="Compute backend (default: cuda)")
    backend.add_argument("--device", "-d", type=int, default=0,
                         help="Device index (default: 0)")
    backend.add_argument("--timeout", type=int, default=None, metavar="SECS",
                         help="Timeout in seconds")
    backend.add_argument("--binary", default=None, metavar="PATH",
                         help="Path to luisa-render-cli binary")

    # --- Misc ---
    parser.add_argument("--dry-run", action="store_true",
                        help="Print modified scene and command without executing")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the temporary scene file after rendering")

    args = parser.parse_args()
    scenes_dir = REPO_ROOT / "scenes"

    if args.list:
        print("Available scenes:")
        for name in list_scenes(scenes_dir):
            print(f"  {name}")
        return 0

    if not args.scene:
        parser.error("scene is required (use --list to see available scenes)")

    binary = args.binary or BINARY
    if not args.dry_run and not shutil.which(binary):
        print(f"error: binary '{binary}' not found in PATH", file=sys.stderr)
        print("install luisa-render-cli or pass --binary <path>", file=sys.stderr)
        return 1

    scene_file = find_scene(scenes_dir, args.scene)
    if scene_file is None:
        print(f"error: scene '{args.scene}' not found", file=sys.stderr)
        print(f"available: {', '.join(list_scenes(scenes_dir))}", file=sys.stderr)
        return 1

    # Parse resolution early so we can error before doing any work
    width = height = None
    if args.resolution:
        try:
            width, height = parse_resolution(args.resolution)
        except ValueError:
            print(f"error: invalid resolution '{args.resolution}' — use WxH or W,H",
                  file=sys.stderr)
            return 1

    # --- Summary ---
    print(f"scene:      {scene_file.relative_to(REPO_ROOT)}")
    extras = []
    if args.integrator in ("MegaPath", "WGPath") and args.samples_per_pass is not None:
        extras.append(f"spp_pass={args.samples_per_pass}")
    if args.integrator == "MegaPath" and args.use_ser is not None:
        extras.append(f"ser={'yes' if args.use_ser else 'no'}")
    if args.integrator == "WGPath":
        if args.unrolled:
            extras.append("unrolled")
        if args.material_binning is not None:
            extras.append(f"binning={args.material_binning}")
    integ_label = args.integrator + (f" ({', '.join(extras)})" if extras else "")
    print(f"integrator: {integ_label}")
    print(f"backend:    {args.backend}, device {args.device}")
    if args.spp is not None:
        print(f"spp:        {args.spp}")
    if width is not None:
        print(f"resolution: {width}x{height}")
    if args.output is not None:
        print(f"output:     {args.output}")

    # --- Rewrite scene ---
    integrator_block = make_integrator_block(
        args.integrator,
        depth=args.depth,
        rr_depth=args.rr_depth,
        rr_threshold=args.rr_threshold,
        sampler=args.sampler,
        samples_per_pass=args.samples_per_pass if args.integrator in ("MegaPath", "WGPath") else None,
        use_ser=args.use_ser if args.integrator == "MegaPath" else None,
        unrolled=True if (args.integrator == "WGPath" and args.unrolled) else None,
        material_binning=args.material_binning if args.integrator == "WGPath" else None,
    )

    content = scene_file.read_text()
    content = rewrite_integrator(content, integrator_block)
    if args.spp is not None:
        content = rewrite_spp(content, args.spp)
    if width is not None:
        content = rewrite_resolution(content, width, height)
    if args.output is not None:
        content = rewrite_output_file(content, args.output)

    tmp_file = write_temp_scene(scene_file, content)

    try:
        if args.dry_run:
            print(f"\ntemp scene: {tmp_file}")
            print(f"command:    {binary} -b {args.backend} -d {args.device} --scene {tmp_file}")
            print(f"\n--- integrator block ---\n{integrator_block}")
            if args.keep:
                print(f"\nkept: {tmp_file}")
                tmp_file = None  # don't clean up in finally
            return 0

        print()
        r = run_render(binary, args.backend, tmp_file, args.device, args.timeout)

        if r.get("timeout"):
            print(f"TIMEOUT after {r['elapsed']:.1f}s")
            return 2
        elif not r["ok"]:
            print(f"FAILED (exit {r['returncode']}) in {r['elapsed']:.1f}s")
            for line in (r["stderr"] + r["stdout"]).strip().splitlines()[-30:]:
                print(f"  {line}")
            return 1
        else:
            print(
                f"done  render={_fmt_ms(r.get('render_ms'))}"
                f"  compile={_fmt_ms(r.get('compile_ms'))}"
                f"  wall={r['elapsed']:.2f}s"
            )
            return 0

    finally:
        if tmp_file is not None and not args.keep:
            try:
                tmp_file.unlink()
            except OSError:
                pass
        elif tmp_file is not None and args.keep:
            print(f"kept temp scene: {tmp_file}")


if __name__ == "__main__":
    sys.exit(main())
