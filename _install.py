
"""Installer automation for the ComfyUI workspace described in this repo.

This script reproduces the manual shell steps listed in the original
docstring by orchestrating `uv`, the `comfy` CLI, the required nodes, and the
model downloads defined in `_installer_files/model_fetch.json`.

Example usage:

    python _install.py

You can rerun the script at any time. Existing virtualenvs, nodes, and model
files are detected and reused unless you pass `--force-model-downloads`.
"""

# from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parent
INSTALLER_DIR = REPO_ROOT / "_installer_files"
MODEL_MANIFEST = INSTALLER_DIR / "model_fetch.json"
WORKSPACE_NAME = "ComfyUI"
NODE_PACKAGES = ["ComfyUI-GGUF", "comfyui-kjnodes", "ComfyUI-VideoHelperSuite", "comfyui-tooling-nodes"]
CACHE_DIR = Path.home() / ".cache" / "temp_comfy_installer"

class InstallError(RuntimeError):
    """Raised when a blocking installation step fails."""


def log(message: str) -> None:
    """Simple logger for consistent installer output."""

    print(f"[install] {message}")


def ensure_uv_available(uv_cmd: str) -> None:
    """Verify that the uv executable exists early."""

    if shutil.which(uv_cmd):
        return
    raise InstallError(
        textwrap.dedent(
            f"""Could not find '{uv_cmd}' on PATH. Install uv first:
            https://docs.astral.sh/uv/getting-started/installation/
            """
        ).strip()
    )


def run_command(cmd: Sequence[str], *, cwd: Path | None = None) -> None:
    """Run a subprocess, streaming stdout/stderr to the terminal."""

    log("$ " + " ".join(cmd))
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - subprocess failure
        raise InstallError(f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}")


def create_virtualenv(uv_cmd: str, venv_path: Path) -> None:
    """Ensure the .venv directory exists by calling `uv venv`."""

    if venv_path.exists():
        log(f"Virtualenv already present at {venv_path}")
        return
    run_command([uv_cmd, "venv", str(venv_path)])

def uv_pip_install(uv_cmd: str, python_exe: Path, packages: Iterable[str], no_build_isolation: bool = False) -> None:
    """Install packages into the created venv via uv pip."""

    for package in packages:
        cmd = [uv_cmd, "pip", "install", "--python", str(python_exe), package]
        if no_build_isolation:
            cmd.append("--no-build-isolation")
        run_command(cmd)

def uv_pip_install_reqs(uv_cmd: str, python_exe: Path, reqs_path: Path) -> None:
    """Install packages into the created venv via uv pip from a requirements file."""
    run_command([uv_cmd, "pip", "install", "--python", str(python_exe), "-r", str(reqs_path)])

def comfy_cli_path(venv_path: Path) -> Path:
    """Return the comfy executable inside the venv."""

    bin_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
    exe_name = "comfy.exe" if os.name == "nt" else "comfy"
    comfy_path = bin_dir / exe_name
    if not comfy_path.exists():
        raise InstallError(
            f"Could not find the comfy CLI at {comfy_path}. "
            "Did the comfy-cli installation step succeed?"
        )
    return comfy_path


def run_comfy_command(comfy_exe: Path, workspace: Path, *args: str) -> None:
    """Execute a comfy CLI command for the configured workspace."""

    run_command([str(comfy_exe), "--workspace", str(workspace), *args])


def install_workspace(comfy_exe: Path, workspace: Path, *, fast_deps: bool = True) -> None:
    """Invoke `comfy install` to set up the workspace directory."""

    args = ["install"]
    if fast_deps:
        args.append("--fast-deps")
    run_comfy_command(comfy_exe, workspace, *args)


def initialize_comfy(comfy_exe: Path, workspace: Path) -> None:
    """Run `comfy initialize` to set up initial configuration files."""

    run_comfy_command(comfy_exe, workspace, "launch", "--background")
    run_comfy_command(comfy_exe, workspace, "stop")
    

def install_nodes(comfy_exe: Path, workspace: Path, nodes: Sequence[str], uv_cmd: str, python_exe: Path) -> None:
    """Install each required node package via the comfy CLI."""

    for node in nodes:
        run_comfy_command(comfy_exe, workspace, "node", "install", node)
        
    custom_node_dir = workspace / "custom_nodes"
    for node in nodes:
        if not (custom_node_dir / node).exists():
            print(f"Node installation appears to have failed for {node}, attempting manual install")
            if not try_manual_node_install(custom_node_dir, node, uv_cmd, python_exe):
                raise InstallError(f"Node installation failed for {node}, aborting.")
            else:
                print(f"Manual installation succeeded for {node}.")

def try_manual_node_install(custom_node_dir: Path, node: str, uv_cmd: str, python_exe: Path) -> None:
    """Attempt to manually install a node package by downloading and extracting it."""

    if node == "ComfyUI-VideoHelperSuite":
        url = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git" 
     
    run_command(["git", "clone", url, str(custom_node_dir / node)])
    if not (custom_node_dir / node).exists():
        return False
    uv_pip_install_reqs(uv_cmd, python_exe, custom_node_dir / node / "requirements.txt")
    return True

def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Load the JSON manifest describing required model downloads."""

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise InstallError(f"Model manifest not found: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise InstallError(f"Model manifest is invalid JSON: {manifest_path}") from exc

    if not isinstance(data, list):
        raise InstallError("Model manifest root must be a list of entries.")
    return data  # type: ignore[return-value]


def download_file(url: str, destination: Path) -> None:
    """Download a URL to the destination path, showing incremental progress."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {url} -> {destination}")
    run_command(["wget", url, "-O", str(destination)])

def fetch_models(workspace: Path, manifest_path: Path, *, force: bool = False) -> None:
    """Fetch all models listed in the manifest into the ComfyUI workspace."""

    manifest = read_manifest(manifest_path)
    for entry in manifest:
        try:
            relative_path = entry["path"]
            url = entry["url"]
        except KeyError as exc:
            raise InstallError("Each manifest entry must contain 'path' and 'url'.") from exc

        destination = workspace / relative_path
        cache_path = CACHE_DIR / url.replace("/", "_").replace(":", "")
        
        if destination.exists() and not force:
            log(f"Model already present, skipping: {destination}")
            continue
        
        if cache_path.exists() and not force:
            log(f"Model found in cache, copying: {cache_path} -> {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, destination)
        else:
            download_file(url, cache_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, destination)

def setup_samples(workspace: Path) -> None:
    """Set up sample files in the workspace (if any)."""

    sample_input = INSTALLER_DIR / "unnamed.jpg"
    sample_input_destination = workspace / "input" / "unnamed.jpg"
    if not sample_input_destination.exists():
        sample_input_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample_input, sample_input_destination)
        log(f"Copied sample input to {sample_input_destination}")
    else:
        log(f"Sample input already present, skipping: {sample_input_destination}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        default=WORKSPACE_NAME,
        help="Name or path for the ComfyUI workspace (default: %(default)s)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MODEL_MANIFEST,
        help="Path to the model manifest JSON",
    )
    parser.add_argument(
        "--uv",
        default="uv",
        help="Name/path of the uv executable (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-model-downloads",
        action="store_true",
        help="Skip fetching models (useful for offline or cached installs)",
    )
    parser.add_argument(
        "--force-model-downloads",
        action="store_true",
        help="Re-download models even if the target files already exist",
    )
    parser.add_argument(
        "--node",
        action="append",
        dest="extra_nodes",
        default=[],
        help="Additional node packages to install (repeatable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_uv_available(args.uv)

    workspace = (REPO_ROOT / args.workspace).resolve()
    venv_path = REPO_ROOT / ".venv"
    python_exe = venv_path / ("Scripts" if os.name == "nt" else "bin") / (
        "python.exe" if os.name == "nt" else "python"
    )

    create_virtualenv(args.uv, venv_path)
    uv_pip_install(args.uv, python_exe, ["comfy-cli"])

    comfy_exe = comfy_cli_path(venv_path)
    install_workspace(comfy_exe, workspace)

    initialize_comfy(comfy_exe, workspace)

    uv_pip_install(args.uv, python_exe, ["sageattention"], no_build_isolation=True)

    nodes = NODE_PACKAGES + list(args.extra_nodes)
    install_nodes(comfy_exe, workspace, nodes, args.uv, python_exe)
    
    setup_samples(workspace)

    uv_pip_install_reqs(args.uv, python_exe, INSTALLER_DIR / "additional_requirements.txt")

    if not args.skip_model_downloads:
        fetch_models(workspace, args.manifest, force=args.force_model_downloads)
    else:
        log("Skipping model downloads per CLI flag")

    log("Installation complete")


if __name__ == "__main__":
    try:
        main()
    except InstallError as err:
        log(f"ERROR: {err}")
        sys.exit(1)
