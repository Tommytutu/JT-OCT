"""Prepare CuPy's optional CUDA runtime for a workspace with a Unicode path."""
import os
import shutil
import sys
from pathlib import Path


def copy_tree_contents(source, destination):
    if not source.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def main():
    packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia = packages / "nvidia"
    target = Path(os.environ.get("JT_OCT_CUDA_ROOT",
                                 Path.home() / ".cache" / "jt_oct" / "cuda12_runtime"))
    (target / "bin").mkdir(parents=True, exist_ok=True)
    (target / "include").mkdir(parents=True, exist_ok=True)
    (target / "lib" / "x64").mkdir(parents=True, exist_ok=True)
    copy_tree_contents(nvidia / "cuda_nvrtc" / "bin", target / "bin")
    copy_tree_contents(nvidia / "cuda_runtime" / "bin", target / "bin")
    copy_tree_contents(nvidia / "cuda_runtime" / "include", target / "include")
    copy_tree_contents(nvidia / "cuda_runtime" / "lib" / "x64", target / "lib" / "x64")
    copy_tree_contents(packages / "cupy" / "_core" / "include" / "cupy",
                       target / "include" / "cupy")
    print(target)


if __name__ == "__main__":
    main()

