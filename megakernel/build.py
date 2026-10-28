"""Build (or load a prebuilt) LeWM-MK torch extension."""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(build_dir: str = "/mk/build", verbose: bool = True, prof: bool = False):
    from torch.utils.cpp_extension import load

    import shutil

    from megakernel.runtime import source_hash

    tk = os.environ.get("THUNDERKITTENS_ROOT", "/tk")
    # Content-addressed build: Modal mounts do not preserve mtimes, so ninja cannot be trusted to notice edits.
    h = source_hash() + ("p" if prof else "")
    out = Path(build_dir) / h
    out.mkdir(parents=True, exist_ok=True)
    src = out / f"lewm_mk_{h}.cu"
    shutil.copyfile(HERE / "lewm_mk.cu", src)
    mod = load(
        name=f"lewm_mk_{h}",
        sources=[str(src)],
        build_directory=str(out),
        extra_include_paths=[f"{tk}/include", f"{tk}/prototype"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3", "-std=c++20", "--use_fast_math", "-DKITTENS_SM90", "-DNDEBUG",
            "-gencode", "arch=compute_90a,code=sm_90a",
            "--expt-extended-lambda", "--expt-relaxed-constexpr", "-lineinfo",
            "-Xptxas=--warn-on-spills", "-Xptxas=-v",
        ] + (["-DLEWM_MK_PROF"] if prof else []),
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )
    tag = "_prof" if prof else ""
    (Path(build_dir) / f"source_hash{tag}.txt").write_text(source_hash())
    (Path(build_dir) / f"so_path{tag}.txt").write_text(mod.__file__)
    return mod.__file__
