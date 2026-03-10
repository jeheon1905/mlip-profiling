"""
MACE Profiling Script

PyTorch Profiler를 사용하여 MACE 모델의 실행 시간을 측정합니다.
MACE 내부 연산(Interaction, ProductBasis, SymmetricContraction 등)별 시간 분석 지원.
"""

import argparse
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from pathlib import Path
from ase.io import read
from mace.calculators import MACECalculator
from mace.modules.profiling import set_profiling_enabled


def load_structures(structure_path: str):
    """구조 파일 로드 (xyz, cif 등 ASE 지원 형식)"""
    structures = read(structure_path, index=":")
    if not isinstance(structures, list):
        structures = [structures]
    return structures


def profile_mace(
    model_path: str,
    structure_path: str,
    device: str = "cuda",
    output_dir: str = "profile_traces_mace",
    warmup_steps: int = 5,
    profile_steps: int = 10,
):
    """MACE 모델 프로파일링 수행"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # MACE Calculator 초기화
    print(f"Loading MACE model from: {model_path}")
    calculator = MACECalculator(
        model_paths=model_path,
        device=device,
        default_dtype="float64",
    )
    
    # 구조 로드
    print(f"Loading structures from: {structure_path}")
    structures = load_structures(structure_path)
    print(f"Loaded {len(structures)} structure(s)")
    
    for idx, atoms in enumerate(structures):
        atoms.calc = calculator
        n_atoms = len(atoms)
        print(f"\nProfiling structure {idx + 1}/{len(structures)} ({n_atoms} atoms)")
        
        # Warmup
        print(f"  Warmup ({warmup_steps} steps)...")
        for _ in range(warmup_steps):
            _ = atoms.get_potential_energy()
            _ = atoms.get_forces()
        
        # Profiling
        print(f"  Profiling ({profile_steps} steps)...")
        set_profiling_enabled(True)  # MACE 내부 프로파일링 훅 활성화
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for step in range(profile_steps):
                with record_function("forward_energy"):
                    energy = atoms.get_potential_energy()
                with record_function("forward_forces"):
                    forces = atoms.get_forces()
        set_profiling_enabled(False)  # 프로파일링 비활성화
        
        # 결과 저장
        trace_file = output_path / f"mace_trace_struct{idx}_atoms{n_atoms}.json"
        prof.export_chrome_trace(str(trace_file))
        print(f"  Trace saved: {trace_file}")
        
        # 요약 출력
        print("\n  === Profiling Summary ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))


def main():
    parser = argparse.ArgumentParser(description="MACE Profiling with PyTorch Profiler")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to MACE model file (.model or .pt)",
    )
    parser.add_argument(
        "--structure-path",
        type=str,
        required=True,
        help="Path to structure file (xyz, cif, etc.)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profile_traces_mace",
        help="Output directory for trace files",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps (default: 5)",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=10,
        help="Number of profiling steps (default: 10)",
    )
    
    args = parser.parse_args()
    
    profile_mace(
        model_path=args.model_path,
        structure_path=args.structure_path,
        device=args.device,
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        profile_steps=args.profile_steps,
    )


if __name__ == "__main__":
    main()
