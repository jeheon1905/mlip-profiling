"""
Structure Builders for MLIP Profiling

Utilities for generating atomic structures for profiling tests.
Follows fairchem's common_structures.py methodology.

Usage:
    from structure_builders import (
        get_fcc_crystal_by_num_atoms,
        get_fcc_crystal_by_num_cells,
        get_water_box,
        build_molecule,
        load_structures_from_files,
    )
    
    # FCC crystal (periodic, for bulk benchmarks)
    atoms = get_fcc_crystal_by_num_cells(5, atom_type="Cu")  # 500 atoms
    
    # Water box (non-periodic, for molecular benchmarks)
    atoms = get_water_box(100, pbc=False)  # 300 atoms
    
    # Load from files
    structures = load_structures_from_files(["water.xyz", "bulk.cif"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms
from ase.build import molecule, bulk
from ase.io import read as ase_read
from ase.lattice.cubic import FaceCenteredCubic


# =============================================================================
# fairchem-compatible Structure Builders
# =============================================================================

def get_fcc_crystal_by_num_atoms(
    num_atoms: int,
    lattice_constant: float = 3.8,
    atom_type: str = "C",
) -> Atoms:
    """
    Generate FCC crystal with approximately num_atoms atoms.
    
    This follows fairchem's benchmarking methodology:
    - lattice_constant=3.8 generates ~50 edges/atom
    
    Args:
        num_atoms: Target number of atoms
        lattice_constant: Lattice constant (default: 3.8 for benchmarking)
        atom_type: Element symbol (default: "C")
    
    Returns:
        ASE Atoms object with pbc=True
    """
    atoms = bulk(atom_type, "fcc", a=lattice_constant)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    indices = np.random.choice(len(atoms), num_atoms, replace=False)
    sampled_atoms = atoms[indices]
    sampled_atoms.info = {"charge": 0, "spin": 0}
    return sampled_atoms


def get_fcc_crystal_by_num_cells(
    n_cells: int,
    atom_type: str = "Cu",
    lattice_constant: float = 3.61,
) -> Atoms:
    """
    Generate FCC crystal supercell with exact n_cells x n_cells x n_cells size.
    
    Args:
        n_cells: Number of unit cells in each direction
        atom_type: Element symbol (default: "Cu")
        lattice_constant: Lattice constant (default: 3.61 for Cu)
    
    Returns:
        ASE Atoms object with pbc=True (4 * n_cells^3 atoms)
    """
    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        symbol=atom_type,
        size=(n_cells, n_cells, n_cells),
        pbc=True,
        latticeconstant=lattice_constant,
    )
    atoms.info = {"charge": 0, "spin": 0}
    return atoms


def get_water_box(
    num_molecules: int = 20,
    box_size: float = 10.0,
    seed: int = 42,
    pbc: bool = True,
) -> Atoms:
    """
    Create a random box of water molecules.
    
    Args:
        num_molecules: Number of water molecules
        box_size: Cubic box size in Angstroms
        seed: Random seed for reproducibility
        pbc: Periodic boundary conditions (default: True for liquid water simulation)
    
    Returns:
        ASE Atoms object
    """
    rng = np.random.default_rng(seed)
    water = molecule("H2O")
    
    all_positions = []
    all_symbols = []
    
    for _ in range(num_molecules):
        # Random position and rotation for each water molecule
        offset = rng.random(3) * box_size
        positions = water.get_positions() + offset
        all_positions.extend(positions)
        all_symbols.extend(water.get_chemical_symbols())
    
    atoms = Atoms(
        symbols=all_symbols,
        positions=all_positions,
        cell=[box_size] * 3,
        pbc=pbc,
    )
    atoms.info = {"charge": 0, "spin": 0}
    return atoms


# =============================================================================
# Additional Structure Builders
# =============================================================================

def build_molecule(name: str) -> Atoms:
    """
    Build a molecule from ASE's database.
    
    Args:
        name: Molecule name (e.g., "H2O", "CH4", "C6H6", "C60")
    
    Returns:
        ASE Atoms object
    """
    atoms = molecule(name)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


# =============================================================================
# Structure Loaders
# =============================================================================

def load_structures_from_files(
    file_paths: list[str],
    default_charge: int = 0,
    default_spin: int = 0,
) -> list[tuple[str, Atoms]]:
    """
    Load structures from XYZ, CIF, or other ASE-supported files.
    
    Args:
        file_paths: List of paths to structure files (xyz, cif, etc.)
        default_charge: Default charge if not present in file
        default_spin: Default spin if not present in file
    
    Returns:
        List of (name, Atoms) tuples
        
    Raises:
        FileNotFoundError: If a file does not exist
        ValueError: If a file cannot be read
    """
    structures = []
    
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Structure file not found: {file_path}")
        
        try:
            # Read single structure or multiple structures
            atoms_or_list = ase_read(file_path, index=":")
            if not isinstance(atoms_or_list, list):
                atoms_or_list = [atoms_or_list]
        except Exception as e:
            raise ValueError(f"Failed to read structure file {file_path}: {e}")
        
        for idx, atoms in enumerate(atoms_or_list):
            # Set default charge/spin if not present
            if "charge" not in atoms.info:
                atoms.info["charge"] = default_charge
            if "spin" not in atoms.info:
                atoms.info["spin"] = default_spin
            
            natoms = len(atoms)
            base_name = path.stem
            if len(atoms_or_list) > 1:
                name = f"{base_name}_{idx}_{natoms}atoms"
            else:
                name = f"{base_name}_{natoms}atoms"
            structures.append((name, atoms))
    
    return structures


# =============================================================================
# Batch Utilities
# =============================================================================

def apply_batching(
    structures: list[tuple[str, Atoms]],
    batch_sizes: list[int],
    copy_fn: Callable | None = None,
) -> list[tuple[str, list[Atoms]]]:
    """
    Convert single-structure test cases to batched test cases.
    
    Args:
        structures: List of (name, Atoms) tuples
        batch_sizes: List of batch sizes to create
        copy_fn: Optional function to create copies of atoms.
                 If None, uses atoms.copy()
    
    Returns:
        List of (name, list[Atoms]) tuples for batched inference
    """
    batched_cases = []
    
    for name, atoms in structures:
        natoms = len(atoms)
        
        for bs in batch_sizes:
            # Create copies for batch
            if copy_fn:
                atoms_list = [copy_fn() for _ in range(bs)]
            else:
                atoms_list = [atoms.copy() for _ in range(bs)]
                # Ensure charge/spin are set on copies
                for a in atoms_list:
                    if "charge" not in a.info:
                        a.info["charge"] = atoms.info.get("charge", 0)
                    if "spin" not in a.info:
                        a.info["spin"] = atoms.info.get("spin", 0)
            
            total_atoms = natoms * bs
            batch_name = f"{name}_x{bs}_batch_{total_atoms}total"
            batched_cases.append((batch_name, atoms_list))
    
    return batched_cases


# =============================================================================
# Test Case Generators (for convenience)
# =============================================================================

def get_fcc_test_cases_by_num_atoms(
    atom_counts: list[int] | None = None,
    lattice_constant: float = 3.8,
    atom_type: str = "C",
) -> list[tuple[str, Atoms]]:
    """
    Get FCC crystal test cases with various atom counts (fairchem-style).
    
    Args:
        atom_counts: List of target atom counts
        lattice_constant: Lattice constant
        atom_type: Element symbol
    
    Returns:
        List of (name, Atoms) tuples
    """
    if atom_counts is None:
        atom_counts = [100, 500, 1000, 2000, 4000]
    
    return [
        (f"fcc_{atom_type}_{n}atoms", get_fcc_crystal_by_num_atoms(n, lattice_constant, atom_type))
        for n in atom_counts
    ]


def get_fcc_test_cases_by_num_cells(
    cell_counts: list[int] | None = None,
    atom_type: str = "Cu",
    lattice_constant: float = 3.61,
) -> list[tuple[str, Atoms]]:
    """
    Get FCC crystal test cases with various cell counts (fairchem-style).
    
    Args:
        cell_counts: List of n_cells values (atoms = 4 * n_cells^3)
        atom_type: Element symbol
        lattice_constant: Lattice constant
    
    Returns:
        List of (name, Atoms) tuples
    """
    if cell_counts is None:
        cell_counts = [2, 3, 4, 5, 6]  # 32, 108, 256, 500, 864 atoms
    
    test_cases = []
    for n in cell_counts:
        atoms = get_fcc_crystal_by_num_cells(n, atom_type, lattice_constant)
        natoms = len(atoms)
        name = f"{atom_type}_fcc_{n}x{n}x{n}_{natoms}atoms"
        test_cases.append((name, atoms))
    return test_cases


def get_water_box_test_cases(
    num_molecules_list: list[int] | None = None,
    box_size: float = 10.0,
    seed: int = 42,
    pbc: bool = True,
) -> list[tuple[str, Atoms]]:
    """
    Get water box test cases with various molecule counts.
    
    Args:
        num_molecules_list: List of molecule counts
        box_size: Box size in Angstroms
        seed: Random seed
        pbc: Periodic boundary conditions (default: True)
    
    Returns:
        List of (name, Atoms) tuples
    """
    if num_molecules_list is None:
        num_molecules_list = [10, 20, 50, 100]
    
    pbc_str = "pbc" if pbc else "nopbc"
    return [
        (f"water_{n}mol_{n*3}atoms_{pbc_str}", get_water_box(n, box_size, seed, pbc))
        for n in num_molecules_list
    ]


def get_molecule_test_cases(
    molecules: list[str] | None = None,
) -> list[tuple[str, Atoms]]:
    """
    Get molecule test cases from ASE's database.
    
    Args:
        molecules: List of molecule names (e.g., ['H2O', 'CH4', 'C6H6'])
                   If None, uses default set.
    
    Returns:
        List of (name, Atoms) tuples
    """
    if molecules is None:
        molecules = ["H2O", "CH4", "C6H6", "C60"]
    return [(name, build_molecule(name)) for name in molecules]


# =============================================================================
# CLI for standalone structure generation
# =============================================================================

def main():
    """Command-line interface for generating structures."""
    import argparse
    from ase.io import write as ase_write
    
    parser = argparse.ArgumentParser(
        description="Generate atomic structures for MLIP profiling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # FCC crystals by atom count (fairchem benchmarking style, periodic)
    python structure_builders.py --fcc-by-atoms --fcc-atom-counts 100 500 1000 2000 --output-dir structures/
    
    # FCC crystals by cell count (exact supercells, periodic)
    python structure_builders.py --fcc-by-cells --fcc-cell-counts 2 3 4 5 --output-dir structures/
    
    # Water boxes (periodic by default, like fairchem)
    python structure_builders.py --water-box --water-molecules 10 20 50 100 --output-dir structures/
    
    # Water boxes (non-periodic, for molecular benchmarks)
    python structure_builders.py --water-box --water-molecules 10 20 --water-no-pbc --output-dir structures/
    
    # Molecules from ASE database (non-periodic)
    python structure_builders.py --molecules H2O CH4 C6H6 --output-dir structures/
        """
    )
    
    parser.add_argument("--output-dir", type=Path, default=Path("./structures"),
                        help="Output directory for structure files")
    parser.add_argument("--format", type=str, default="xyz",
                        help="Output file format (default: xyz)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    # fairchem-style FCC crystals (periodic)
    parser.add_argument("--fcc-by-atoms", action="store_true",
                        help="Generate FCC crystals by target atom count (fairchem-style, periodic)")
    parser.add_argument("--fcc-atom-counts", type=int, nargs="+", default=None,
                        help="Target atom counts for --fcc-by-atoms")
    parser.add_argument("--fcc-atom-type", type=str, default="C",
                        help="Element for --fcc-by-atoms (default: C)")
    parser.add_argument("--fcc-lattice-constant", type=float, default=3.8,
                        help="Lattice constant for --fcc-by-atoms (default: 3.8)")
    
    parser.add_argument("--fcc-by-cells", action="store_true",
                        help="Generate FCC crystals by cell count (exact supercells, periodic)")
    parser.add_argument("--fcc-cell-counts", type=int, nargs="+", default=None,
                        help="Cell counts for --fcc-by-cells (atoms = 4*n^3)")
    parser.add_argument("--fcc-cell-element", type=str, default="Cu",
                        help="Element for --fcc-by-cells (default: Cu)")
    
    # Water boxes (periodic by default, like fairchem)
    parser.add_argument("--water-box", action="store_true",
                        help="Generate water boxes (periodic by default)")
    parser.add_argument("--water-molecules", type=int, nargs="+", default=None,
                        help="Number of water molecules")
    parser.add_argument("--water-box-size", type=float, default=10.0,
                        help="Water box size in Angstroms (default: 10.0)")
    parser.add_argument("--water-no-pbc", action="store_true",
                        help="Disable periodic boundary conditions for water boxes")
    
    # Molecules (non-periodic)
    parser.add_argument("--molecules", type=str, nargs="+", default=None,
                        help="Molecule names from ASE database (non-periodic)")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    structures = []
    
    # fairchem-style: FCC by atom count (periodic)
    if args.fcc_by_atoms:
        structures.extend(get_fcc_test_cases_by_num_atoms(
            args.fcc_atom_counts,
            args.fcc_lattice_constant,
            args.fcc_atom_type,
        ))
    
    # fairchem-style: FCC by cell count (periodic)
    if args.fcc_by_cells:
        structures.extend(get_fcc_test_cases_by_num_cells(
            args.fcc_cell_counts,
            args.fcc_cell_element,
        ))
    
    # Water boxes (periodic by default)
    if args.water_box:
        structures.extend(get_water_box_test_cases(
            args.water_molecules,
            args.water_box_size,
            args.seed,
            pbc=not args.water_no_pbc,
        ))
    
    # Molecules (non-periodic)
    if args.molecules:
        structures.extend(get_molecule_test_cases(args.molecules))
    
    if not structures:
        print("No structures specified. Use --help for options.")
        return
    
    for name, atoms in structures:
        output_path = args.output_dir / f"{name}.{args.format}"
        ase_write(output_path, atoms)
        print(f"Saved: {output_path} ({len(atoms)} atoms, pbc={atoms.pbc.any()})")
    
    print(f"\nGenerated {len(structures)} structure(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
