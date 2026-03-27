"""
Tests for structure_builders.py
"""
import pytest
from pathlib import Path


class TestLoadStructuresFromFiles:
    """Tests for load_structures_from_files()"""
    
    def test_load_xyz_file(self, structures_dir):
        """Test loading .xyz file."""
        from structure_builders import load_structures_from_files
        
        xyz_files = list(structures_dir.glob("*.xyz"))
        if not xyz_files:
            pytest.skip("No xyz files in structures/")
        
        structures = load_structures_from_files([str(xyz_files[0])])
        
        assert len(structures) == 1
        name, atoms = structures[0]
        assert atoms is not None
        assert len(atoms) > 0
    
    def test_load_multiple_files(self, structures_dir):
        """Test loading multiple files."""
        from structure_builders import load_structures_from_files
        
        xyz_files = list(structures_dir.glob("*.xyz"))[:3]
        if len(xyz_files) < 2:
            pytest.skip("Need at least 2 xyz files")
        
        structures = load_structures_from_files([str(f) for f in xyz_files])
        
        assert len(structures) == len(xyz_files)
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        from structure_builders import load_structures_from_files
        
        with pytest.raises(FileNotFoundError):
            load_structures_from_files(["nonexistent_file.xyz"])
    
    def test_default_charge_spin(self, tmp_path):
        """Test that default charge and spin are set for files without them."""
        from structure_builders import load_structures_from_files
        
        # Create a minimal xyz file without charge/spin info
        xyz_content = """2
Simple H2 molecule
H 0.0 0.0 0.0
H 0.74 0.0 0.0
"""
        xyz_file = tmp_path / "h2.xyz"
        xyz_file.write_text(xyz_content)
        
        structures = load_structures_from_files(
            [str(xyz_file)],
            default_charge=1,
            default_spin=2,
        )
        
        name, atoms = structures[0]
        assert atoms.info.get("charge") == 1
        assert atoms.info.get("spin") == 2
    
    def test_empty_file_list(self):
        """Test handling of empty file list."""
        from structure_builders import load_structures_from_files
        
        structures = load_structures_from_files([])
        assert structures == []


class TestGetFccCrystal:
    """Tests for get_fcc_crystal_by_num_cells()"""
    
    def test_create_fcc_crystal(self):
        """Test FCC crystal generation."""
        from structure_builders import get_fcc_crystal_by_num_cells
        
        atoms = get_fcc_crystal_by_num_cells(n_cells=2, atom_type="Cu")
        
        # 2x2x2 FCC has 4 atoms per unit cell * 8 = 32 atoms
        assert len(atoms) == 32
        assert atoms.pbc.all()
    
    def test_different_sizes(self):
        """Test various cell sizes."""
        from structure_builders import get_fcc_crystal_by_num_cells
        
        sizes = [1, 2, 3, 4]
        for n in sizes:
            atoms = get_fcc_crystal_by_num_cells(n_cells=n, atom_type="Cu")
            expected = 4 * (n ** 3)  # FCC has 4 atoms per unit cell
            assert len(atoms) == expected, f"Failed for n={n}"
    
    def test_charge_spin_set(self):
        """Test that charge and spin info are set."""
        from structure_builders import get_fcc_crystal_by_num_cells
        
        atoms = get_fcc_crystal_by_num_cells(n_cells=2, atom_type="Cu")
        
        assert "charge" in atoms.info
        assert "spin" in atoms.info
    
    def test_different_elements(self):
        """Test with different elements."""
        from structure_builders import get_fcc_crystal_by_num_cells
        
        for element in ["Cu", "Au", "Ag", "Al"]:
            atoms = get_fcc_crystal_by_num_cells(n_cells=2, atom_type=element)
            assert all(s == element for s in atoms.get_chemical_symbols())


class TestGetFccCrystalByNumAtoms:
    """Tests for get_fcc_crystal_by_num_atoms()"""
    
    def test_approximate_atom_count(self):
        """Test that atom count is approximately correct."""
        from structure_builders import get_fcc_crystal_by_num_atoms
        
        target = 100
        atoms = get_fcc_crystal_by_num_atoms(num_atoms=target)
        
        # Should be exactly target (sampled down)
        assert len(atoms) == target
    
    def test_has_pbc(self):
        """Test that PBC is set."""
        from structure_builders import get_fcc_crystal_by_num_atoms
        
        atoms = get_fcc_crystal_by_num_atoms(num_atoms=50)
        assert atoms.pbc.any()


class TestBuildMolecule:
    """Tests for build_molecule()"""
    
    def test_build_water(self):
        """Test building water molecule."""
        from structure_builders import build_molecule
        
        atoms = build_molecule("H2O")
        
        assert len(atoms) == 3
        symbols = atoms.get_chemical_symbols()
        assert symbols.count("H") == 2
        assert symbols.count("O") == 1
    
    def test_build_methane(self):
        """Test building methane molecule."""
        from structure_builders import build_molecule
        
        atoms = build_molecule("CH4")
        
        assert len(atoms) == 5
    
    def test_charge_spin_set(self):
        """Test that charge and spin info are set."""
        from structure_builders import build_molecule
        
        atoms = build_molecule("H2O")
        
        assert "charge" in atoms.info
        assert "spin" in atoms.info


class TestGetWaterBox:
    """Tests for get_water_box()"""
    
    def test_water_box_atom_count(self):
        """Test water box has correct atom count."""
        from structure_builders import get_water_box
        
        num_molecules = 10
        atoms = get_water_box(num_molecules=num_molecules)
        
        # Each water has 3 atoms
        assert len(atoms) == num_molecules * 3
    
    def test_water_box_composition(self):
        """Test water box has correct H:O ratio."""
        from structure_builders import get_water_box
        
        atoms = get_water_box(num_molecules=5)
        symbols = atoms.get_chemical_symbols()
        
        h_count = symbols.count("H")
        o_count = symbols.count("O")
        
        assert h_count == 2 * o_count  # H2O ratio
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same structure."""
        from structure_builders import get_water_box
        
        atoms1 = get_water_box(num_molecules=5, seed=42)
        atoms2 = get_water_box(num_molecules=5, seed=42)
        
        import numpy as np
        assert np.allclose(atoms1.positions, atoms2.positions)
    
    def test_different_seeds_different_positions(self):
        """Test that different seeds produce different structures."""
        from structure_builders import get_water_box
        
        atoms1 = get_water_box(num_molecules=5, seed=42)
        atoms2 = get_water_box(num_molecules=5, seed=123)
        
        import numpy as np
        assert not np.allclose(atoms1.positions, atoms2.positions)
    
    def test_pbc_option(self):
        """Test PBC option."""
        from structure_builders import get_water_box
        
        atoms_pbc = get_water_box(num_molecules=5, pbc=True)
        atoms_no_pbc = get_water_box(num_molecules=5, pbc=False)
        
        assert atoms_pbc.pbc.all()
        assert not atoms_no_pbc.pbc.any()
