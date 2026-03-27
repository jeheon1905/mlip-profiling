"""
Pytest configuration and shared fixtures.
"""
import sys
from pathlib import Path

import pytest
from ase.build import bulk

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def structures_dir(project_root):
    """Return structures directory."""
    return project_root / "structures"


@pytest.fixture
def sample_atoms():
    """Create a simple Cu FCC structure for testing."""
    return bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)


@pytest.fixture
def sample_atoms_with_pbc():
    """Create atoms with periodic boundary conditions."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True) * (3, 3, 3)
    atoms.pbc = True
    return atoms
