"""Data layer: data contracts and representation converters (Sprint 0).

Dataset loading, splitting, and HDF5 storage follow in Sprints 1 and 4; the
schema below is the contract all of them must satisfy.
"""

from atomistic_pinn.data.converters import structure_from_ase, structure_to_ase
from atomistic_pinn.data.schema import AtomicStructure, Dataset, DatasetMetadata

__all__ = [
    "AtomicStructure",
    "Dataset",
    "DatasetMetadata",
    "structure_from_ase",
    "structure_to_ase",
]
