"""Data layer: data contracts, representation converters, dataset loaders.

Dataset loading, splitting, and HDF5 storage follow in Sprints 1 and 4; the
schema below is the contract all of them must satisfy.
"""

from atomistic_pinn.data.ani_al import dataset_from_ani_al
from atomistic_pinn.data.converters import structure_from_ase, structure_to_ase
from atomistic_pinn.data.schema import AtomicStructure, Dataset, DatasetMetadata

__all__ = [
    "AtomicStructure",
    "Dataset",
    "DatasetMetadata",
    "dataset_from_ani_al",
    "structure_from_ase",
    "structure_to_ase",
]
