"""
Defines several constants used throughout the project.
Constants are grouped using NamedTuple objects for convenient attribute-based access.


Examples
--------
>>> from mpp import constants
>>> constants.PATHS.ROOT  # Path to the project root
>>> constants.VOCAB["bohren"]  # Get token ID for a process step
"""
# standard imports
from pathlib import Path
from collections import namedtuple


# Determine ROOT (two levels up from this file: src/mpp/ -> src/ -> ROOT)
_ROOT = Path(__file__).parents[2]

# Dictionary of relevant project paths
_path_dict = {
    "ROOT":   _ROOT,
    "CONFIG": _ROOT / "config",
}

# -------------------------------
# Define token vocabulary
# -------------------------------

# Vocabulary for manufacturing process steps and special tokens
VOCAB = {
    "fräsen": 0,
    "schleifen": 1,
    "bohren": 2,
    "schweißen": 3,
    "drehen": 4,
    "prüfen": 5,
    "kontrollieren": 6,
    "START": 7,
    "STOP": 8,
    "PAD": 9,
}

# check if all keys and values are unique
assert len(VOCAB) == len(set(VOCAB.values())), ValueError("VOCAB values must be unique")
assert len(VOCAB) == len(set(VOCAB.keys())), ValueError("VOCAB keys must be unique")

# Inverted vocabulary: maps token IDs back to string labels
INV_VOCAB = {v: k for k, v in VOCAB.items()}

# Convert path dictionary to a namedtuple for attribute-style access
Paths = namedtuple("Paths", list(_path_dict.keys()))
PATHS = Paths(**_path_dict)

# clean up
del _path_dict
del Paths
del _ROOT
del namedtuple
del Path
