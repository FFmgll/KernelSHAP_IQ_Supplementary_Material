from .all import (
    NLPLookupGame,
    ParameterizedSparseLinearModel,
    NLPGame,
    MachineLearningMetaGame,
    MachineLearningGame,
    ConvergenceGame,
    LookUpGame,
)
from .generator_game import GeneratorGame

try:
    from .vit_game import ViTGame
except ImportError:
    ViTGame = NotImplementedError("ViTGame requires the pytorch library to be installed.")

__all__ = [
    "LookUpGame",
    "NLPLookupGame",
    "ParameterizedSparseLinearModel",
    "NLPGame",
    "ViTGame",
    "GeneratorGame",
    "MachineLearningMetaGame",
    "MachineLearningGame",
    "ConvergenceGame",
]
