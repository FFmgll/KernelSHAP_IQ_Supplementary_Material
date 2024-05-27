from abc import abstractmethod, ABCMeta
from typing import List, Union, Any

from .player import create_generic_players


class BaseGame(metaclass=ABCMeta):
    """
    An abstract base class for games. This class defines a basic structure and essential methods that
    must be implemented by any subclass.

    Attributes:
        players (List[Player]): A list of players participating in the game.
    """

    def __init__(self, player_names_or_indices: Union[List[str], List[int]], X: Any) -> None:
        """
        Initializes a BaseGame instance.

        Args:
            player_names_or_indices (Union[List[str], List[int]): A list of names given to players
        """
        self.players = create_generic_players(player_names_or_indices, X)

    @abstractmethod
    def __call__(self, S: List[List]):
        """
        Abstract method to be implemented by subclasses. Typically used to evaluate the game with a given
        set of player coalitions.

        Args:
            S (List[List]): A list of coalitions, where each coalition is a list of players.
        """
        raise NotImplementedError

    @abstractmethod
    def save_config(self):
        """
        Abstract method to save the game configuration. Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def load_config(self):
        """
        Abstract method to load a game configuration. Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def precompute(self):
        """
        Abstract method for precomputing game values. Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def save_game(self,
                  filename: str,
                  output_dir: str
                  ) -> None:
        """
        Abstract method to save the game state to a file.

        Args:
            filename (str): The name of the file to save the game state.
            output_dir (str): The directory where the file will be saved.
        """
        raise NotImplementedError

    @abstractmethod
    def load_game(self,
                  filename: str,
                  path_to_file: str
                  ) -> None:
        """
        Abstract method to load a game state from a file.

        Args:
            filename (str): The name of the file containing the game state.
            path_to_file (str): The path to the file.
        """
        raise NotImplementedError
