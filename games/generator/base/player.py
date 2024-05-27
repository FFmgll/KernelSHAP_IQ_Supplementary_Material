from typing import List, Optional, Union, Tuple, Any
import pandas as pd


class Player:
    """
    Represents a player in a game.

    Attributes:
        number (int): A unique identifier for the player.
        name (str): The name of the player. Defaults to "Player(number)" if not
        provided.
    """

    def __init__(self, number: int, column_idx: Optional[int] = None, name: Optional[str] = None) -> None:
        """
        Initializes a Player instance.

        Args:
            number (int): A unique identifier for the player.
            column_idx (Optional[int]): The index of the column in the dataset that corresponds to this player.
            name (str, optional): The name of the player. Defaults to None, in
            which case the name is set to "Player(number)".
        """
        self.number: int = number
        self.name: str
        if name is None:
            self.name = f"Player {self.number}"
        else:
            self.name = name
        self.column_idx = column_idx

    def __str__(self) -> str:
        """
        Return the name of a player.

        Returns:
            str: The name of the player.
        """
        return self.name

    def __repr__(self) -> str:
        """
        Return a string representation that can be used to recreate a player.
        Only works for players of type Player, since derived classes may
        have different constructors.

        Returns:
            str: The string representation of the player
        """
        if type(self) is Player:
            result = f"Player(number={self.number}, name='{self.name}, column_idx={self.column_idx})"
        else:
            result = object.__repr__(self)
        return result

    def __lt__(self, other):
        return self.number < other.number


def create_generic_players(player_names_or_indices: Union[List[int], List[str]], X: Any = None):
    """
    Return a list of newly created players without any attributes
    other than player number and name.

    Args:
        player_names_or_indices (Union[List[int], List[str]])): Player names (str) or
        list of unique indices that are used to create the players.

    Returns:
        List[Player]: A list of generic players.
    """

    # in case we are given a list of names:
    if isinstance(player_names_or_indices, list) and all(
        isinstance(item, str) for item in player_names_or_indices
    ):
        if X is not None:
            # and we have X as a dataframe:

            if isinstance(X, pd.Series):
                X = X.to_frame().transpose()

            if isinstance(X, pd.DataFrame):
                # if for all given names there are corresponding columsn in X:
                if all(name in X.columns for name in player_names_or_indices):
                    column_indices = [X.columns.get_loc(name) for name in player_names_or_indices]
                    result = [
                        Player(i, c_idx, name)
                        for i, name, c_idx
                        in zip(range(len(player_names_or_indices)), player_names_or_indices, column_indices)
                        ]
                else:
                    raise ValueError("Not all column names are present in the Dataset.")
            # and we have X in some other format that may not have named columns:
            else:
                raise ValueError("Can only instantiate with strings when the dataset is a dataframe.")
        # if X is none, create named generic players, e.g. for PrecomputedGame
        else:
            result = [Player(number, name=name) for number, name in enumerate(player_names_or_indices)]
        
    # in case we are given a list of ints:
    elif (
        isinstance(player_names_or_indices, list)
        and all(isinstance(item, int) for item in player_names_or_indices)
        and
        # to make sure there are no duplicate indices:
        len(set(player_names_or_indices)) == len(player_names_or_indices)
    ):
        result = [Player(number, column_idx=column_idx) for number, column_idx in enumerate(player_names_or_indices)]
    else:
        raise ValueError(
            f"""Player names or indices must be either a list of strings or list of unique ints.
            Given: {player_names_or_indices}"""
        )
    return result


def names_or_indices_to_players(
    players: List[Player],
    player_names_or_column_idcs: Union[List[str], List[int], List[Player]]
) -> List[Player]:
    """
    Return the players that have the given names or indices. If a name/index is not found, raise a ValueError, if there
    are multiple players with the same name, the first one according to player.number is included in the result.
    If the objects are already of type player, the same player list is returned.

    Args:
        player_names_or_column_idcs (Union[List[str], List[int], List[Player]]): Names or column names of returned
        players
    """
    # in case we only have players
    all_players = all(isinstance(elem, Player) for elem in player_names_or_column_idcs)
    if all_players:
        return player_names_or_column_idcs

    all_names = all(isinstance(elem, str) for elem in player_names_or_column_idcs)
    all_indices = all(isinstance(elem, int) for elem in player_names_or_column_idcs)

    # in case the given list does not only contain strings or ints:
    if not all_names and not all_indices:
        raise ValueError("Players have to be queried with a list of only ints or list of only names (strings).")

    # convert from strings or ints to players:
    result = []
    if all_names:
        for name in player_names_or_column_idcs:
            matched_player = next(
                (player for player in players if player.name == name), None
            )
            if matched_player is None:
                raise ValueError(f"No player found with name '{name}'")
            result.append(matched_player)
    else:
        for idx in player_names_or_column_idcs:
            matched_player = next(
                (player for player in players if player.column_idx == idx), None
            )
            if matched_player is None:
                raise ValueError(f"No player found with index '{idx}'")
            result.append(matched_player)
    return result


def players_to_str(players: Tuple[Player]) -> str:
    """
    Convert a tuple of players to a string of the form e.g. "1|4|2|9" where each number corresponds to a player number.
    These strings are used in the non-memory-efficient csv files to save the value function.

    Args:
        players (Tuple[Player]): a tuple of players to convert to a string

    Returns:

    """
    numbers = [str(player.number) for player in players]
    return "|".join(numbers) if len(numbers) != 0 else "EMPTY_SET"


def str_to_players(players_str: str, all_players: List[Player] = None) -> Tuple[Player]:
    """
    Take a string (e.g. "1|4|2|9") that corresponds to a coalition of players and return those players.
    These strings are used in the non-memory-efficient csv files to save the value function.

    Args:
        players_str (str): a string of the form e.g. "1|4|2|9" where each number is a player number.
        all_players (List[Player]): the player string is mapped to a subset of these players with the corresponding
        player numbers.

    Returns:
    Tuple[Player]: A tuple of players with the given player numbers. Tuples are used insead of lists in order to make
    the result hashable.
    """
    player_numbers = players_str.split("|")
    return tuple(
        player for player in all_players if str(player.number) in player_numbers
    )
