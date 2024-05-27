from .base_game import BaseGame


class SyntheticGame(BaseGame):

    def __init__(self, player_names):
        super(SyntheticGame, self).__init__(player_names=player_names)

    def __call__(self, S):
        raise NotImplementedError

    def save_config(self):
        raise NotImplementedError

    def load_config(self):
        raise NotImplementedError

    def precompute(self):
        raise NotImplementedError

    def save_game(self, filename, output_dir):
        raise NotImplementedError

    def load_game(self, path_to_file, filename):
        raise NotImplementedError
