from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from .generator import SklearnLocalMLGame


class GeneratorGame:
    """Wrapper Class to use the generator as a basis for game."""

    def __init__(self, n: int):
        self.game_name = "generator_game"
        self.n = n

        X, y = make_regression(n_samples=1_000, n_features=self.n)

        self.model = DecisionTreeRegressor(random_state=42)
        self.model.fit(X, y)

        player_names = [i for i in range(X.shape[1])]

        self.game = SklearnLocalMLGame(
            X=X[0],
            y=y[0],
            model=self.model,
            player_names_or_indices=player_names,
            imputation_strategy="impute_central_measures",
            task_type="REG",
            impute_params={"background_X": X},
            memory_efficient=False,
        )
        self.empty_value = 0
        self.empty_value = self.set_call(set())

    def set_call(self, S: set):
        return float(self.game([list(S)])) - self.empty_value


if __name__ == "__main__":
    # test the class
    game = GeneratorGame(100)
    print(game.set_call({0}))
    print(game.set_call({0, 1}))
