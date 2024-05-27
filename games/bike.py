"""This module contains the script to pre-compute the values for the bike data set"""
import copy
import os
import random
import warnings
from typing import Union, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm import tqdm

from approximators.base import powerset
from datasets_own import BikeSharing
from xgboost import XGBRegressor


class BikeGame:
    def __init__(
        self,
        model: Optional[XGBRegressor] = None,
        explanation_id: int = None,
        set_zero: bool = True,
        renormalize: bool = False,
        target_min_max=True,
        seed: int = 42,
    ):
        random.seed(seed)
        dataset = BikeSharing(normalize=True)
        self.x_data = dataset.x_data
        self.y_data = dataset.y_data
        self.feature_names = dataset.feature_names
        self.n = dataset.n_features
        self.seed = seed

        self.target_min_max = target_min_max
        if not self.target_min_max:
            self.y_data = self.y_data.apply(np.log10)
        else:
            # minmax normalize y_data
            self.y_data = (self.y_data - self.y_data.min()) / (
                self.y_data.max() - self.y_data.min()
            )

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_data.values, self.y_data.values, test_size=0.33, random_state=seed
        )

        if model is None:
            self.model = XGBRegressor(random_state=seed)
            self.model.fit(self.x_train, self.y_train)
        else:
            self.model = model

        self.explanation_id = explanation_id
        if explanation_id is None:
            self.explanation_id = random.randint(0, len(self.x_test) - 1)
        self.x = self.x_test[self.explanation_id]

        self.set_zero = set_zero

        imputation_values = []
        for feature in self.x_data.columns:
            if feature in dataset.cat_feature_names:
                imputation_values.append(self.x_data[feature].mode().values[0])
            else:
                imputation_values.append(self.x_data[feature].mean())
        self.imputation_values = np.asarray(imputation_values)

        self.renormalize = renormalize
        if self.target_min_max:
            self.renormalize = False
        self.renormalize_call = self.label_reverse_empty
        if self.renormalize:
            self.renormalize_call = self.label_reverse_transform

        self.empty_value = 0.0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S: set) -> float:
        """Returns the output of the model with features not in S set to imputation values."""
        x = copy.copy(self.x)
        x[list(S)] = self.imputation_values[list(S)]
        output = self.renormalize_call(float(self.model.predict(x.reshape(1, -1)).item()))
        return output - self.empty_value

    @staticmethod
    def label_reverse_empty(values):
        return values

    @staticmethod
    def label_reverse_transform(values):
        return 10**values

    def train_model(self) -> XGBRegressor:
        model = XGBRegressor(random_state=self.seed)
        model.fit(self.x_train, self.y_train)

        score = model.score(self.x_test, self.y_test)
        print("Score of the model: ", score)
        return model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # precomp parameters ---------------------------------------------------------------------------

    N_PRECOMP = 98

    # setup storage and all ------------------------------------------------------------------------
    n = 12
    N = set(range(n))

    storage_folder = os.path.join("data", "bike", str(n))
    os.makedirs(storage_folder, exist_ok=True)

    for i in tqdm(range(N_PRECOMP)):
        game = BikeGame(seed=i, set_zero=False)
        explanation_index = int(game.explanation_id)
        storage_run = []
        for S in powerset(N):
            S_id = "s_" + "_".join([str(player) for player in sorted(S)])
            value = game.set_call(S)
            storage_run.append({"set": S_id, "value": value})

        df = pd.DataFrame(storage_run)
        file_name = str(explanation_index) + ".csv"
        file_path = os.path.join(storage_folder, file_name)
        df.to_csv(file_path, index=False)
