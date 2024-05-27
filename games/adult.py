"""This module contains the script to pre-compute the values for the adult data set"""
import copy
import os
import random
import warnings
from typing import Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from approximators.base import powerset
from datasets_own import Adult


class AdultGame:
    def __init__(
        self,
        model: Optional[RandomForestClassifier] = None,
        explanation_id: int = None,
        set_zero: bool = True,
        seed: int = 42,
    ):
        random.seed(seed)
        dataset = Adult()
        self.x_data = dataset.x_data
        self.y_data = dataset.y_data
        self.feature_names = dataset.feature_names
        self.n = dataset.n_features
        self.seed = seed

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_data.values, self.y_data.values, test_size=0.33, random_state=seed
        )

        if model is None:
            self.model = RandomForestClassifier(random_state=seed)
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

        self.empty_value = 0.0
        if set_zero:
            self.empty_value = self.set_call(set())

        self.original_class = self.y_test[self.explanation_id]
        if self.original_class == "<=50K":
            self.original_class_index = 0
        else:
            self.original_class_index = 1

    def set_call(self, S: set) -> float:
        """Returns the output of the model with features not in S set to imputation values."""
        x = copy.copy(self.x)
        x[list(S)] = self.imputation_values[list(S)]
        output = float(self.model.predict_proba(x.reshape(1, -1))[0][self.original_class_index])
        return output - self.empty_value

    def train_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(random_state=self.seed)
        model.fit(self.x_train, self.y_train)

        score = model.score(self.x_test, self.y_test)
        print("Score of the model: ", score)
        return model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # precomp parameters ---------------------------------------------------------------------------

    N_PRECOMP = 100

    # setup storage and all ------------------------------------------------------------------------
    n = 14
    N = set(range(n))

    storage_folder = os.path.join("data", "adult", str(n))
    os.makedirs(storage_folder, exist_ok=True)

    for i in tqdm(range(N_PRECOMP)):
        game = AdultGame(seed=i, set_zero=False)
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
