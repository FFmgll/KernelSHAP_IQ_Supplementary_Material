"""This module contains the game class for the California Housing dataset. If this module is called as a script it will pre-compute the values."""
import copy
import os
import random
from typing import Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import warnings

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics.functional import mean_absolute_error, r2_score

from approximators.base import powerset


class SmallNeuralNetwork(nn.Module):
    def __init__(self, n_input, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 5),
            nn.Linear(5, n_classes),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CaliforniaGame:
    def __init__(
        self,
        model: Union[str, "SmallNeuralNetwork"] = None,
        explanation_id: int = None,
        set_zero: bool = True,
        renormalize: bool = False,
        target_min_max=False,
        seed: int = 42,
        normalize: bool = True,
    ):
        # set random states
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.target_min_max = target_min_max

        if not self.target_min_max:
            self.default_model_path = os.path.join(
                "games", "models", "california_0.812511_0.076331"
            )
        else:
            self.default_model_path = os.path.join(
                "games", "models", "california_0.785998_0.0756_minmax"
            )
        self.model_save_dir = os.path.join("models")

        # load data
        data = fetch_california_housing()
        self.feature_names = data["feature_names"]

        x_data, y_data = data.data, data.target
        n_features = x_data.shape[-1]
        self.n = n_features

        self.scaler = StandardScaler()
        x_data = self.scaler.fit_transform(x_data)
        if normalize:
            if not target_min_max:
                y_data = np.log10(y_data)
            else:
                y_data = (y_data - y_data.min()) / (y_data.max() - y_data.min())
        else:
            renormalize = False

        self.imputation_values = np.mean(x_data, axis=0)
        self.imputation_values = torch.tensor(self.imputation_values, dtype=torch.float32)

        self.x_data = torch.tensor(x_data, dtype=torch.float32)
        self.y_data = torch.tensor(y_data, dtype=torch.float32).reshape(-1, 1)

        # train-test split for model evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.x_data, self.y_data, train_size=0.7, shuffle=True, random_state=seed
        )

        # Convert to 2D PyTorch tensors
        self.x_train = torch.tensor(X_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        self.x_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

        self.model = model
        if self.model is None:  # set default path and go into next if
            self.model = self.default_model_path

        # load model
        if type(self.model) is str:
            load_path = self.model
            self.model = SmallNeuralNetwork(n_input=self.n, n_classes=1)
            try:
                self.model.load_state_dict(torch.load(load_path))
            except FileNotFoundError:
                print(f"Model not found at {load_path}.")
                print("Training new model.")
                self.model = self.train_model()
        elif type(self.model) is SmallNeuralNetwork:
            pass
        else:
            self.model = self.train_model()
        self.model.eval()

        self.explanation_id = explanation_id
        if self.explanation_id is None:
            self.explanation_id = random.randint(a=0, b=len(self.x_test))
        self.x = self.x_test[self.explanation_id]

        self.original_output = float(self.model(self.x).item())

        self.renormalize = renormalize
        if self.target_min_max:
            self.renormalize = False
        self.renormalize_call = self.label_reverse_empty
        if renormalize:
            self.renormalize_call = self.label_reverse_transform

        self.empty_value = 0.0
        if set_zero:
            self.empty_value = self.set_call(set())

    def set_call(self, S: set) -> float:
        """Returns the output of the model with features not in S set to imputation values."""
        x = self.x.clone()
        x[list(S)] = self.imputation_values[list(S)]
        output = self.renormalize_call(float(self.model(x).item()))
        return output - self.empty_value

    @staticmethod
    def label_reverse_empty(values):
        return values

    @staticmethod
    def label_reverse_transform(values):
        return 10**values

    def train_model(self) -> SmallNeuralNetwork:
        """Trains the model and returns it."""
        model = SmallNeuralNetwork(n_input=self.n, n_classes=1)
        network_loss_function = nn.MSELoss()
        network_validation_loss = mean_absolute_error
        network_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Training the Model -----------------------------------------------------------------------
        epochs = 1000
        batch_size = 100

        dataset_train = TensorDataset(self.x_train, self.y_train)
        train_loader = DataLoader(dataset_train, batch_size=batch_size)

        best_r2 = -1
        mae_of_best = np.inf
        best_weights = None
        for epoch in range(epochs):
            train_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                # zero the parameter gradients
                network_optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = network_loss_function(outputs, labels)
                loss.backward()
                network_optimizer.step()
                train_loss += loss.item()

            model.eval()
            predictions = model(self.x_test)
            mae_test = mean_absolute_error(predictions, self.y_test)
            r2_test = r2_score(predictions, self.y_test)

            print(
                f"[{epoch + 1}] "
                f"train-loss: {train_loss:.6f} "
                f"test-mae: {mae_test:.6f}, "
                f"test-r2: {r2_test:.6f}"
            )

            if r2_test > best_r2:
                best_r2 = r2_test
                mae_of_best = mae_test
                best_weights = copy.deepcopy(model.state_dict())

        save_name = "_".join(
            ("california", str(round(float(best_r2), 6)), str(round(float(mae_of_best), 6)))
        )
        os.makedirs(self.model_save_dir, exist_ok=True)
        save_path = os.path.join(self.model_save_dir, save_name)

        if self.target_min_max:
            save_path += "_minmax"

        if not os.path.exists(save_path):
            print(f"Saving best Model with a R2 of {best_r2:.6f} MAE of {mae_of_best:.6f}")
            torch.save(best_weights, save_path)
        else:
            print(f"Model with a R2 of {best_r2:.6f} MAE of {mae_of_best:.6f} already exists.")

        return model


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # precomp parameters ---------------------------------------------------------------------------
    TRAIN = False
    N_PRECOMP = 100
    TARGET_MIN_MAX = True

    if TRAIN:
        game = CaliforniaGame(seed=42, set_zero=False, target_min_max=TARGET_MIN_MAX)
        model = game.train_model()
    else:
        # setup storage and all --------------------------------------------------------------------
        n = 8
        N = set(range(n))

        if TARGET_MIN_MAX:
            storage_folder = os.path.join("data", "california_minmax", str(n))
        else:
            storage_folder = os.path.join("data", "california", str(n))
        os.makedirs(storage_folder, exist_ok=True)

        for i in tqdm(range(N_PRECOMP)):
            game = CaliforniaGame(seed=i, set_zero=False)
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
