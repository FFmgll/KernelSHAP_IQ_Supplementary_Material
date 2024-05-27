import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import train_test_split

from ..base.ml_game import MLGame
from ..base.player import Player


class SklearnMLGame(MLGame):
    """
    A subclass of MLGame specifically tailored for use with scikit-learn models.

    Attributes:
        task_type (str): The type of ML task, e.g., 'REG' for regression or 'BINARY_CLF' for binary classification.
    """

    def __init__(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: np.ndarray,
                 model: Any,
                 player_names_or_indices: List[Union[str, int]],
                 imputation_strategy: Union[Callable, str],
                 task_type: str,
                 loss_function: Optional[Callable] = None,
                 get_null_set_predictions: Optional[Callable] = None,
                 impute_params: Optional[Dict[str, Any]] = None,
                 lower_is_better: bool = True,
                 memory_efficient: bool = True
                 ):
        """
        Initializes the SklearnMLGame class.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature data.
            y (np.ndarray): Target variable.
            model (Any): The scikit-learn model to be used.
            player_names_or_indices (List[str]): List of player names.
            imputation_strategy (Union[Callable, str]): Strategy for imputing data.
                                                        It can be a callable or one of the predefined strategies:
                                                        'marginal', 'conditional', or 'remove_refit'.
            task_type (str): Type of ML task ('REG' or 'BINARY_CLF').
            loss_function (Optional[Callable]): Function to calculate the loss.
            get_null_set_predictions (Optional[Callable]): Function to get predictions for an empty set.
            impute_params (Optional[Dict[str, Any]]): Parameters for the imputer function. Expected keys include
                                                      'train_test_split' for the proportion of the dataset to include
                                                       in the test split and 'random_seed'
            for the random number generator seed.
            lower_is_better (bool): Whether a lower value is considered better for the loss function.
            memory_efficient (bool): If True, uses memory-efficient storage.
        """

        super(SklearnMLGame, self).__init__(X=X,
                                            y=y,
                                            model=model,
                                            player_names_or_indices=player_names_or_indices,
                                            get_null_set_predictions=get_null_set_predictions,
                                            imputation_strategy=imputation_strategy,
                                            impute_params=impute_params, lower_is_better=lower_is_better,
                                            memory_efficient=memory_efficient
                                            )

        self.task_type = task_type

        if isinstance(X, pd.DataFrame):
            self.X = self.X.values

        if loss_function is not None:
            self.loss_function = loss_function
        else:
            if self.task_type == 'REG':
                self.loss_function = mean_squared_error
            elif self.task_type == 'BINARY_CLF':
                self.loss_function = log_loss

        if callable(imputation_strategy):
            self.imputer = imputation_strategy
        elif isinstance(imputation_strategy, str):
            if self.imputation_strategy == 'marginal':
                self.imputer = self._impute_marginals
            elif self.imputation_strategy == 'conditional':
                self.imputer = self._impute_conditionals
            elif self.imputation_strategy == 'remove_refit':
                self.imputer = self._remove_and_refit
        else:
            raise ValueError("The argument is neither a callable nor a string.")

        if get_null_set_predictions is not None:
            self.get_null_set_predictions = get_null_set_predictions
        else:
            self.get_null_set_predictions = self._get_null_set_predictions

        self.value_empty_set = self._compute_value([])

    def _get_null_set_predictions(self, y: np.ndarray) -> np.ndarray:
        """
        Provides null set predictions based on the task type.

        Args:
            y (np.ndarray): The target variable.

        Returns:
            np.ndarray: Predicted values for the null set.
        """
        n_samples = y.shape[0]
        if self.task_type == 'REG' or self.task_type == 'BINARY_CLF':
            pred_imputed_X = np.repeat(np.mean(y), n_samples)
            return pred_imputed_X

    def _remove_and_refit(self,
                          s: List[Player],
                          model: Any,
                          X: np.ndarray,
                          y: np.ndarray,
                          impute_params: Optional[Dict[str, Any]] = None
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Removes a subset of players, refits the model, and predicts.

        Args:
            s (List[Union[str, int]]): Subset of players to remove.
            model (Any): The scikit-learn model.
            X (np.ndarray): Feature data.
            y (np.ndarray): Target variable.
            impute_params (Optional[Dict[str, Any]]): Parameters for the imputer.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Predicted values and optionally the test target values.
        """
        X_imputed = X.copy()
        s_c = list(set(self.players) - set(s))
        if len(s_c) > 0:
            dataset_indices = sorted([element.column_idx for element in s_c])
            X_imputed = np.delete(X_imputed, dataset_indices, axis=1)
        if 0 < impute_params['train_test_split'] < 1:
            X_train, X_test, y_train, y_test = train_test_split(X_imputed,
                                                                y,
                                                                random_state=impute_params['random_seed']
                                                                )
            new_model = model.fit(X_train, y_train)
            if self.task_type == 'REG':
                pred_X_imputed = new_model.predict(X_test)
            elif self.task_type == 'BINARY_CLF':
                pred_X_imputed = new_model.predict_proba(X_test)
            else:
                raise NotImplementedError("This task is not supported yet")
            return pred_X_imputed, y_test
        else:
            new_model = model.fit(X_imputed, y)
            if self.task_type == 'REG':
                pred_X_imputed = new_model.predict(X_imputed)
            elif self.task_type == 'BINARY_CLF':
                pred_X_imputed = new_model.predict_proba(X_imputed)
            else:
                raise NotImplementedError("This task is not supported yet")
            return pred_X_imputed, None

    def _impute_marginals(self,
                          S: List[List],
                          model: Any,
                          X: Any,
                          y: Optional[Any] = None,
                          impute_params: Optional[Any] = None
                          ) -> None:
        raise NotImplementedError

    def _impute_conditionals(self,
                             S: List[List],
                             model: Any,
                             X: Any,
                             y: Optional[Any] = None,
                             impute_params: Optional[Any] = None
                             ) -> None:
        raise NotImplementedError

    def save_config(self):
        raise NotImplementedError

    def load_config(self):
        raise NotImplementedError
