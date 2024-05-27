import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..base.ml_game import MLGame
from ..utils.helper_functions import compute_mean_mode, identity_function


class SklearnLocalMLGame(MLGame):
    """
    A subclass of MLGame specifically designed for scikit-learn models.

    This class handles local feature importance games using scikit-learn models.
    It supports various imputation strategies and handles both regression and binary classification tasks.

    Attributes:
        task_type (str): The type of ML task, e.g., 'REG' for regression or 'BINARY_CLF' for binary classification.
        X (np.ndarray): Feature data.
        y (np.ndarray): Target variable.
        model (Any): The scikit-learn model to be used.
        players (List[Union[str, int]]): List of player names or indices.
        imputer (Callable): Strategy for imputing data.
        central_measures (np.ndarray): Central measures computed from background data.
        value_empty_set (float): The value computed for an empty set of features.
    """

    def __init__(self,
                 X: Union[pd.DataFrame, np.ndarray],
                 y: np.ndarray,
                 model: Any,
                 player_names_or_indices: List[Union[str, int]],
                 imputation_strategy: Union[Callable, str],
                 task_type: str,
                 get_null_set_predictions: Optional[Callable] = None,
                 impute_params: Optional[Dict[str, Any]] = None,
                 lower_is_better: bool = False,
                 memory_efficient: bool = True
                 ):
        """
        Initializes the SklearnLocalMLGame class.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Feature data.
            y (np.ndarray): Target variable.
            model (Any): The scikit-learn model to be used.
            player_names_or_indices (List[Union[str, int]]): List of player names or indices.
            imputation_strategy (Union[Callable, str]): Strategy for imputing data.
            task_type (str): Type of ML task ('REG' or 'BINARY_CLF').
            get_null_set_predictions (Optional[Callable]): Function to get predictions for an empty set.
            impute_params (Optional[Dict[str, Any]]): Additional parameters for the imputer function.
            lower_is_better (bool): Whether a lower value is better for the loss function.
            memory_efficient (bool): If True, uses memory-efficient storage.
        """

        super(SklearnLocalMLGame, self).__init__(X=X,
                                                 y=y,
                                                 model=model,
                                                 player_names_or_indices=player_names_or_indices,
                                                 get_null_set_predictions=get_null_set_predictions,
                                                 imputation_strategy=imputation_strategy,
                                                 impute_params=impute_params, lower_is_better=lower_is_better,
                                                 memory_efficient=memory_efficient
                                                 )

        self.task_type = task_type
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            self.X = self.X.values

        try:
            self.X = self.X.reshape(1, -1)
        except ValueError:
            raise ValueError("Enter a 1D array for local feature importance games")

        self.loss_function = identity_function

        if callable(imputation_strategy):
            self.imputer = imputation_strategy
        elif isinstance(imputation_strategy, str):
            if self.imputation_strategy == 'marginal':
                self.imputer = self._impute_marginals
            elif self.imputation_strategy == 'conditional':
                self.imputer = self._impute_conditionals
            elif self.imputation_strategy == 'impute_central_measures':
                self.imputer = self._impute_central_measures
        else:
            raise ValueError("The argument is neither a callable nor a valid string.")

        if get_null_set_predictions is not None:
            self.get_null_set_predictions = get_null_set_predictions

        self.central_measures = compute_mean_mode(self.impute_params['background_X'])

        self.value_empty_set = self._compute_value([])

    def _compute_value(self, s: List) -> float:
        """
        Computes the value of a given subset of players.

        Args:
            s (List): A subset of players.

        Returns:
            float: The computed value for the subset.
        """
        if len(s) > 0:
            is_player_present = all(elem in self.players for elem in s)
            if is_player_present:
                pred_imputed_X, _ = self.imputer(s=s, model=self.model, X=self.X)
            else:
                raise ValueError("Players in this coalition were not provided during instantiation")
        else:
            pred_imputed_X, _ = self.imputer(s=s, model=self.model, X=self.X)
        value = self.loss_function(self.y, pred_imputed_X)
        return value

    def _impute_central_measures(self,
                                 s: List,
                                 model: Any,
                                 X: Union[np.ndarray, pd.DataFrame]
                                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Imputes the central measures for a subset of features.

        This method replaces the values of the features not in the subset 's' with their corresponding central measures
        (mean or mode) and then predicts using the model. It supports regression and binary classification tasks.

        Args:
            s (List): The subset of features/players for which the values will not be imputed.
            model (Any): The scikit-learn model to be used for prediction.
            X (Union[np.ndarray, pd.DataFrame]): The input data where imputation is to be applied.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: The predicted values after imputation
        """
        X_imputed = X.copy()
        s_c = list(set(self.players) - set(s))
        if len(s_c) > 0:
            player_indices = sorted([element.number for element in s_c])
            dataset_indices = sorted([element.column_idx for element in s_c])
            central_values = self.central_measures[player_indices]
            X_imputed[:, dataset_indices] = central_values
        if self.task_type == 'REG':
            pred_X_imputed = model.predict(X_imputed)
        elif self.task_type == 'BINARY_CLF':
            pred_X_imputed = model.predict_proba(X_imputed)[0][1]
        else:
            raise NotImplementedError("This task is not supported yet")
        return pred_X_imputed, None

    def _remove_and_refit(self,
                          s: List[Union[str, int]],
                          model: Any,
                          X: np.ndarray,
                          y: np.ndarray,
                          impute_params: Optional[Dict[str, Any]] = None
                          ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

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
