import numpy as np
import pandas as pd
import math
import torch

from typing import Union, Callable

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from rit.tree import DecisionTreeClassifier as MyDecisionTreeClassifier
from rit.tree import DecisionTreeRegressor as MyDecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from rit.ensemble import RandomForestClassifier as MyRandomForestClassifier
from rit.ensemble import RandomForestRegressor as MyRandomForestRegressor

from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, auc, roc_auc_score, log_loss, mean_squared_error, r2_score
from sklearn.base import clone, is_classifier, is_regressor

class CustomBoost(object):
    def __init__(
            self,
            base_estimator: None,
            n_estimators: int = 10000,
            rit_alpha: float = None,
            max_depth: int = None,
            max_leaf_nodes: int = None,
            nu: float = 1,
            loss: callable = mean_squared_error,
            random_state: int = 0,
            task: bool = None, # True = Regression, False == Classification
            ):

        self.n_estimators = n_estimators
        self.rit_alpha = rit_alpha
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.loss = loss
        self.random_state = random_state
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.nu = nu
        self.task = task

        self.classes_ = None
        self.n_classes_ = 0
        self.fin_i = 0

        if base_estimator is None:
            if task or loss == mean_squared_error:
                self.base_estimator = MyDecisionTreeRegressor(
                    random_state=random_state,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    rit_alpha=rit_alpha
                )
            elif not task or loss == log_loss:
                self.base_estimator = MyDecisionTreeClassifier(
                    random_state=random_state,
                    max_depth=max_depth,
                    max_leaf_nodes=max_leaf_nodes,
                    rit_alpha=rit_alpha
                )
        else:
            self.base_estimator = base_estimator

    def _check_estimator_type(self):
        if self.loss == log_loss and not is_classifier(self.base_estimator):
            raise ValueError("base_estimator should be classifier")
        elif self.loss != log_loss and not is_regressor(self.base_estimator):
            raise ValueError("base_estimator should be regressor")

    def get_params(self, deep=True) -> None:
        return {
            'n_estimators': self.n_estimators,
            'rit_alpha': self.rit_alpha,
            'max_depth': self.max_depth,
            'max_leaf_nodes': self.max_leaf_nodes,
            'nu': self.nu,
            'base_estimator': self.base_estimator,
            'loss': self.loss,
            'random_state': self.random_state,
            'task': self.task
        }

    def set_params(self, **params):
        estimator_params = {}
        for key in ['max_depth', 'max_leaf_nodes', 'rit_alpha']:
            if key in params:
                estimator_params[key] = params.pop(key)
        if estimator_params:
            self.base_estimator.set_params(**estimator_params)
            for key, value in estimator_params.items():
                setattr(self, key, value)
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).float().to(self.device)

        self.fin_i = -1
        self.estimators_ = [clone(self.base_estimator) for _ in range(self.n_estimators)]

        if self.task:
            f = torch.full_like(y_tensor, y_tensor.mean(), device=self.device)

            f.requires_grad_(True)

            for clf in self.estimators_:
                loss = torch.nn.functional.mse_loss(f, y_tensor)
                loss.backward()
                gradient = f.grad.cpu().numpy()
                clf.fit(X, -gradient)

                self.fin_i += 1
                if clf.tree_.node_count == 1:
                    break

                with torch.no_grad():
                    pred = torch.from_numpy(clf.predict(X)).float().to(self.device)
                    f += self.nu * pred
                    f.grad.zero_()

        elif not self.task:
            pos_prob = y_tensor.float().mean()
            self.initial_log_odds = torch.log(pos_prob / (1 - pos_prob)) if pos_prob not in {0, 1} else torch.tensor(0.0)
            f = torch.full_like(y_tensor, self.initial_log_odds, device=self.device)

            f.requires_grad_(True)

            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

            if self.n_classes_ > 2:
                raise NotImplementedError("Multiclass not supported")

            for clf in self.estimators_:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(f, y_tensor)
                loss.backward()
                gradient = f.grad.cpu().numpy()

                clf.fit(X, -gradient)
                self.fin_i += 1
                if clf.tree_.node_count == 1:
                    break

                with torch.no_grad():
                    pred = torch.from_numpy(clf.predict(X)).float().to(self.device)
                    f += self.nu * pred
                    f.grad.zero_()

        else:
            raise NotImplementedError(
                'wrong estimator class, estimator class should ve either MyDecisionTreeRegressor or MyDecisionTreeClassifier')

    def predict_proba(self, X):
        if self.task:
            raise ValueError("predict_proba only for classification")
        
        f = np.zeros(X.shape[0])
        for clf in self.estimators_[:self.fin_i + 1]:
            f += self.nu * clf.predict(X)
        
        proba = 1 / (1 + np.exp(-f))
        return np.vstack([1 - proba, proba]).T

    def predict(self, X):
        X_tensor = torch.from_numpy(X).float().to(self.device)
        if self.fin_i == -1:
            if self.task:
                return np.zeros(X.shape[0])
            else:
                return np.zeros(X.shape[0], dtype=int)
        if self.task:
            preds = [self.nu * torch.from_numpy(clf.predict(X)).to(self.device)
                   for clf in self.estimators_[:self.fin_i + 1]]
            return torch.stack(preds).sum(dim=0).cpu().numpy()
        
        elif not self.task:
            logits = torch.zeros(X.shape[0], device=self.device)
            for clf in self.estimators_[:self.fin_i + 1]:
                logits += self.nu * torch.from_numpy(clf.predict(X)).to(self.device)
            proba = torch.sigmoid(logits).cpu().numpy()
            predictions = (proba >= 0.5).astype(int)
            return predictions