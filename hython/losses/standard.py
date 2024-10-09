from typing import Optional, List

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.distributions import Normal

class RMSELoss(_Loss):
    __name__ = "RMSE"

    def __init__(
        self,
        target_weight: dict = None,
    ):
        """
        Root Mean Squared Error (RMSE) loss for regression task.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(RMSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.target_weight = target_weight

    def forward(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Shape
        y_true: torch.Tensor of shape (N, T).
        y_pred: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets.

        Returns:
        torch.Tensor: The RMSE loss.
        """
        if self.target_weight is None:
            total_rmse_loss = torch.sqrt(self.mseloss(y_true, y_pred))
        else:
            if len(self.target_weight.keys()) > 1:
                total_rmse_loss = 0
                for idx, k in enumerate(self.target_weight):
                    w = self.target_weight[k]
                    rmse_loss = torch.sqrt(self.mseloss(y_true[:, idx], y_pred[:, idx]))
                    loss = rmse_loss * w
                    total_rmse_loss += loss
            else:
                total_rmse_loss = torch.sqrt(self.mseloss(y_true, y_pred))

        return total_rmse_loss


class MSELoss(_Loss):
    __name__ = "MSE"

    def __init__(
        self,
        target_weight: dict = None,
    ):
        """
        Root Mean Squared Error (RMSE) loss for regression task.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(MSELoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.target_weight = target_weight

    def forward(self, y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) between two tensors.

        Parameters:
        y_true (torch.Tensor): The true values.
        y_pred (torch.Tensor): The predicted values.

        Shape
        y_true: torch.Tensor of shape (N, T).
        y_pred: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets.

        Returns:
        torch.Tensor: The RMSE loss.
        """
        if self.target_weight is None:
            total_mse_loss = self.mseloss(y_true, y_pred)

        else:
            total_mse_loss = 0
            for idx, k in enumerate(self.target_weight):
                w = self.target_weight[k]
                mse_loss = self.mseloss(y_true[:, idx], y_pred[:, idx])
                loss = mse_loss * w
                total_mse_loss += loss

        return total_mse_loss

class nll_loss(_Loss):
    __name__ = "NLL"

    def __init__(
        self,
        target_weight: dict = None,
    ):
        """
        Negative log-likelihood (NLL) loss for normal distribution.

         Parameters:
         target_weight: List of targets that contribute in the loss computation, with their associated weights.
                        In the form {target: weight}
        """

        super(nll_loss, self).__init__()
        self.target_weight = target_weight

    def forward(self, y_true, distr_mean, distr_std):
        """
        Calculate the negative log-likelihood of the underlying normal distribution.

        Parameters:
        y_true (torch.Tensor): The true values.
        distr_mean (torch.Tensor): The predicted mean values. 
        distr_std (torch.Tensor): The predicted std values.

        Shape
        y_true: torch.Tensor of shape (N, T).
        distr_mean: torch.Tensor of shape (N, T).
        distr_std: torch.Tensor of shape (N, T).
        (256,3) means 256 samples with 3 targets.

        Returns:
        torch.Tensor: The NLL loss.
        """
        if self.target_weight is None:
            dist = Normal(distr_mean, distr_std)
            total_nll_loss = -dist.log_prob(y_true).mean()

        else:
            total_nll_loss = 0
            for idx, k in enumerate(self.target_weight):
                w = self.target_weight[k]
                dist = Normal(distr_mean[:, idx], distr_std[:, idx])
                nll_loss = -dist.log_prob(y_true[:, idx]).mean()
                loss = nll_loss * w
                total_nll_loss += loss

        return total_nll_loss