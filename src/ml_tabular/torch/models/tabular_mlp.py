from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from ml_tabular.exceptions import ModelError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


TaskType = Literal["regression", "binary", "multiclass"]


@dataclass(frozen=True)
class TabularMLPConfig:
    """Configuration for a fully-connected MLP on tabular data.

    Parameters
    ----------
    input_dim:
        Number of input features per example (D_in). Must be > 0.
    task_type:
        Type of prediction task:
          - "regression": continuous target(s), typically trained with MSE/BCE (non-logit).
          - "binary": binary classification (0/1) with a single logit output, typically
                      trained with BCEWithLogitsLoss.
          - "multiclass": multiclass classification with num_classes logits,
                          typically trained with CrossEntropyLoss.
    hidden_dims:
        Sequence of hidden layer sizes. If empty, the model is a single linear layer
        from input_dim to output_dim.
    activation:
        Name of activation function to use after each hidden layer.
        Supported: "relu", "gelu", "silu", "leaky_relu", "tanh", "elu".
    dropout:
        Dropout probability applied after each hidden layer (0.0 = no dropout).
    batch_norm:
        If True, apply BatchNorm1d after each hidden Linear layer.
        Mutually exclusive with layer_norm.
    layer_norm:
        If True, apply LayerNorm after each hidden Linear layer.
        Mutually exclusive with batch_norm.
    output_dim:
        Number of outputs for the final linear layer. For:
          - regression: defaults to 1 if None.
          - binary: must be 1 (if provided and != 1 -> error).
          - multiclass: must equal num_classes (if provided and != num_classes -> error).
    num_classes:
        Number of classes for multiclass classification. Required when task_type is
        "multiclass"; ignored otherwise.
    """

    input_dim: int
    task_type: TaskType = "regression"
    hidden_dims: Sequence[int] = (128, 64)
    activation: str = "relu"
    dropout: float = 0.0
    batch_norm: bool = False
    layer_norm: bool = False
    output_dim: Optional[int] = None
    num_classes: Optional[int] = None

    def effective_output_dim(self) -> int:
        """Compute the effective output dimension based on task_type and fields."""
        if self.input_dim <= 0:
            raise ModelError(
                "input_dim must be a positive integer.",
                code="tabular_mlp_bad_input_dim",
                context={"input_dim": self.input_dim},
                location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
            )

        # Regression: default to a single output if not specified
        if self.task_type == "regression":
            if self.output_dim is not None:
                if self.output_dim <= 0:
                    raise ModelError(
                        "output_dim must be positive for regression.",
                        code="tabular_mlp_bad_output_dim_regression",
                        context={"output_dim": self.output_dim},
                        location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
                    )
                return self.output_dim
            return 1

        # Binary classification: must be a single logit
        if self.task_type == "binary":
            if self.output_dim is not None and self.output_dim != 1:
                raise ModelError(
                    "For binary classification, output_dim must be 1 (single logit).",
                    code="tabular_mlp_bad_output_dim_binary",
                    context={"output_dim": self.output_dim},
                    location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
                )
            return 1

        # Multiclass classification
        if self.task_type == "multiclass":
            if self.num_classes is None:
                raise ModelError(
                    "num_classes must be provided for multiclass classification.",
                    code="tabular_mlp_missing_num_classes",
                    context={},
                    location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
                )
            if self.num_classes <= 1:
                raise ModelError(
                    "num_classes must be > 1 for multiclass classification.",
                    code="tabular_mlp_bad_num_classes",
                    context={"num_classes": self.num_classes},
                    location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
                )

            if self.output_dim is not None and self.output_dim != self.num_classes:
                raise ModelError(
                    "For multiclass classification, output_dim must equal num_classes.",
                    code="tabular_mlp_bad_output_dim_multiclass",
                    context={
                        "output_dim": self.output_dim,
                        "num_classes": self.num_classes,
                    },
                    location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
                )
            return self.num_classes

        # Should not reach here
        raise ModelError(
            f"Unsupported task_type: {self.task_type!r}.",
            code="tabular_mlp_bad_task_type",
            context={"task_type": self.task_type},
            location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.effective_output_dim",
        )

    def validate(self) -> None:
        """Perform additional sanity checks independent of output_dim logic."""
        if self.batch_norm and self.layer_norm:
            raise ModelError(
                "batch_norm and layer_norm are mutually exclusive; choose at most one.",
                code="tabular_mlp_norm_conflict",
                context={"batch_norm": self.batch_norm, "layer_norm": self.layer_norm},
                location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.validate",
            )

        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ModelError(
                "dropout must be in [0.0, 1.0).",
                code="tabular_mlp_bad_dropout",
                context={"dropout": self.dropout},
                location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.validate",
            )

        for h in self.hidden_dims:
            if h <= 0:
                raise ModelError(
                    "All hidden_dims must be positive integers.",
                    code="tabular_mlp_bad_hidden_dims",
                    context={"hidden_dims": list(self.hidden_dims)},
                    location="ml_tabular.torch.models.tabular_mlp.TabularMLPConfig.validate",
                )


# ---------------------------------------------------------------------------
# Utility: activation factory
# ---------------------------------------------------------------------------


def _make_activation(name: str) -> nn.Module:
    """Return an activation module by name."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()

    raise ModelError(
        f"Unsupported activation: {name!r}.",
        code="tabular_mlp_bad_activation",
        context={"activation": name},
        location="ml_tabular.torch.models.tabular_mlp._make_activation",
    )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class TabularMLP(nn.Module):
    """Fully-connected MLP for tabular data.

    This class is intentionally generic and minimal in behavior:

      - It accepts a dense 2D input tensor X of shape (batch_size, input_dim).
      - It produces raw logits or continuous outputs of shape (batch_size, output_dim).
      - It does NOT apply a final Sigmoid/Softmax by default, so that you can use
        BCEWithLogitsLoss / CrossEntropyLoss correctly.

    Depending on `config.task_type`:

      - "regression":
          * outputs are continuous values.
          * typical loss: nn.MSELoss, nn.L1Loss, etc.

      - "binary":
          * outputs are single logits per example (output_dim=1).
          * typical loss: nn.BCEWithLogitsLoss.
          * apply torch.sigmoid() in evaluation/inference if you need probabilities.

      - "multiclass":
          * outputs are logits over classes (output_dim=num_classes).
          * typical loss: nn.CrossEntropyLoss.
          * apply torch.softmax(logits, dim=-1) in evaluation/inference if you need probabilities.
    """

    def __init__(self, config: TabularMLPConfig) -> None:
        super().__init__()

        # Validate and compute effective output_dim
        config.validate()
        output_dim = config.effective_output_dim()

        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = output_dim

        layers = []
        in_dim = self.input_dim

        # Hidden layers
        for idx, hidden_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))

            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(_make_activation(config.activation))

            if config.dropout > 0.0:
                layers.append(nn.Dropout(p=config.dropout))

            in_dim = hidden_dim

        # Final output layer (no activation; logits/continuous outputs)
        layers.append(nn.Linear(in_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

        self._initialize_weights()

        logger.info(
            "Initialized TabularMLP: input_dim=%d, output_dim=%d, hidden_dims=%s, "
            "task_type=%s, activation=%s, dropout=%.3f, batch_norm=%s, layer_norm=%s",
            self.input_dim,
            self.output_dim,
            list(config.hidden_dims),
            config.task_type,
            config.activation,
            config.dropout,
            config.batch_norm,
            config.layer_norm,
        )

    def _initialize_weights(self) -> None:
        """Apply a sensible default weight initialization for MLPs."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming uniform works well for ReLU-family activations
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward + helpers
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (batch_size, input_dim). If a tensor of
            higher rank is provided, it will be flattened except for the
            batch dimension, i.e.:

                x = x.view(batch_size, -1)

            This is primarily a convenience; for tabular data you should
            normally provide x as (batch_size, D_in) already.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim):
              - regression: continuous values.
              - binary: logits of shape (batch_size, 1).
              - multiclass: logits of shape (batch_size, num_classes).
        """
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)

        if x.shape[1] != self.input_dim:
            raise ModelError(
                "Input tensor has wrong number of features.",
                code="tabular_mlp_bad_input_shape",
                context={"expected": self.input_dim, "got": int(x.shape[1])},
                location="ml_tabular.torch.models.tabular_mlp.TabularMLP.forward",
            )

        return self.net(x)

    # Convenience helpers (optional, for inference-time use)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities for classification tasks.

        For:
          - binary: applies sigmoid to logits and returns shape (batch_size, 1).
          - multiclass: applies softmax over logits and returns shape (batch_size, num_classes).

        For regression tasks, this method raises a ModelError.
        """
        logits = self.forward(x)

        if self.config.task_type == "binary":
            return torch.sigmoid(logits)

        if self.config.task_type == "multiclass":
            return torch.softmax(logits, dim=-1)

        raise ModelError(
            "predict_proba is only defined for classification tasks.",
            code="tabular_mlp_predict_proba_on_regression",
            context={"task_type": self.config.task_type},
            location="ml_tabular.torch.models.tabular_mlp.TabularMLP.predict_proba",
        )

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return the number of parameters in the model.

        Parameters
        ----------
        trainable_only:
            If True, count only parameters with requires_grad=True.

        Returns
        -------
        int
            Number of parameters.
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
