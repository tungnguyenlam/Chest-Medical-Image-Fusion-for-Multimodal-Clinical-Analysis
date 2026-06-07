from .AsymetricLoss import AsymetricLoss
from .CompositeLoss import CompositeLoss
from .FocalLoss import FocalLoss

# Name -> loss class. Names are what ``--loss`` / ``model.loss`` accept (resolved
# case-insensitively). Register a new loss here to make it generally available.
LOSS_REGISTRY = {
    "ASL": AsymetricLoss,
    "FC": FocalLoss,
}

__all__ = ["AsymetricLoss", "FocalLoss", "CompositeLoss", "LOSS_REGISTRY"]
