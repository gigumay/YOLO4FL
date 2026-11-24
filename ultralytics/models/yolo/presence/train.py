# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
from typing import Optional

from ultralytics.models import yolo
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import PresenceModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_results


class PresenceTrainer(DetectionTrainer):
    """
    Trainer for the presence task.

    Everything data-related is inherited from DetectionTrainer: the dataset still holds the usual YOLO box labels, and
    the presence targets are derived from them on the fly inside the loss.

    Examples:
        >>> from ultralytics.models.yolo.presence import PresenceTrainer
        >>> args = dict(model="yolo11n-presence.yaml", data="coco8.yaml", epochs=3, single_cls=True)
        >>> trainer = PresenceTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[dict] = None, _callbacks=None):
        """Initialize a PresenceTrainer, forcing task='presence'."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "presence"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg: Optional[str] = None, weights: Optional[str] = None, verbose: bool = True):
        """Return a PresenceModel, optionally initialized from weights (matching layers only)."""
        model = PresenceModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a PresenceValidator."""
        self.loss_names = ("pres_loss",)
        return yolo.presence.PresenceValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_metrics(self):
        """Plot losses and presence metrics from the results CSV."""
        plot_results(file=self.csv, presence=True, on_plot=self.on_plot)  # save results.png
