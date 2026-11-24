# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import PresenceMetrics
from ultralytics.utils.plotting import overlay_heatmap
from ultralytics.utils.tal import fuse_presence_logits, presence_batch_targets


class PresenceValidator(DetectionValidator):
    """
    Validator for the presence task.

    Builds the per-level presence targets from the box labels (same rule as in training) and scores every grid cell
    of every level as a binary prediction: precision, recall and F1 at `pres_conf`, and AP over all thresholds. Cells
    that the targets ignore (partly covered by a box) are not scored. Also scores the fused map and the agreement
    between adjacent levels (see PresenceMetrics).

    Examples:
        >>> from ultralytics.models.yolo.presence import PresenceValidator
        >>> args = dict(model="best.pt", data="coco8.yaml")
        >>> validator = PresenceValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """Initialize the validator with presence metrics."""
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "presence"
        self.metrics = PresenceMetrics(conf=self.args.pres_conf)
        if self.args.save_json or self.args.save_txt:
            LOGGER.warning("save_json and save_txt are not supported for task=presence, disabling them.")
            self.args.save_json = self.args.save_txt = False

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Reset counters and metrics for a new validation run."""
        self.names = model.names
        self.nc = len(model.names)
        self.seen = 0
        self.nt = 0  # number of labelled instances
        self.jdict = []
        self.metrics.reset()
        self.metrics.names = model.names
        self.metrics.conf = self.args.pres_conf

    def get_desc(self) -> str:
        """Return the header of the results table."""
        return ("%22s" + "%11s" * 6) % ("Level", "Images", "Instances", "P", "R", "F1", "AP")

    def postprocess(self, preds: Any) -> Dict[str, List[torch.Tensor]]:
        """Split the model output into per-level probabilities and logits (tuple from PyTorch, list via AutoBackend)."""
        return {"probs": list(preds[0]), "logits": list(preds[1])}

    def update_metrics(self, preds: Dict[str, List[torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Accumulate the predictions of a batch with its presence targets.

        During training the loss has already built the targets of this batch, and they are reused from the batch.
        """
        logits = preds["logits"]
        bs, _, h, _ = batch["img"].shape
        shapes = [tuple(x.shape[2:]) for x in logits]
        strides = [h / s[0] for s in shapes]
        targets, weights = presence_batch_targets(
            batch,
            bs,
            shapes,
            strides,
            self.nc,
            self.args.pres_tau,
            self.args.pres_inst_pow,
            device=logits[0].device,
        )
        fused = fuse_presence_logits(logits)
        preds["fused"] = fused.sigmoid()  # for the plots
        self.metrics.update(logits, targets, [w > 0 for w in weights], fused, strides)
        self.seen += bs
        self.nt += len(batch["cls"])

    def finalize_metrics(self) -> None:
        """Attach speed and save directory to the metrics."""
        self.metrics.speed = self.speed
        self.metrics.save_dir = self.save_dir

    def get_stats(self) -> Dict[str, Any]:
        """Compute the metrics from the accumulated counts."""
        self.metrics.process()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print one row per level, one for the fused map, and the level agreement."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 4  # print format
        for row in self.metrics.summary():
            LOGGER.info(pf % (row["Level"], self.seen, self.nt, row["P"], row["R"], row["F1"], row["AP"]))
        LOGGER.info(
            f"{'level agreement':>22}  predictions {self.metrics.level_agree:.3g}, "
            f"targets {self.metrics.level_agree_gt:.3g} (IoU of positive cells, adjacent levels)"
        )
        if self.nt == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

    def plot_predictions(self, batch: Dict[str, Any], preds: Dict[str, List[torch.Tensor]], ni: int) -> None:
        """
        Save heatmap overlays for the first images of a batch: one row per image, one panel per level plus the fused
        map, each blended over the network input image.
        """
        names = [f"P{int(np.log2(batch['img'].shape[2] / p.shape[2]))}" for p in preds["probs"]] + ["fused"]
        maps = preds["probs"] + [preds["fused"]]
        rows = []
        for i in range(min(len(batch["img"]), 8)):
            img = (batch["img"][i].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]  # BGR
            scale = 320 / max(img.shape[:2])
            img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)))
            panels = []
            for name, m in zip(names, maps):
                panel = overlay_heatmap(img, m[i].amax(0))  # max over classes
                cv2.putText(panel, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                panels.append(panel)
            rows.append(np.hstack(panels))
        fname = self.save_dir / f"val_batch{ni}_pred.jpg"
        cv2.imwrite(str(fname), np.vstack(rows))
        if self.on_plot:
            self.on_plot(fname)
