# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import PresenceMetrics
from ultralytics.utils.plotting import overlay_heatmap, plot_images
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
        False-positive regions (class -1, see PresenceDataset) are not targets and not counted as instances; their
        weight `pres_fp_weight` only matters to the loss, and is passed here so that the cached targets match.
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
            fp_weight=self.args.pres_fp_weight,
            device=logits[0].device,
        )
        fused = fuse_presence_logits(logits)
        valid = [w > 0 for w in weights]
        preds.update(fused=fused.sigmoid(), targets=targets, valid=valid)  # for the plots
        self.metrics.update(logits, targets, valid, fused, strides)
        self.seen += bs
        self.nt += int((batch["cls"] >= 0).sum())  # labelled animals only, not false-positive regions

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

    @staticmethod
    def most_accurate_map(
        probs: List[torch.Tensor], targets: List[torch.Tensor], valid: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Return, for each image, the index of the score map that best matches its targets.

        On an image with labels, the best map has the highest soft F1 over the scored cells: precision and recall
        computed with the scores p in place of 0/1 flags, i.e. F1 = 2 * sum(p * t) / (sum(p) + sum(t)). It rewards
        high scores on the positive cells and penalises score anywhere else, relative to the size of the animals
        rather than of the image, and needs no threshold. On an image without labels every map has an F1 of 0, and
        the best map is the one with the lowest mean score, i.e. the least false alarm per image area.

        Args:
            probs (List[torch.Tensor]): Candidate score maps, each (B, nc, h, w) with values in [0, 1].
            targets (List[torch.Tensor]): 0/1 targets of each map, same shapes.
            valid (List[torch.Tensor]): Boolean masks of the cells to score in each map, same shapes.

        Returns:
            (torch.Tensor): Index into `probs` of the best map of each image, shape (B,).
        """
        f1, mean, npos = [], [], []
        for p, t, v in zip(probs, targets, valid):
            p, t, v = (x.flatten(1).float() for x in (p, t, v))
            sp, st = (p * v).sum(1), (t * v).sum(1)
            f1.append(2 * (p * t * v).sum(1) / (sp + st).clamp(min=1e-9))
            mean.append(sp / v.sum(1).clamp(min=1))
            npos.append(st)
        labelled = (torch.stack(npos, 1) > 0).any(1, keepdim=True)
        return torch.where(labelled, torch.stack(f1, 1), -torch.stack(mean, 1)).argmax(1)

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """Do nothing: plot_predictions draws the labels, as heatmaps of the level it shows for each image."""

    def plot_predictions(self, batch: Dict[str, Any], preds: Dict[str, List[torch.Tensor]], ni: int) -> None:
        """
        Save heatmap mosaics of the first 16 images of a batch, one tile per image, blended over the network input.

        val_batch{ni}_pred.jpg shows, for each image, only the most accurate of its level maps and the fused map (see
        `most_accurate_map`), and val_batch{ni}_labels.jpg the targets of that same level, so the two mosaics compare
        tile by tile. The fused map is scored against, and shown with, the targets of the finest level, as in the
        metrics. Target cells are drawn at 1 if positive, 0 if negative, and 0.5 if ignored (not scored). Each tile is
        captioned with the level shown and the image file name.
        """
        probs = preds["probs"] + [preds["fused"]]
        finest = max(range(len(preds["probs"])), key=lambda i: probs[i].shape[-2] * probs[i].shape[-1])
        targets = preds["targets"] + [preds["targets"][finest]]
        valid = preds["valid"] + [preds["valid"][finest]]
        names = [f"P{int(np.log2(batch['img'].shape[2] / p.shape[2]))}" for p in preds["probs"]] + ["fused"]

        n = min(len(batch["img"]), 16)  # plot_images shows at most 16 images
        best = self.most_accurate_map([x[:n] for x in probs], [x[:n] for x in targets], [x[:n] for x in valid])
        best = best.tolist()
        h, w = batch["img"].shape[2:]
        scale = min(1.0, 1920 / math.ceil(n**0.5) / max(h, w))  # tile size in plot_images' 1920 px mosaic
        pred_tiles, label_tiles = [], []
        for i, k in enumerate(best):
            img = (batch["img"][i, :3].float().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[..., ::-1]  # BGR
            if scale < 1:
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            gt = targets[k][i].float() + 0.5 * (~valid[k][i]).float()  # 1 positive, 0.5 ignored, 0 negative
            pred_tiles.append(overlay_heatmap(img, probs[k][i].amax(0)))  # max over classes
            label_tiles.append(overlay_heatmap(img, gt.amax(0)))
        captions = [f"{names[k]} | {Path(f).name}" for k, f in zip(best, batch["im_file"])]
        for kind, tiles in (("labels", label_tiles), ("pred", pred_tiles)):
            plot_images(
                labels={},
                images=np.stack(tiles)[..., ::-1].transpose(0, 3, 1, 2).astype(np.float32) / 255,  # RGB CHW in [0, 1]
                paths=captions,
                fname=self.save_dir / f"val_batch{ni}_{kind}.jpg",
                on_plot=self.on_plot,
            )
