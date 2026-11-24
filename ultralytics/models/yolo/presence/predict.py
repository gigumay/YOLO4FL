# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.plotting import overlay_heatmap
from ultralytics.utils.tal import fuse_presence_maps


class PresenceResults(Results):
    """
    Inference results of a presence model for one image.

    All maps are raw probabilities in [0, 1] with no threshold applied, at native grid resolution (one value per
    cell). Grid cells that lie entirely in the letterbox padding are cropped off. Images are letterboxed top-left
    aligned, so cell (i, j) of a level with stride s covers original-image pixels
    [j*s/ratio, (j+1)*s/ratio) x [i*s/ratio, (i+1)*s/ratio), clipped to the image.

    Attributes:
        presence (List[torch.Tensor]): Per-level maps, each (nc, h_i, w_i).
        presence_strides (List[float]): Stride of each level in network-input pixels.
        presence_ratio (float): Resize ratio from the original image to the network input.
        presence_fused (torch.Tensor): Fused map (levels averaged in logit space) on the finest grid, (nc, h, w).
        presence_conf (float): Score at or above which `summary()` lists a cell (the predictor's `pres_conf`).
    """

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: Dict[int, str],
        presence: List[torch.Tensor],
        presence_strides: List[float],
        presence_ratio: float,
        presence_fused: torch.Tensor,
        presence_conf: float = 0.5,
        speed: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the results with the per-level maps, strides, resize ratio, fused map and summary threshold."""
        super().__init__(orig_img, path=path, names=names, speed=speed)
        self.presence = presence
        self.presence_strides = presence_strides
        self.presence_ratio = presence_ratio
        self.presence_fused = presence_fused
        self.presence_conf = presence_conf

    def __len__(self) -> int:
        """Return the number of levels."""
        return len(self.presence)

    def new(self):
        """Return a new PresenceResults with the same image, path, names, speed, maps and threshold."""
        return PresenceResults(
            self.orig_img,
            self.path,
            self.names,
            self.presence,
            self.presence_strides,
            self.presence_ratio,
            self.presence_fused,
            presence_conf=self.presence_conf,
            speed=self.speed,
        )

    def _apply(self, fn: str, *args, **kwargs):
        """Apply a tensor method (cpu, numpy, cuda, to) to all maps and return a new PresenceResults."""

        def convert(m):
            if fn == "numpy":  # also works for maps on the GPU or already converted
                return m if isinstance(m, np.ndarray) else m.detach().cpu().numpy()
            return getattr(torch.as_tensor(m), fn)(*args, **kwargs)  # NumPy maps (after .numpy()) become tensors

        r = self.new()
        r.presence = [convert(m) for m in self.presence]
        r.presence_fused = convert(self.presence_fused)
        return r

    def _level(self, level: Optional[int]):
        """Return the map and stride of a level index, or of the fused map (finest stride) when level is None."""
        if level is None:
            return self.presence_fused, min(self.presence_strides)
        return self.presence[level], self.presence_strides[level]

    def upsample_presence(self, level: Optional[int] = None) -> torch.Tensor:
        """
        Resize a presence map to the original image size; every pixel takes the value of the cell it falls in.

        Args:
            level (int, optional): Level index into `presence`, or None for the fused map.

        Returns:
            (torch.Tensor | np.ndarray): Map of shape (nc, H0, W0), of the same type as the stored maps.
        """
        m, s = self._level(level)
        t = torch.as_tensor(m)  # NumPy maps (after .numpy()) are viewed as tensors without a copy
        h0, w0 = self.orig_shape
        rows = (torch.arange(h0, device=t.device) * self.presence_ratio / s).long().clamp_(max=t.shape[1] - 1)
        cols = (torch.arange(w0, device=t.device) * self.presence_ratio / s).long().clamp_(max=t.shape[2] - 1)
        out = t[:, rows][:, :, cols]
        return out.numpy() if isinstance(m, np.ndarray) else out

    def plot(
        self,
        level: Optional[int] = None,
        alpha: float = 0.5,
        show: bool = False,
        save: bool = False,
        filename=None,
        **kwargs,
    ):
        """
        Blend a presence map (max over classes) over the original image.

        Args:
            level (int, optional): Level index into `presence`, or None for the fused map.
            alpha (float): Weight of the heatmap in the blend where the score is 1; it scales with the score.
            show (bool): Whether to display the image.
            save (bool): Whether to save the image to `filename`.
            filename (str, optional): Path to save the image to.
            **kwargs (Any): Ignored; accepted so the predictor's plotting call works unchanged.

        Returns:
            (np.ndarray): BGR image with the heatmap overlay.
        """
        heat = torch.as_tensor(self.upsample_presence(level)).amax(0)  # max over classes
        img = overlay_heatmap(self.orig_img, heat, alpha=alpha)
        if show:
            Image.fromarray(np.ascontiguousarray(img[..., ::-1])).show(title=str(self.path))  # BGR to RGB
        if save:
            cv2.imwrite(str(filename), img)
        return img

    def summary(
        self, normalize: bool = False, decimals: int = 5, level: Optional[int] = None, conf: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        List the flagged cells of a presence map, one row per cell and class, like the boxes of a detection result.

        A cell is flagged when its score is at least `conf`. Its box is the cell's rectangle in original-image pixels,
        clipped to the image. Rows are sorted by score, highest first. `to_df()`, `to_json()`, `to_csv()` and the
        other export methods are built on this.

        Args:
            normalize (bool): Whether to divide the box coordinates by the image width and height.
            decimals (int): Number of decimals to round scores and coordinates to.
            level (int, optional): Level index into `presence`, or None for the fused map.
            conf (float, optional): Score threshold. Defaults to `presence_conf`.

        Returns:
            (List[Dict[str, Any]]): One dict per flagged cell with 'name', 'class', 'confidence', 'box' (x1, y1, x2,
                y2) and 'level' ('P3', 'P4', ... or 'fused').
        """
        m, s = self._level(level)
        m = torch.as_tensor(m).detach().float().cpu()
        conf = self.presence_conf if conf is None else conf
        h0, w0 = self.orig_shape
        gx, gy = (w0, h0) if normalize else (1, 1)
        side = s / self.presence_ratio  # cell side in original-image pixels
        name = "fused" if level is None else f"P{int(math.log2(s))}"
        c, i, j = torch.nonzero(m >= conf, as_tuple=True)
        scores = m[c, i, j]
        rows = []
        for k in torch.argsort(scores, descending=True, stable=True).tolist():
            ck, ik, jk = int(c[k]), int(i[k]), int(j[k])
            x1, y1, x2, y2 = jk * side, ik * side, min((jk + 1) * side, w0), min((ik + 1) * side, h0)
            rows.append(
                {
                    "name": self.names[ck],
                    "class": ck,
                    "confidence": round(float(scores[k]), decimals),
                    "box": {
                        "x1": round(x1 / gx, decimals),
                        "y1": round(y1 / gy, decimals),
                        "x2": round(x2 / gx, decimals),
                        "y2": round(y2 / gy, decimals),
                    },
                    "level": name,
                }
            )
        return rows

    def verbose(self) -> str:
        """Return a log string with the maximum presence probability per level."""
        return "".join(
            f"P{int(math.log2(s))} max {float(m.max()):.2f}, " for s, m in zip(self.presence_strides, self.presence)
        )

    def save_txt(self, *args, **kwargs):
        """Not supported for presence results."""
        LOGGER.warning("save_txt is not supported for task=presence.")

    def save_crop(self, *args, **kwargs):
        """Not supported for presence results."""
        LOGGER.warning("save_crop is not supported for task=presence.")


class PresencePredictor(DetectionPredictor):
    """
    Predictor for the presence task.

    Returns one PresenceResults per image with the unthresholded per-level heatmaps (see PresenceResults). Images are
    letterboxed top-left aligned (padding only at the bottom/right), so cell (i, j) of every level maps to a fixed
    region of the original image with no offset.

    Examples:
        >>> from ultralytics import YOLO
        >>> model = YOLO("runs/presence/train/weights/best.pt")
        >>> r = model.predict("image.jpg")[0]
        >>> [m.shape for m in r.presence]  # e.g. [(1, 60, 80), (1, 30, 40), (1, 15, 20)]
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the predictor with task='presence'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "presence"

    def pre_transform(self, im: List[np.ndarray]) -> List[np.ndarray]:
        """Letterbox as BasePredictor does, but pad only at the bottom/right so cell grids start at the image origin."""
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.pt or (getattr(self.model, "dynamic", False) and not self.model.imx)),
            stride=self.model.stride,
            center=False,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Crop the per-level maps to the image area and wrap them in PresenceResults."""
        probs, logits = list(preds[0]), list(preds[1])
        fused = fuse_presence_maps(logits)
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        h, w = img.shape[2:]
        strides = [h / p.shape[2] for p in probs]
        results = []
        for i, (orig_img, path) in enumerate(zip(orig_imgs, self.batch[0])):
            h0, w0 = orig_img.shape[:2]
            r = min(h / h0, w / w0)  # LetterBox resize ratio
            hu, wu = round(h0 * r), round(w0 * r)  # resized image size before padding

            def crop(m, s):
                """Keep only the cells that touch the image (drop cells lying fully in the padding)."""
                return m[i, :, : math.ceil(hu / s), : math.ceil(wu / s)]

            results.append(
                PresenceResults(
                    orig_img,
                    path=path,
                    names=self.model.names,
                    presence=[crop(p, s) for p, s in zip(probs, strides)],
                    presence_strides=strides,
                    presence_ratio=r,
                    presence_fused=crop(fused, min(strides)),
                    presence_conf=self.args.pres_conf,
                )
            )
        return results
