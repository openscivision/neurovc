"""Streaming-friendly FlowMag wrappers compatible with momag-style interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pathlib import Path

from neurovc.contrib.flowmag.model import MotionMagModel


class OnlineMagnifier(Protocol):
    """Protocol matching OnlineMotionMagnifier ergonomics."""

    def update_reference(self, ref: Any, *, landmarks: Any | None = None) -> None: ...

    def __call__(
        self,
        frame: Any,
        *,
        alpha: float | None = None,
        mask: Any | None = None,
        landmarks: Any | None = None,
        depth: Any | None = None,
        return_info: bool = False,
    ) -> Any: ...


def unwrap(model: nn.Module) -> nn.Module:
    """Return the underlying module if wrapped by DP/DDP."""
    return model.module if hasattr(model, "module") else model


def default_alpha_policy(alpha: float, max_alpha: float) -> Tuple[float, int]:
    """Match FlowMag inference recursion (sqrt if above max_alpha)."""
    if alpha <= max_alpha:
        return alpha, 1
    a = alpha**0.5
    if a <= max_alpha:
        return a, 2
    raise ValueError(f"alpha={alpha} exceeds supported range (max={max_alpha})")


def _to_tensor(img: Any, device: torch.device) -> Tuple[torch.Tensor, bool]:
    """Convert HWC BGR/RGB/CHW to CHW float tensor; returns (tensor, swapped_to_rgb)."""
    swapped = False
    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 3 and t.shape[0] in (1, 3):
            pass
        elif t.ndim == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
    else:
        arr = np.asarray(img)
        if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
            raise ValueError("expected HxWxC image with 1 or 3 channels")
        # assume OpenCV BGR input; swap to RGB for the model
        arr = arr[..., ::-1]
        swapped = True
        t = torch.from_numpy(arr).permute(2, 0, 1)
    t = t.float()
    if t.max() > 1.5:
        t = t / 255.0
    return t.to(device), swapped


def _to_numpy(img: torch.Tensor, ref_like: Any, *, swap_to_bgr: bool) -> np.ndarray:
    arr = img.detach().cpu().permute(1, 2, 0).numpy()
    if swap_to_bgr:
        arr = arr[..., ::-1]
    if isinstance(ref_like, np.ndarray) and ref_like.dtype == np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def _crop_to_multiple(
    tensor: torch.Tensor, multiple: int = 8, target_hw: Tuple[int, int] | None = None
) -> torch.Tensor:
    _, h, w = tensor.shape
    if target_hw is None:
        h_new = (h // multiple) * multiple
        w_new = (w // multiple) * multiple
    else:
        h_new = min(h, target_hw[0])
        w_new = min(w, target_hw[1])
        h_new = (h_new // multiple) * multiple
        w_new = (w_new // multiple) * multiple
    return tensor[:, :h_new, :w_new]


def load_flowmag_model(
    *,
    config_path: str | None = None,
    checkpoint_path: str,
    device: str | torch.device = "cuda",
    training: bool = False,
) -> Tuple[nn.Module, float, Optional[int]]:
    """Build and load a FlowMag model; returns (model, max_alpha, epoch)."""
    if config_path is None or not Path(config_path).exists():
        # fall back to packaged default config
        import neurovc.contrib.flowmag as flowmag_pkg

        default_cfg = (
            Path(flowmag_pkg.__file__).resolve().parent
            / "configs"
            / "alpha16.color10.yaml"
        )
        if not default_cfg.exists():
            raise FileNotFoundError(
                f"Could not find FlowMag config at '{config_path}' "
                f"or default '{default_cfg}'"
            )
        config_path = str(default_cfg)

    config = OmegaConf.load(str(config_path))
    config.config = str(config_path)
    config.train.ngpus = 1
    config.train.is_training = bool(training)
    config.data.batch_size = 1

    dev = torch.device(device)
    model = MotionMagModel(config).to(dev)
    model = nn.DataParallel(model, device_ids=[0])
    chkpt = torch.load(checkpoint_path, map_location=dev)
    model.load_state_dict(chkpt["state_dict"], strict=False)
    max_alpha = float(config.train.alpha_high)
    epoch = chkpt.get("epoch")
    return model, max_alpha, epoch


class FlowMagOnline:
    """Stateful wrapper for frame-pair inference with FlowMag."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: str | torch.device = "cuda",
        max_alpha: float | None = None,
        default_alpha: float = 2.0,
        alpha_policy: Callable[
            [float, float], Tuple[float, int]
        ] = default_alpha_policy,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        m = unwrap(self.model)
        if max_alpha is not None:
            self.max_alpha = max_alpha
        else:
            cfg = getattr(m, "config", None)
            if (
                cfg is not None
                and hasattr(cfg, "train")
                and hasattr(cfg.train, "alpha_high")
            ):
                self.max_alpha = float(cfg.train.alpha_high)
            else:
                self.max_alpha = 16.0
        self.default_alpha = default_alpha
        self.alpha_policy = alpha_policy
        self.training_status = bool(getattr(m, "training", False))
        self._ref: torch.Tensor | None = None
        self._ref_like: Any = None
        self._target_hw: Tuple[int, int] | None = None
        self._ref_was_bgr: bool = False

        self.model.eval()

    def update_reference(self, ref: Any, landmarks: Any | None = None) -> None:
        ref_t, ref_swapped = _to_tensor(ref, self.device)
        ref_t = _crop_to_multiple(ref_t, multiple=8)
        self._ref = ref_t
        self._ref_like = ref
        self._ref_was_bgr = ref_swapped
        _, h, w = ref_t.shape
        self._target_hw = (h, w)

    def _forward_pair(
        self,
        ref: torch.Tensor,
        frame: torch.Tensor,
        *,
        alpha: float,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        our_alpha, num_recursions = self.alpha_policy(alpha, self.max_alpha)
        frames = torch.stack([ref.unsqueeze(0), frame.unsqueeze(0)], dim=2)
        preds = None
        for _ in range(num_recursions):
            out = self.model(frames, alpha=our_alpha, mask=mask)
            preds = out[0] if isinstance(out, (tuple, list)) else out
            frames = torch.stack([ref.unsqueeze(0), preds[0, :, 0].unsqueeze(0)], dim=2)
        return preds[0, :, 0]

    def __call__(
        self,
        frame: Any,
        *,
        alpha: float | None = None,
        mask: Any | None = None,
        landmarks: Any | None = None,
        depth: Any | None = None,
        return_info: bool = False,
    ) -> Any:
        if self._ref is None:
            self.update_reference(frame, landmarks)
        frame_t, frame_swapped = _to_tensor(frame, self.device)
        ref_t = self._ref
        if ref_t is None:
            raise RuntimeError("Reference frame is not set.")

        frame_t = _crop_to_multiple(frame_t, multiple=8, target_hw=self._target_hw)
        ref_t = _crop_to_multiple(ref_t, multiple=8, target_hw=self._target_hw)
        self._ref = ref_t

        mask_t: torch.Tensor | None = None
        if mask is not None:
            mask_t, _ = _to_tensor(mask, self.device)
            mask_t = _crop_to_multiple(mask_t, multiple=8, target_hw=self._target_hw)
            if mask_t.ndim == 3 and mask_t.shape[0] != 1:
                mask_t = mask_t[0:1]

        used_alpha = self.default_alpha if alpha is None else alpha
        with torch.no_grad():
            pred = self._forward_pair(ref_t, frame_t, alpha=used_alpha, mask=mask_t)

        output = _to_numpy(
            pred,
            self._ref_like if self._ref_like is not None else frame,
            swap_to_bgr=self._ref_was_bgr or frame_swapped,
        )
        if return_info:
            info = {
                "alpha": used_alpha,
                "max_alpha": self.max_alpha,
                "device": str(self.device),
            }
            return output, info
        return output


@dataclass
class TTAStep:
    loss: float
    info: Dict[str, float]


class FlowMagTTA:
    """Minimal test-time adaptation helper, separated from I/O concerns."""

    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-4,
        device: str | torch.device = "cuda",
        optimizer_factory: Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
        | None = None,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        m = unwrap(self.model)
        params = (
            m.trainable_parameters()
            if hasattr(m, "trainable_parameters")
            else self.model.parameters()
        )
        self.opt = (
            optimizer_factory(params)
            if optimizer_factory is not None
            else torch.optim.Adam(params, lr=lr)
        )

    def step(self, batch: torch.Tensor, *, alpha: float | None = None) -> TTAStep:
        self.model.train()
        batch = batch.to(self.device)
        out = self.model(batch, alpha=alpha)
        if isinstance(out, (tuple, list)):
            _, loss, info = out
        else:
            raise RuntimeError("Model did not return loss tuple in TTA mode.")

        loss = loss.mean()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        info_f = {k: float(v.detach().mean().cpu()) for k, v in (info or {}).items()}
        return TTAStep(loss=float(loss.detach().cpu()), info=info_f)


__all__ = [
    "OnlineMagnifier",
    "FlowMagOnline",
    "FlowMagTTA",
    "TTAStep",
    "unwrap",
    "default_alpha_policy",
    "load_flowmag_model",
]
