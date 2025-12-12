# RAFT wrappers based on the official RAFT github https://github.com/princeton-vl/RAFT

from os import mkdir
from os.path import join
from os.path import isdir, dirname, abspath, exists
from inspect import getsourcefile
from urllib.request import urlretrieve

import argparse

import torch

from neurovc.contrib.raft.raft import RAFT
from neurovc.contrib.raft.utils.flow_viz import flow_to_image
from neurovc.contrib.raft.utils.utils import InputPadder

import cv2

try:
    from torch.cuda.amp import GradScaler
except (ImportError, AttributeError):
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    # lightweight fallback that no-ops when tensorboard is unavailable
    class SummaryWriter:
        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None


class RAFTOpticalFlow:
    def __init__(
        self,
        iters=20,
        model=None,
        path=None,
        small=False,
        mixed_precision=False,
        alternate_corr=False,
        device=0,
    ):
        if model is None or not exists(model):
            model_path = join(dirname(abspath(getsourcefile(lambda: 0))), "models")
            model = join(model_path, "raft-casme2.pth")
        if not exists(model):
            if not isdir(model_path):
                mkdir(model_path)
            print("Model could not be found, downloading raft-casme2 model...")
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context
            urlretrieve(
                "https://cloud.hiz-saarland.de/s/McMNXZ5o7xteE6n/download/raft-casme2.pth",
                model,
            )
            print("done.")

        args = argparse.Namespace(
            model=model,
            path=path,
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
        )

        """ This does not work without a cuda device: """
        model = torch.nn.DataParallel(RAFT(args))
        """ And this does not work with the old state_dict: """
        # model = RAFT(args)
        model.load_state_dict(torch.load(args.model))

        self.model = model.module
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.last_flow = None
        self.padder = None
        self.iters = iters

    def init(self):
        self.last_flow = None
        self.padder = None

    def calc(self, ref, frame, flow=None):
        ref_torch = torch.from_numpy(ref).permute(2, 0, 1).float()
        ref_torch = ref_torch[None].to(self.device)
        frame_torch = torch.from_numpy(frame).permute(2, 0, 1).float()
        frame_torch = frame_torch[None].to(self.device)

        if self.padder is None:
            self.padder = InputPadder(ref_torch.shape)
        ref_torch, frame_torch = self.padder.pad(ref_torch, frame_torch)

        flow_low, flow_up = self.model(
            ref_torch, frame_torch, iters=self.iters, flow_init=flow, test_mode=True
        )
        flow_up = self.padder.unpad(flow_up)
        self.last_flow = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()

        return self.last_flow

    def visualize(self, title="flow", flow=None):
        if flow is None and self.last_flow is not None:
            cv2.imshow(title, flow_to_image(self.last_flow))
        if flow is not None:
            cv2.imshow(title, flow_to_image(flow))


class RAFTLogger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [
            self.running_loss[k] / self.SUM_FREQ
            for k in sorted(self.running_loss.keys())
        ]
        training_str = "[{:6d}, {:10.7f}] ".format(
            self.total_steps + 1, self.scheduler.get_last_lr()[0]
        )
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()

        for k in self.running_loss:
            self.writer.add_scalar(
                k, self.running_loss[k] / self.SUM_FREQ, self.total_steps
            )
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % self.SUM_FREQ == self.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()
