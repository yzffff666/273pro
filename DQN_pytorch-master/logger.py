from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision.utils as vutils
import torch

class Logger(object):
    def __init__(self, log_dir):
        """Create a SummaryWriter logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable (e.g., reward, loss)."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step):
        """Log a batch of images (expects [B, C, H, W] or [B, H, W, C])."""
        if isinstance(images, np.ndarray):
            if images.ndim == 4 and images.shape[-1] in [1, 3]:  # [B, H, W, C]
                images = torch.from_numpy(images).permute(0, 3, 1, 2) / 255.0  # Convert to [B, C, H, W]
            else:
                images = torch.from_numpy(images)
        image_grid = vutils.make_grid(images)
        self.writer.add_image(tag, image_grid, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of tensor or array values."""
        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values)
        self.writer.add_histogram(tag, values, step, bins=bins)

    def close(self):
        self.writer.close()