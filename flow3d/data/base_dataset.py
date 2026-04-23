from abc import abstractmethod
import os
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate, Sampler
import torch.nn.functional as F

DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


class BaseDataset(Dataset):
    segment_palette = list(DAVIS_PALETTE)

    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return torch.arange(self.num_frames)

    @abstractmethod
    def get_w2cs(self) -> torch.Tensor: ...

    @abstractmethod
    def get_Ks(self) -> torch.Tensor: ...

    @abstractmethod
    def get_image(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_depth(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_mask(self, index: int) -> torch.Tensor: ...

    def get_img_wh(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_tracks_3d(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns 3D tracks:
            coordinates (N, T, 3),
            visibles (N, T),
            invisibles (N, T),
            confidences (N, T),
            colors (N, 3)
        """
        ...

    @abstractmethod
    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns background points:
            coordinates (N, 3),
            normals (N, 3),
            colors (N, 3)
        """
        ...

    @staticmethod
    def train_collate_fn(batch):
        collated = {}
        for k in batch[0]:
            if k not in [
                "query_tracks_2d",
                "target_ts",
                "target_w2cs",
                "target_Ks",
                "target_tracks_2d",
                "target_visibles",
                "target_track_depths",
                "target_invisibles",
                "target_confidences",
                "target_track_masks",
            ]:
                collated[k] = default_collate([sample[k] for sample in batch])
            else:
                collated[k] = [sample[k] for sample in batch]
        return collated

class CustomBatchSampler(Sampler):
    """
    Custom BatchSampler that yields batches of (idx, ta, tb).
    Each batch contributes totally batch_size samples, each sample is drawn from every range repeatedly
    Each epoch contains num_batches batches
    """
    def __init__(self, ranges, batch_size: int, num_batches: int):
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be > 0")

        self._validate_and_set_ranges(ranges)

    def _validate_and_set_ranges(self, ranges):
        self.ranges = []
        for r in ranges:
            if not (isinstance(r, tuple) and len(r) == 4):
                raise TypeError(f"Each range must be a 4-tuple (a,b,ta,tb), got {r}")
            a, b, ta, tb = r
            if not all(isinstance(x, int) for x in (a, b, ta, tb)):
                raise TypeError(f"Range values must be integers, got {r}")
            if not (a < b and ta < tb):
                raise ValueError(f"Bad range: {r}")
            self.ranges.append(r)

        if not self.ranges:
            raise ValueError("At least one active range is required")

    def set_ranges(self, ranges):
        self._validate_and_set_ranges(ranges)

    def set_batch_size(self, batch_size):
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be > 0")
        self.batch_size = int(batch_size)

    def set_num_batches(self, num_batches):
        if int(num_batches) <= 0:
            raise ValueError("num_batches must be > 0")
        self.num_batches = int(num_batches)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        """
        Generate all batches of samples in one epoch.
        """
        # snapshot config at iterator start
        ranges = self.ranges.copy()
        n_ranges = len(ranges)
        batch_size = self.batch_size
        num_batches = self.num_batches
        full_rounds, remainder = divmod(batch_size, n_ranges)

        # yield batch samples
        for batch_idx in range(num_batches):
            batch = []
            for _ in range(full_rounds):
                for (a, b, ta, tb) in ranges:
                    batch.append((np.random.randint(a, b), ta, tb))

            # remainder items
            start = batch_idx % n_ranges
            for i in range(remainder):
                a, b, ta, tb = ranges[(start + i) % n_ranges]
                batch.append((np.random.randint(a, b), ta, tb))

            assert len(batch) == batch_size
            yield batch


class CustomSequentialSampler(Sampler):
    """
    Sequential sampler in range 0...max_frames-1, capped by len(data_source).
    """
    def __init__(self, data_source, max_frames: int):
        self.data_source = data_source
        self.set_max_frames(max_frames)

    def set_max_frames(self, max_frames: int) -> None:
        n = len(self.data_source)
        m = int(max_frames)
        self.max_frames = max(0, min(m, n))

    def __iter__(self):
        return iter(range(self.max_frames))

    def __len__(self) -> int:
        return self.max_frames