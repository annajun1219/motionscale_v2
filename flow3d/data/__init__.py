from dataclasses import asdict

from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .casual_dataset import CasualDataset, CasualDatasetVideoView, CustomDataConfig, DavisDataConfig


DATASET_REGISTRY = {
    "davis": DavisDataConfig,
    "custom": CustomDataConfig,
}


def get_train_val_datasets(
    data_cfg: DavisDataConfig | CustomDataConfig, load_val: bool
) -> tuple[BaseDataset, Dataset | None]:
    train_video_view = None
    val_img_dataset = None
    val_kpt_dataset = None
    if isinstance(data_cfg, DavisDataConfig) or isinstance(data_cfg, CustomDataConfig):
        train_dataset = CasualDataset(**asdict(data_cfg))
        train_video_view = CasualDatasetVideoView(train_dataset)
    else:
        raise ValueError(f"Unknown data config: {data_cfg}")
    return train_dataset, train_video_view, val_img_dataset, val_kpt_dataset
