from datasets import load_dataset
from torch.utils.data.dataset import Dataset


class CalibrationDataset(Dataset):
    def __init__(self, dataset_name: str,
                 max_samples: int,
                 split: str = "train"):
        self.dataset_name = dataset_name
        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        print(f"[INFO] calibration dataset loaded. dataset name: {dataset_name}, sample numbers: {max_samples}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> str:
        text = self.dataset[idx]["text"]
        return text
