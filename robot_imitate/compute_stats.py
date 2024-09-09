
from copy import deepcopy
from math import ceil

import datasets
import einops
import torch
import tqdm
import os
from pathlib import Path


CODEBASE_VERSION = "v1.4"
from typing import Dict

def load_previous_and_future_frames(
    item: dict[str, torch.Tensor],
    hf_dataset: datasets.Dataset,
    episode_data_index: dict[str, torch.Tensor],
    delta_timestamps: dict[str, list[float]],
    tolerance_s: float,
) -> dict[torch.Tensor]:
   
    # get indices of the frames associated to the episode, and their timestamps
    ep_id = item["episode_index"].item()
    ep_data_id_from = episode_data_index["from"][ep_id].item()
    ep_data_id_to = episode_data_index["to"][ep_id].item()
    
    ep_data_ids = torch.arange(ep_data_id_from, ep_data_id_to, 1) 
    ep_timestamps = hf_dataset.select_columns("timestamp")[ep_data_id_from:ep_data_id_to]["timestamp"]
    
    
    # list elements to torch, because stack require tuple of tensors
    ep_timestamps = [torch.tensor(item) for item in ep_timestamps]
    ep_timestamps = torch.stack(ep_timestamps)
    
    # we make the assumption that the timestamps are sorted
    ep_first_ts = ep_timestamps[0]
    ep_last_ts = ep_timestamps[-1]
    current_ts = item["timestamp"].item()
    
    for key in delta_timestamps:
        # get timestamps used as query to retrieve data of previous/future frames
        delta_ts = delta_timestamps[key]
        query_ts = current_ts + torch.tensor(delta_ts)
        

        # compute distances between each query timestamp and all timestamps of all the frames belonging to the episode
        dist = torch.cdist(query_ts[:, None], ep_timestamps[:, None], p=1)
        min_, argmin_ = dist.min(1)

        # TODO(rcadene): synchronize timestamps + interpolation if needed

        is_pad = min_ > tolerance_s

        # check violated query timestamps are all outside the episode range
        assert ((query_ts[is_pad] < ep_first_ts) | (ep_last_ts < query_ts[is_pad])).all(), (
            f"One or several timestamps unexpectedly violate the tolerance ({min_} > {tolerance_s=}) inside episode range."
            "This might be due to synchronization issues with timestamps during data collection."
        )

        # get dataset indices corresponding to frames to be loaded
        data_ids = ep_data_ids[argmin_]
    
        # load frames modality
        item[key] = hf_dataset.select_columns(key).select(data_ids)[key]
        column = hf_dataset.select_columns(key).select(data_ids)
        
        # get element from sorted list
        # function from lib take only one element in original implementation work
        items= []
        for i in column:
            items.append(i[key])
        item[key] = items.copy()

        # list elements to torch, because stack require tuple of tensors
        item[key] = [torch.tensor(item) for item in item[key]]
        item[key] = torch.stack(item[key])

        item[f"{key}_is_pad"] = is_pad
    
    return item




def calculate_episode_data_index_for_custom_dataset(hf_dataset: datasets.Dataset) -> Dict[str, torch.Tensor]:
    episode_data_index = {"from": [], "to": []}

    current_episode = None

    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index

    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx

    # Add the ending index for the last episode
    episode_data_index["to"].append(idx + 1)

    # Convert lists to tensors
    episode_data_index["from"] = torch.tensor(episode_data_index["from"])
    episode_data_index["to"] = torch.tensor(episode_data_index["to"])

    return episode_data_index


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        version: str,
        root: Path,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
    ):
        super().__init__()
        self.repo_id = repo_id
        self.version = version
        self.root = root
        self.split = split
        self.transform = transform
        self.delta_timestamps = delta_timestamps
      
        self.hf_dataset = load_hf_dataset(repo_id, version, root, split)
        if split == "train":
            self.episode_data_index = calculate_episode_data_index_for_custom_dataset(self.hf_dataset)
        else:
            self.episode_data_index = calculate_episode_data_index_for_custom_dataset(self.hf_dataset)

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return 10

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.
        Returns False if it only loads images from png files.
        """
        return None

    @property
    def features(self) -> datasets.Features:
        return self.hf_dataset.features

    @property
    def num_samples(self) -> int:
        """Number of samples/frames."""
        return len(self.hf_dataset)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return len(self.hf_dataset.unique("episode_index"))

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]

        if self.delta_timestamps is not None:
            item = load_previous_and_future_frames(
                item,
                self.hf_dataset,
                self.episode_data_index,
                self.delta_timestamps,
                self.tolerance_s,
            )

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository ID: '{self.repo_id}',\n"
            f"  Version: '{self.version}',\n"
            f"  Split: '{self.split}',\n"
            f"  Number of Samples: {self.num_samples},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.transform},\n"
            f")"
        )

    @classmethod
    def from_preloaded(
        cls,
        repo_id: str,
        version: str | None = CODEBASE_VERSION,
        root: Path | None = None,
        split: str = "train",
        transform: callable = None,
        delta_timestamps: dict[list[float]] | None = None,
        hf_dataset=None,
        episode_data_index=None
    ):
        # create an empty object of type LeRobotDataset
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.version = version
        obj.root = root
        obj.split = split
        obj.transform = transform
        obj.delta_timestamps = delta_timestamps
        obj.hf_dataset = hf_dataset
        obj.episode_data_index = episode_data_index
        return obj

def get_stats_einops_patterns(dataset: LeRobotDataset | datasets.Dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}
    print(dataset.features.items())
    print('________________________________________')
    
   
    for key, feats_type in dataset.features.items():
        assert batch[key].dtype != torch.float64
        print('key, feats_type: ', key, feats_type)

        if key == 'observation.image':
            print('_______________ is image _____________ ',  batch[key].shape)
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            print(c, h, w)
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {feats_type}, {batch[key].shape}")

    return stats_patterns


def compute_stats(
    dataset: LeRobotDataset | datasets.Dataset, batch_size=32, num_workers=16, max_num_samples=None
):
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float()
        std[key] = torch.tensor(0.0).float()
        max[key] = torch.tensor(-float("inf")).float()
        min[key] = torch.tensor(float("inf")).float()

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute mean, min, max")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = mean[key] + this_batch_size * (batch_mean - mean[key]) / running_item_count
            max[key] = torch.maximum(max[key], einops.reduce(batch[key], pattern, "max"))
            min[key] = torch.minimum(min[key], einops.reduce(batch[key], pattern, "min"))

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float()
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = std[key] + this_batch_size * (batch_std - std[key]) / running_item_count

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key],
            "std": std[key],
            "max": max[key],
            "min": min[key],
        }
    return stats

from PIL import Image as PILImage
from datasets import load_dataset
from torchvision import transforms

def hf_transform_to_torch(items_dict):
    from PIL import Image
    import io
    for key in items_dict:
        first_item = items_dict[key][0]
        if key == 'observation.image':
            first_item = items_dict[key][0]['bytes']
            first_item = Image.open(io.BytesIO(first_item))

           
        if isinstance(first_item, PILImage.Image):
            to_tensor = transforms.ToTensor()
            for item in items_dict[key]:
                item = item['bytes']
                items_dict[key] = to_tensor(Image.open(io.BytesIO(item)))
                items_dict[key] = items_dict[key].unsqueeze(0)
                
        elif key == 'observation.state' or key == 'action':
                import numpy as np
                for next_line  in items_dict[key]:
                    next_line = next_line[1:-1].split(',')
                    next_line = [float(item) for item in next_line]
                    next_line = np.array(next_line)

                    items_dict[key] = torch.tensor(next_line, dtype=torch.float32)
                    items_dict[key] = items_dict[key].unsqueeze(0)
        else:

            items_dict[key] = torch.tensor(items_dict[key]) #[torch.tensor(x) for x in items_dict[key]]
            
    return items_dict


def load_hf_dataset(repo_id, version, root, split) -> datasets.Dataset:
    
    if root is not None:
        print("roooooooooooooooot: ", root)
        hf_dataset = load_dataset('parquet', data_files = root, split = split)
    print()
    hf_dataset.set_transform(hf_transform_to_torch)

    return hf_dataset


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep="/"):
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict



def save_meta_data(info, stats, episode_data_index, meta_data_dir):
    # meta_data_dir.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
   
    # save stats
    stats_path = meta_data_dir + "stats.safetensors"
    save_file(flatten_dict(stats), stats_path)


def calculate_episode_data_index_for_custom_dataset(hf_dataset: datasets.Dataset) -> Dict[str, torch.Tensor]:
    episode_data_index = {"from": [], "to": []}

    current_episode = None

    if len(hf_dataset) == 0:
        episode_data_index = {
            "from": torch.tensor([]),
            "to": torch.tensor([]),
        }
        return episode_data_index

    for idx, episode_idx in enumerate(hf_dataset["episode_index"]):
        if episode_idx != current_episode:
            # We encountered a new episode, so we append its starting location to the "from" list
            episode_data_index["from"].append(idx)
            # If this is not the first episode, we append the ending location of the previous episode to the "to" list
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            # Let's keep track of the current episode index
            current_episode = episode_idx

    # Add the ending index for the last episode
    episode_data_index["to"].append(idx + 1)

    # Convert lists to tensors
    episode_data_index["from"] = torch.tensor(episode_data_index["from"])
    episode_data_index["to"] = torch.tensor(episode_data_index["to"])

    return episode_data_index

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute stats')
    parser.add_argument('--path', type=str, help='Data path')
    parsed_args = parser.parse_args()
    # print(parsed_args.path)

    
    repo_id = 'test'
    revision = 0 

    # root = 'robot_imitate/data/2024_08_21_20_10_53.parquet'
    root = parsed_args.path
    print(root)
    if root is None:
        print('[ERROR] Must specified path to data!')
        return

    hf_dataset = load_hf_dataset(repo_id, revision, root, 'train')

    episode_data_index = calculate_episode_data_index_for_custom_dataset(hf_dataset)
    print('==================================================')
    print(episode_data_index)
    print('==================================================')
    lerobot_dataset = LeRobotDataset.from_preloaded(
        repo_id=repo_id,
        version=revision,
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index
    )
    
    stats = compute_stats(lerobot_dataset, 2, 4)

   

    meta_data_dir = 'robot_imitate/metadata/'
    print('[INFO] Metadata path: ', meta_data_dir)

    if not os.path.exists(meta_data_dir):
        os.mkdir(meta_data_dir)

    save_meta_data(None, stats, episode_data_index, meta_data_dir)

    print(stats)


if __name__ == "__main__":
   main()