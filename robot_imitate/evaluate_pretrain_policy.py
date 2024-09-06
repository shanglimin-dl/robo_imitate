import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
from pathlib import Path
import gymnasium as gym
import imageio
import numpy
import torch
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from dataclasses import dataclass, field
import torch
from torch import Tensor, nn



def create_stats_buffers(
    shapes: dict[str, list[int]],
    modes: dict[str, str],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}

    for key, mode in modes.items():
        assert mode in ["mean_std", "min_max"]

        shape = tuple(shapes[key])

        if "image" in key:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if mode == "mean_std":
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif mode == "min_max":
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )

        if stats is not None:
            # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
            # tensors anywhere (for example, when we use the same stats for normalization and
            # unnormalization). See the logic here
            # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
            if mode == "mean_std":
                buffer["mean"].data = stats[key]["mean"].clone()
                buffer["std"].data = stats[key]["std"].clone()
            elif mode == "min_max":
                buffer["min"].data = stats[key]["min"].clone()
                buffer["max"].data = stats[key]["max"].clone()

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        shapes: dict[str, list[int]],
        modes: dict[str, str],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, mode in self.modes.items():
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = (batch[key] - mean) / (std + 1e-8)
            elif mode == "min_max":
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                # normalize to [0,1]
                # print(batch)
                # breakpoint()
                batch[key] = (batch[key] - min) / (max - min)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(mode)
        return batch


class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        shapes: dict[str, list[int]],
        modes: dict[str, str],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.shapes = shapes
        self.modes = modes
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(shapes, modes, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        for key, mode in self.modes.items():
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if mode == "mean_std":
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif mode == "min_max":
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(mode)
        return batch

















@dataclass
class DiffusionConfig:
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - At least one key starting with "observation.image is required as an input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy (takes the
            current step and additional steps going back).
        horizon: Diffusion model action prediction size as detailed in `DiffusionPolicy.select_action`.
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
            See `DiffusionPolicy.select_action` for more details.
        input_shapes: A dictionary defining the shapes of the input data for the policy. The key represents
            the input data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "observation.image" refers to an input from a camera with dimensions [3, 96, 96],
            indicating it has three color channels and 96x96 resolution. Importantly, `input_shapes` doesn't
            include batch dimension or temporal dimension.
        output_shapes: A dictionary defining the shapes of the output data for the policy. The key represents
            the output data name, and the value is a list indicating the dimensions of the corresponding data.
            For example, "action" refers to an output shape of [14], indicating 14-dimensional actions.
            Importantly, `output_shapes` doesn't include batch dimension or temporal dimension.
        input_normalization_modes: A dictionary with key representing the modality (e.g. "observation.state"),
            and the value specifies the normalization mode to apply. The two available modes are "mean_std"
            which subtracts the mean and divides by the standard deviation and "min_max" which rescale in a
            [-1, 1] range.
        output_normalization_modes: Similar dictionary as `normalize_input_modes`, but to unnormalize to the
            original scale. Note that this is also used for normalizing the training targets.
        vision_backbone: Name of the torchvision resnet backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        crop_is_random: Whether the crop should be random at training time (it's always a center crop in eval
            mode).
        pretrained_backbone_weights: Pretrained weights from torchvision to initalize the backbone.
            `None` means no pretrained weights.
        use_group_norm: Whether to replace batch normalization with group normalization in the backbone.
            The group sizes are set to be about 16 (to be precise, feature_dim // 16).
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax.
        down_dims: Feature dimension for each stage of temporal downsampling in the diffusion modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the diffusion modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
        diffusion_step_embed_dim: The Unet is conditioned on the diffusion timestep via a small non-linear
            network. This is the output dimension of that network, i.e., the embedding dimension.
        use_film_scale_modulation: FiLM (https://arxiv.org/abs/1709.07871) is used for the Unet conditioning.
            Bias modulation is used be default, while this parameter indicates whether to also use scale
            modulation.
        noise_scheduler_type: Name of the noise scheduler to use. Supported options: ["DDPM", "DDIM"].
        num_train_timesteps: Number of diffusion steps for the forward diffusion schedule.
        beta_schedule: Name of the diffusion beta schedule as per DDPMScheduler from Hugging Face diffusers.
        beta_start: Beta value for the first forward-diffusion step.
        beta_end: Beta value for the last forward-diffusion step.
        prediction_type: The type of prediction that the diffusion modeling Unet makes. Choose from "epsilon"
            or "sample". These have equivalent outcomes from a latent variable modeling perspective, but
            "epsilon" has been shown to work better in many deep neural network settings.
        clip_sample: Whether to clip the sample to [-`clip_sample_range`, +`clip_sample_range`] for each
            denoising step at inference time. WARNING: you will need to make sure your action-space is
            normalized to fit within this range.
        clip_sample_range: The magnitude of the clipping range as described above.
        num_inference_steps: Number of reverse diffusion steps to use at inference time (steps are evenly
            spaced). If not provided, this defaults to be the same as `num_train_timesteps`.
        do_mask_loss_for_padding: Whether to mask the loss when there are copy-padded actions. See
            `LeRobotDataset` and `load_previous_and_future_frames` for mor information. Note, this defaults
            to False as the original Diffusion Policy implementation does the same.
    """

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.image": [3, 96, 96],
            "observation.state": [4],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [4],
        }
    )

    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.image": "mean_std",
            "observation.state": "min_max",
        }
    )
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {"action": "min_max"})

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18" #"efficientnet_b0"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    # Noise scheduler.
    noise_scheduler_type: str = "DDIM"
    num_train_timesteps: int = 50
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0


    ITER = 1

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        # if not self.vision_backbone.startswith("resnet"):
        #     raise ValueError(
        #         f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
        #     )
        image_keys = {k for k in self.input_shapes if k.startswith("observation.image")}
        if self.crop_shape is not None:
            for image_key in image_keys:
                if (
                    self.crop_shape[0] > self.input_shapes[image_key][1]
                    or self.crop_shape[1] > self.input_shapes[image_key][2]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within `input_shapes[{image_key}]`. Got {self.crop_shape} "
                        f"for `crop_shape` and {self.input_shapes[image_key]} for "
                        "`input_shapes[{image_key}]`."
                    )
        # Check that all input images have the same shape.
        first_image_key = next(iter(image_keys))
        for image_key in image_keys:
            if self.input_shapes[image_key] != self.input_shapes[first_image_key]:
                raise ValueError(
                    f"`input_shapes[{image_key}]` does not match `input_shapes[{first_image_key}]`, but we "
                    "expect all image shapes to match."
                )
        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )





import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn
# from lerobot.common.policies.normalize import Normalize, Unnormalize
# from lerobot.common.policies.utils import (
#     get_device_from_parameters,
#     get_dtype_from_parameters,
#     populate_queues,
# )


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype

"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from huggingface_hub import PyTorchModelHubMixin
from torch import Tensor, nn

# from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)


class DiffusionPolicy(nn.Module, PyTorchModelHubMixin):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__()
        if config is None:
            config = DiffusionConfig()
        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.images": deque(maxlen=self.config.n_obs_steps),
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... |n-o+1+h|
            |observation is used | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps < horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            print('***************************************')
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action



    @torch.no_grad
    def select_action_new_from_hub(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... |n-o+1+h|
            |observation is used | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps < horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        return {"loss": loss}


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        self.rgb_encoder = DiffusionRgbEncoder(config)
        num_images = len([k for k in config.input_shapes if k.startswith("observation.image")])
        self.unet = DiffusionConditionalUnet1d(
            config,
            global_cond_dim=(
                config.input_shapes["observation.state"][0] + self.rgb_encoder.feature_dim * num_images
            )
            * config.n_obs_steps,
        )

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.output_shapes["action"][0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        # Extract image feature (first combine batch, sequence, and camera index dims).
        img_features = self.rgb_encoder(
            einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
        )
        # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the feature
        # dim (effectively concatenating the camera features).
        img_features = einops.rearrange(
            img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )
        # Concatenate state and image features then flatten to (B, global_cond_dim).
        return torch.cat([batch["observation.state"], img_features], dim=-1).flatten(start_dim=1)

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)
            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "observation.images", "action", "action_is_pad"})
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encoder an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.input_shapes` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.input_shapes`.
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        # Note: we have a check in the config class to make sure all images have the same shape.
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
        )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out






def call_ros2_service(activate_controllers, deactivate_controllers):
    service_name = '/controller_manager/switch_controller'
    service_type = 'controller_manager_msgs/srv/SwitchController'
    strictness = '2'
    activate_asap = 'true'

    command = f'ros2 service call {service_name} {service_type} "{{activate_controllers: [\"{activate_controllers}\"], deactivate_controllers: [\"{deactivate_controllers}\"], strictness: {strictness}, activate_asap: {activate_asap}}}"'
    try:
        result = subprocess.run(command, shell=True,
                                check=True, capture_output=True, text=True)
        match = re.search(r'response:\n(.*)', result.stdout, re.DOTALL)
        print(f"{activate_controllers}:", match.group(1).strip())
    except subprocess.CalledProcessError as e:
        print(f"Error calling ROS 2 service: {e}")


from http.server import BaseHTTPRequestHandler, HTTPServer
import os

class MJPEGStreamHandler(BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server, frame_queue):
        self.frame_queue = frame_queue
        self.streaming_started = False  # Flag to track streaming start

        if not os.path.exists('images'):
            os.makedirs('images')
        super().__init__(request, client_address, server)

    def do_GET(self):
        if self.path == '/':
            if not self.streaming_started:
                self.start_streaming()
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()

            while True:
                try:
                    frame = self.frame_queue.get()

                    _, img_encoded = cv2.imencode('.jpg', frame)
                    frame_bytes = img_encoded.tobytes()

                    # Send the frame to the browser
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', len(frame_bytes))
                    self.end_headers()
                    self.wfile.write(frame_bytes)
                    self.wfile.write(b'\r\n--frame\r\n')
                except Exception as e:
                    print("Exception: ", e)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def start_streaming(self):
        self.streaming_started = True

    def start_streaming(self):
        self.streaming_started = True


def start_stream_server(frame_queue):
    ip_address = '0.0.0.0'
    server = HTTPServer((ip_address, 8080), lambda *args, **kwargs: MJPEGStreamHandler(*args, **kwargs, frame_queue=frame_queue))
    print(f'Starting streaming server on http://'+ ip_address +':8080/')
    server.serve_forever()



def are_elements_same(queue):
    first_element = queue[0]
    return all(np.array_equal(first_element, element) for element in queue)


from geometry_msgs.msg import PoseStamped, Twist, Pose
from std_msgs.msg import Float64MultiArray, Header
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Int32
import ros2_numpy as rnp
from collections.abc import Sequence
import tf2_ros

from enum import Enum
import subprocess
import re
import time
import subprocess
import re


MOTION_STEP = 0.01
from collections import deque
import transforms3d as t3d
from ros2_numpy_tf import numpy2ros, ros2numpy

class OperationState(Enum):
    IDLE = 0
    INFERENCE = 1
    GO_CLOSE = 2
    CLOSE_GRIPPER = 3
    PICK_UP = 4
    OPEN_GRIPPER = 5
    END = 6


def check_pose_stamped_values(pose_stamped_msg):
    position = pose_stamped_msg.pose.position
    orientation = pose_stamped_msg.pose.orientation
    is_position_zero = position.x == 0.0 and position.y == 0.0 and position.z == 0.0
    
    is_orientation_zero_except_w = orientation.x == 0.0 and orientation.y == 0.0 and orientation.z == 0.0 and orientation.w == 1.0
    
    return is_orientation_zero_except_w


def is_end_of_episode(action):
    arr = np.array([action[0], action[1], action[2], action[3], action[4], action[5], action[6]])
    close_to_zero = np.isclose(arr, 0, atol=0.05)
    
    return np.all(close_to_zero)


def plot_action_trajectory(sim, positions1, positions2=None):
    import matplotlib.pyplot as plt
    from datetime import datetime
    import os

    positions1 = np.array(positions1)
    norm1 = plt.Normalize(positions1[:, 2].min(), positions1[:, 2].max())
    print(norm1)
    colors1 = plt.cm.viridis(norm1(positions1[:, 2]))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc1 = ax.scatter(positions1[:, 0], positions1[:, 1], positions1[:, 2], c=colors1, marker='o', s=50, alpha=0.8, edgecolor='k', linewidth=0.5, label='Trajektorija observacije')

    if positions2 is not None:
        positions2 = np.array(positions2)
        norm2 = plt.Normalize(positions2[:, 2].min(), positions2[:, 2].max())
        colors2 = plt.cm.plasma(norm2(positions2[:, 2]))
        print(norm2)

        sc2 = ax.scatter(positions2[:, 0], positions2[:, 1], positions2[:, 2], c=colors2, marker='^', s=50, alpha=0.8, edgecolor='k', linewidth=0.5, label='Trajektorija akcije')

    ax.set_xlabel('X osa', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y osa', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z osa', fontsize=12, fontweight='bold')

    ax.set_title('Razlika između akcije i observacije', fontsize=14, fontweight='bold')

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))  
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # ax.view_init(elev=20, azim=-30)
    ax.legend(loc='best')

    now = datetime.now()
    real_path = 'action_trajectoris/real/'
    sim_path = 'action_trajectoris/sim/'

    if not os.path.exists('action_trajectoris'):
        os.mkdir('action_trajectoris')
    if not os.path.exists(real_path):
        os.mkdir(real_path)
    if not os.path.exists(sim_path):
        os.mkdir(sim_path)

    name = real_path + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'
    if sim:
        name = sim_path + now.strftime("%Y_%m_%d_%H_%M_%S") + '.png'

    plt.savefig(name, bbox_inches='tight', dpi=300)

    # plt.show()

from queue import Queue
import threading

class CmdVelPublisher(Node):

    def __init__(self, sim):
        super().__init__('cmd_vel_publisher')
        self.get_logger().info("CmdVelPublisher node started")

        self.subscription = self.create_subscription(
            Image,
            '/rgb',
            self.listener_callback,
            1)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.1, self.publish_cmd_vel)

        self.publisher_speed_limiter = self.create_publisher(
            PoseStamped, '/target_frame_raw', 1)

        self.publisher_gripper = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 1)

        self.current_pose_subscriber = self.create_subscription(
            PoseStamped,
            '/current_pose',
            self.current_pose_callback,
            1)
        self.current_pose_subscriber
        

        self.x_translation = 0.0
        self.y_translation = 0.0
        self.z_translation = 0.0

        self.output_directory = Path("/home/marija/exp-lerobot/examples/outputs/eval/example_pusht_diffusion")
        # self.output_directory.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda")
        self.pretrained_policy_path = Path("/home/marija/exp-lerobot/examples/outputs/train/example_pusht_diffusion")

        self.policy = DiffusionPolicy.from_pretrained(self.pretrained_policy_path)
        print(self.policy)
        self.policy.eval()
        self.policy.to(self.device)
        
        self.policy.reset()
        self.current_pose_relativ = PoseStamped()
        self.current_pose = PoseStamped()
        self.step = 0
        
        self.image  = None
        self.action = None

        self.current_x =  0.0
        self.current_y =  0.0
        self.current_z =  0.0
        # 60 steps for sim 90 for real env
        self.max_episode_steps = 90
        self.frames = []
        self.twist_msg = Twist()

        self.counter = 0
        self.queue = deque(maxlen=3)
        self.sim = sim
        self.gripper_state = 0

        self.observation_pose = None
        self.observation_current_pose = PoseStamped()
        if self.sim:
            # on init open griper and move screwdriver on random position
            self.gripper_msg = Float64MultiArray()
            self.gripper_msg.data = [0.0]
            self.publisher_gripper.publish(self.gripper_msg)

            self.publisher_respawn = self.create_publisher(Twist, '/respawn', 1)
            msg_arr = Twist()
            self.publisher_respawn.publish(msg_arr)

        self.timer = time.time()

        self.publisher_joint_init = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 1)
        self.joint_state = JointTrajectory()
        self.joint_names = ['joint1', 'joint2',
                            'joint3', 'joint4', 'joint5', 'joint6']

        point = JointTrajectoryPoint()
        point.positions = [0.00148, 0.06095, 1.164, -0.00033, 1.122, -0.00093]
        point.time_from_start.sec = 3
        point.time_from_start.nanosec = 0

        self.joint_state.points = [point]
        self.joint_state.joint_names = self.joint_names

        if not self.sim:
            self.switcher_publisher = self.create_publisher(Int32, '/gripper_switcher', 1)
            msg = Int32()
            msg.data = 1
            self.switcher_publisher.publish(msg)

            time.sleep(1)
            msg = Int32()
            msg.data = 2
            self.switcher_publisher.publish(msg)
        # move arm to init pose
        call_ros2_service('joint_trajectory_controller',
                              'cartesian_motion_controller')
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.publisher_joint_init.publish(self.joint_state)

        time.sleep(3)
        call_ros2_service('cartesian_motion_controller', 'joint_trajectory_controller')
        
        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer, self)

        self.state = OperationState.IDLE
        self.observation_msg = PoseStamped()
        self.start_pose = None


        self.plotting_observations = []
        self.plotting_actions = []

        # self.frame_queue = Queue(maxsize=2)
        # self.server_thread = threading.Thread(target=start_stream_server, args=(self.frame_queue, ))
        # self.server_thread.daemon = True
        # self.server_thread.start()

        
    
    def get_transform(self, target_frame, source_frame):
        try:
            transform = self.tfBuffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time())
            return transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().error(f"Failed to get transform: {e}")
            return None
    
    def current_pose_callback(self, msg):
        self.observation_msg = msg
        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        z_pos = msg.pose.position.z

        x_ori = msg.pose.orientation.x
        y_ori = msg.pose.orientation.y
        z_ori = msg.pose.orientation.z
        w_ori = msg.pose.orientation.w

        quat = [w_ori, x_ori, y_ori, z_ori]
        euler_angle = t3d.euler.quat2euler(quat)

        # self.observation_pose = [x_pos, y_pos, z_pos, x_ori, y_ori, z_ori, w_ori]
        self.observation_pose = [x_pos, y_pos, z_pos]
        self.observation_current_pose = msg

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv_image = cv2.resize(cv_image, (96, 96))
        self.image = cv_image
        self.frames.append(self.image)
        image_name = "/home/marija/exp-lerobot/examples/outputs/rollout_images/" + str(self.counter) + ".jpg"
        cv2.imwrite(image_name, self.image)
        self.counter +=1
          
    def publish_cmd_vel(self):

        if self.observation_pose is None:
            return
        
        if self.image is None:
            return

        base_gripper_tf = self.get_transform('link_base', 'gripper_base_link')
        if base_gripper_tf is None:
            return
        
        self.get_logger().info(f'========================={self.state}=========================')
        # if self.state == OperationState.INFERENCE:
        #     # self.get_logger().info(f'{self.state}')
        #     base_gripper = ros2numpy(base_gripper_tf.transform)
        #     # self.get_logger().info(f'{self.current_pose_relativ.pose}')
        #     # gripper_target = ros2numpy(self.current_pose_relativ.pose)

        #     # target_transform =  base_gripper @ gripper_target
        #     current_pose_target = ros2numpy(self.current_pose_relativ.pose)
        #     # target_transform = ros2numpy(self.observation_current_pose.pose) @ current_pose_target

        #     target_rotation = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
        #     target_rotation = target_rotation[:3, :3]

        #     target_transform = ros2numpy(self.start_pose.pose) @ current_pose_target

        #     self.plotting_actions.append(target_transform[:3, 3])
        #     self.plotting_observations.append(self.observation_pose[:3])

        #     target_transform[:3, :3] = target_rotation

        #     pose = numpy2ros(target_transform, Pose)
        #     self.start_pose.pose = pose
        #     self.current_pose.pose = pose

        self.current_pose.header.stamp = self.get_clock().now().to_msg()
        self.current_pose.header.frame_id = 'link_base'

    
        if self.state == OperationState.IDLE:
            
            self.start_pose = self.observation_current_pose
            if self.start_pose  is not None:
                self.state = OperationState.INFERENCE
                self.timer = time.time()

        elif self.state == OperationState.INFERENCE:
            # Prepare observation for the policy running in Pytorch
            state = torch.from_numpy(np.array(self.observation_pose))
            state = state.to(torch.float32).to(self.device, non_blocking=True).unsqueeze(0)
            image = torch.from_numpy(self.image).to(torch.float32).permute(2, 0, 1).to(self.device, non_blocking=True).unsqueeze(0) / 255

            # Create the policy input dictionary
            observation = {
                "observation.state": state,
                "observation.image": image,
            }

            # Predict the next action with respect to the current observation
            tick = time.time()
            with torch.inference_mode():
                self.action = self.policy.select_action(observation).squeeze(0).cpu().numpy()
            tock = time.time()
            if self.action is None:
                return
            
            self.get_logger().info(f"Step {self.step}, Action: {self.action}")

            self.current_pose_relativ.header.stamp = self.get_clock().now().to_msg()
            self.current_pose_relativ.header.frame_id = 'gripper_link_base'
            self.current_pose_relativ.pose.position.x = (float(self.action[0]) / 1.5)
            self.current_pose_relativ.pose.position.y = (float(self.action[1]) / 1.0) 
            self.current_pose_relativ.pose.position.z = (float(self.action[2]) / 2.5)

            # quat = t3d.euler.euler2quat(0.0, 0.0, self.action[5])

            self.current_pose_relativ.pose.orientation.x = 0.0 #float(quat[1]) #0.0
            self.current_pose_relativ.pose.orientation.y = 0.0 #float(quat[2]) #0.0
            self.current_pose_relativ.pose.orientation.z = 0.0 #float(quat[3]) #0.0
            self.current_pose_relativ.pose.orientation.w = 1.0 #float(quat[0]) #1.0


            current_pose_target = ros2numpy(self.current_pose_relativ.pose)
            target_rotation = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
            target_rotation = target_rotation[:3, :3]

            target_transform = ros2numpy(self.observation_current_pose.pose) @ current_pose_target
            # print(target_transform)

            self.plotting_actions.append(target_transform[:3, 3])
            self.plotting_observations.append(self.observation_pose[:3])

            target_transform[:3, :3] = target_rotation

            pose = numpy2ros(target_transform, Pose)
            self.start_pose.pose = pose
            self.current_pose.pose = pose

            if self.observation_msg.pose.position.z < 0.12:
                self.state = OperationState.OPEN_GRIPPER
                self.timer = time.time()
                # time.sleep(15)
                self.get_logger().info('___________________________________ END ___________________________________')



            # print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^', self.current_pose)
            self.publisher_speed_limiter.publish(self.current_pose)
            # time.sleep(0.05)

            self.step += 1

            if self.step > self.max_episode_steps:
                plot_action_trajectory(self.sim, self.plotting_observations, self.plotting_actions)
                self.state = OperationState.OPEN_GRIPPER
                self.timer = time.time()
                # rclpy.shutdown()
                # exit(0)

        elif self.state == OperationState.OPEN_GRIPPER:
            # open later because make noise
            if not self.sim:
                msg = Int32()
                msg.data = 1
                self.switcher_publisher.publish(msg)

            if time.time() - self.timer > 1.5:
                self.state = OperationState.GO_CLOSE
                self.timer = time.time()

        elif self.state == OperationState.GO_CLOSE:
                self.observation_msg.pose.position.z = 0.085
                self.publisher_speed_limiter.publish(self.observation_msg)
                if time.time() - self.timer > 2.5:
                    self.state = OperationState.CLOSE_GRIPPER
                    self.timer  =time.time()
        
        elif self.state == OperationState.CLOSE_GRIPPER:
            if self.sim:
                self.gripper_msg.data = [-0.01]
                self.publisher_gripper.publish(self.gripper_msg)
            else:
                msg = Int32()
                msg.data = 0
                self.switcher_publisher.publish(msg)

            if time.time() -  self.timer > 1.5:
                self.state = OperationState.PICK_UP
                self.timer = time.time()
        elif self.state == OperationState.PICK_UP:
                self.observation_msg.pose.position.z = 0.29
                self.publisher_speed_limiter.publish(self.observation_msg)
                if time.time() - self.timer > 1.5:
                    self.state = OperationState.END
                    self.timer = time.time()

        elif self.state == OperationState.END:
            # self.gripper_msg.data = [0.0]
            # self.publisher_gripper.publish(self.gripper_msg)

            if time.time() - self.timer > 2.5:
                if not self.sim:
                    # shutdown gripper
                    msg = Int32()
                    msg.data = 2
                    self.switcher_publisher.publish(msg)

                self.save_video()

                self.get_logger().info(f"Finished publishing. Shutting down node...")
                rclpy.shutdown()
                exit(0)

       
        if check_pose_stamped_values(self.current_pose_relativ):
            return
        # if self.state == OperationState.PICK_UP or self.state == OperationState.GO_CLOSE:
        #     self.publisher_speed_limiter.publish(self.observation_msg)
        #     # time.sleep(3)
            # self.get_logger().info(f'{self.current_pose}')

    def save_video(self):
        # video_path = self.output_directory / "rollout_new_long_hz_model.mp4"
        # imageio.mimsave(str(video_path), numpy.stack(self.frames), fps=30)
        # self.get_logger().info(f"Video saved at {video_path}")
        pass

def main(args=None):

    import argparse
    parser = argparse.ArgumentParser(description='CmdVelPublisher')
    parser.add_argument('--sim', action='store_true', help='Use simulation')
    parsed_args = parser.parse_args()

    rclpy.init(args=args)
    cmd_vel_publisher = CmdVelPublisher(parsed_args.sim)
    rclpy.spin(cmd_vel_publisher)
    cmd_vel_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
