from typing import Any, Union, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    import jax.numpy as jnp

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, "jnp.ndarray", torch.Tensor]
