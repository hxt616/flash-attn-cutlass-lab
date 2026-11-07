import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(1234)


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch.device("cuda")
