from types import SimpleNamespace

import torch

from kernel_eval.data.data_generator import DataGenerator
from kernel_eval.utils.param_builder import ParamBuilder


def _golden(x: torch.Tensor):
    return x + 1


def test_rank_zero_case_input_is_generated_and_passed_to_golden():
    input_shapes = [[]]
    tensors = DataGenerator().generate_input_tensors_from_case(
        input_shapes=input_shapes,
        dtypes=["float32"],
        value_ranges=[[2.0, 2.0]],
        seed=7,
    )

    case = SimpleNamespace(input_shapes=input_shapes, attrs={})
    params = ParamBuilder().build_call_params(_golden, case, tensors)
    result = _golden(**params)

    assert len(tensors) == 1
    assert tensors[0].shape == torch.Size([])
    assert tensors[0].item() == 2.0
    assert list(params) == ["x"]
    assert result.shape == torch.Size([])
    assert result.item() == 3.0


def test_rank_zero_case_input_keeps_integer_range():
    tensors = DataGenerator().generate_input_tensors_from_case(
        input_shapes=[[]],
        dtypes=["int32"],
        value_ranges=[[-3, -3]],
        seed=7,
    )

    assert len(tensors) == 1
    assert tensors[0].shape == torch.Size([])
    assert tensors[0].item() == -3


def test_optional_none_input_remains_omitted():
    tensors = DataGenerator().generate_input_tensors_from_case(
        input_shapes=[None],
        dtypes=["float32"],
        value_ranges=[],
        seed=7,
    )

    assert tensors == [None]
