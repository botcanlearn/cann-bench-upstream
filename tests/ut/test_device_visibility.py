#!/usr/bin/python3
# coding=utf-8

"""
测试子进程设备可见性设置

关键语义（实机核对 Ascend910_9362 / CANN 9.0.0）：
- task.device_id 是**相对父进程可见集的逻辑索引**（0..card_count-1），
  因为 card_count 来自 torch.npu.device_count()，已受父进程可见性约束。
- 子进程 ASCEND_RT_VISIBLE_DEVICES 接受的也是父进程可见空间内的相对索引，
  而非全局物理卡号。故 _build_env_for_task 直接写 str(task.device_id)。
- 反例：父进程用 ASCEND_VISIBLE_DEVICES=12,13 收窄时，若子进程写
  ASCEND_RT_VISIBLE_DEVICES=12 会导致 device_count=0、set_device 失败。
"""

import unittest
from unittest.mock import Mock, patch

from kernel_eval.eval.process_pool import (
    ProcessPoolCoordinator,
    TaskUnit,
)
from kernel_eval.base.models import CaseSpec


def _make_task(device_id: int) -> TaskUnit:
    case = CaseSpec(
        case_id=f"case_{device_id}",
        operator="test_op",
        rel_path="test/path",
        case_num=1,
    )
    return TaskUnit(
        operator="test_op",
        rel_path="test/path",
        cases=[case],
        device_id=device_id,
    )


class TestChildDeviceVisibility(unittest.TestCase):
    """测试 _build_env_for_task 的设备可见性设置"""

    def _coordinator(self):
        return ProcessPoolCoordinator.__new__(ProcessPoolCoordinator)

    def test_sets_relative_index_as_visible_device(self):
        """子进程 ASCEND_RT_VISIBLE_DEVICES 应为相对索引 task.device_id"""
        coord = self._coordinator()

        env0 = coord._build_env_for_task({"PATH": "/usr/bin"}, _make_task(0))
        self.assertEqual(env0["ASCEND_RT_VISIBLE_DEVICES"], "0")

        env1 = coord._build_env_for_task({"PATH": "/usr/bin"}, _make_task(1))
        self.assertEqual(env1["ASCEND_RT_VISIBLE_DEVICES"], "1")

    def test_does_not_map_to_physical_number(self):
        """即使父进程用物理号收窄，子进程也不应写物理号（回归保护）

        父进程 ASCEND_VISIBLE_DEVICES=12,13（物理卡 12/13 → 逻辑 0/1）时，
        子进程仍应写相对索引 0/1，而不是 12/13。
        """
        coord = self._coordinator()
        base_env = {
            "ASCEND_VISIBLE_DEVICES": "12,13",
            "NPU_VISIBLE_DEVICES": "12,13",
        }

        env = coord._build_env_for_task(base_env, _make_task(0))
        # 必须是相对索引 "0"，不能是物理号 "12"
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "0")
        self.assertNotEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "12")

    def test_preserves_base_env(self):
        """base_env 的其他变量应保留，且不修改原 dict"""
        coord = self._coordinator()
        base_env = {"PATH": "/usr/bin", "FOO": "bar"}

        env = coord._build_env_for_task(base_env, _make_task(1))
        self.assertEqual(env["FOO"], "bar")
        self.assertEqual(env["PATH"], "/usr/bin")
        # 不污染原 dict
        self.assertNotIn("ASCEND_RT_VISIBLE_DEVICES", base_env)


if __name__ == '__main__':
    unittest.main()
