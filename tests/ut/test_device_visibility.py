#!/usr/bin/python3
# coding=utf-8

"""
测试子进程设备可见性设置

关键语义（实机核对 Ascend910_9362 / CANN 9.0.0）：
- task.device_id 是**相对父进程可见集的逻辑索引**（0..card_count-1），
  因为 card_count 来自 torch.npu.device_count()，已受父进程可见性约束。
- 但 benchsite-runner 的父进程 (eval-child) 中 torch_npu 初始化会
  原地改写 os.environ['ASCEND_RT_VISIBLE_DEVICES']，导致 base_env
  丢失了原始的物理可见集。此时 task.device_id=0 会被 CANN 解释为
  "物理 chip 0" 而非 "可见集内第 0 个设备"。
- 通过 BENCH_DEVICE_VISIBILITY（或 ASCEND_VISIBLE_DEVICES）备份还原
  物理可见集，再把 task.device_id 映射为正确的物理 chip。
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

    def test_maps_logical_index_to_physical_chip(self):
        """父进程用物理号收窄时，子进程应映射到正确的物理 chip

        父进程 ASCEND_VISIBLE_DEVICES=12,13（物理卡 12/13 → 逻辑 0/1）时，
        torch_npu init 会原地覆盖 os.environ['ASCEND_RT_VISIBLE_DEVICES']，
        导致 base_env 丢失原始可见集。通过 BENCH_DEVICE_VISIBILITY 或
        ASCEND_VISIBLE_DEVICES 备份还原，把 task.device_id=0 映射为物理 "12"。
        """
        coord = self._coordinator()
        base_env = {
            "ASCEND_VISIBLE_DEVICES": "12,13",
            "NPU_VISIBLE_DEVICES": "12,13",
        }

        env = coord._build_env_for_task(base_env, _make_task(0))
        # 逻辑 0 → 物理 chip 12
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "12")

        env1 = coord._build_env_for_task(base_env, _make_task(1))
        # 逻辑 1 → 物理 chip 13
        self.assertEqual(env1["ASCEND_RT_VISIBLE_DEVICES"], "13")

    def test_falls_back_to_relative_index_without_visibility(self):
        """无 BENCH_DEVICE_VISIBILITY 和 ASCEND_VISIBLE_DEVICES 时回退为相对索引"""
        coord = self._coordinator()
        base_env = {"PATH": "/usr/bin"}

        env = coord._build_env_for_task(base_env, _make_task(0))
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "0")

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
