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
- 通过 BENCH_DEVICE_VISIBILITY 备份还原物理可见集，再把 task.device_id
  映射为正确的物理 chip。
- 该映射**只**由 BENCH_DEVICE_VISIBILITY 触发。容器/k8s 设备插件施加的
  ASCEND_VISIBLE_DEVICES 已经把设备空间收窄并重编号，再映射一次会挑空
  （分到非 0 号物理卡时子进程 device_count()==0，aclInit 107001）。
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

    def test_container_narrowed_visibility_keeps_relative_index(self):
        """容器已用 ASCEND_VISIBLE_DEVICES 收窄时，子进程仍应拿相对索引

        k8s 设备插件把非 0 号物理卡分给 pod 时(ASCEND_VISIBLE_DEVICES=1),设备空间
        **已经**被收窄并重编号为 {0} —— 父进程 torch.npu.device_count()==1、用 npu:0
        正常。此时若把逻辑 0 映射回物理 "1" 写进 ASCEND_RT_VISIBLE_DEVICES,等于在
        已收窄的空间里再挑 1 号 → 挑空,子进程 device_count()==0:

            ChgUserDevIdToDeviceId failed because value 0 for parameter userDevId
            is invalid. Expected value: [0, 0)
            RuntimeError: ... NPU function error: aclInit, error code is 107001

        实测:分到 0 号卡时 ST 全绿,分到 1 号/5 号卡时 20/20 case 全灭(CI j_tnucrATi)。
        物理映射只在 BENCH_DEVICE_VISIBILITY 那条路径成立(见下一个用例),不能由
        ASCEND_VISIBLE_DEVICES 触发。
        """
        coord = self._coordinator()

        env = coord._build_env_for_task({"ASCEND_VISIBLE_DEVICES": "1"}, _make_task(0))
        self.assertEqual(env["ASCEND_RT_VISIBLE_DEVICES"], "0")

    def test_container_narrowed_multi_card_keeps_relative_index(self):
        """容器收窄到多张非 0 号卡时，逐 task 仍是相对索引

        682db28b 的 docstring 记的就是这个回归：父进程 ASCEND_VISIBLE_DEVICES=12,13
        时写出 ASCEND_RT_VISIBLE_DEVICES=12 会让子进程 device_count()==0。
        ec90f0c4 又把物理映射加了回来，此用例把它钉住。
        """
        coord = self._coordinator()
        base_env = {"ASCEND_VISIBLE_DEVICES": "12,13", "NPU_VISIBLE_DEVICES": "12,13"}

        self.assertEqual(
            coord._build_env_for_task(base_env, _make_task(0))["ASCEND_RT_VISIBLE_DEVICES"], "0")
        self.assertEqual(
            coord._build_env_for_task(base_env, _make_task(1))["ASCEND_RT_VISIBLE_DEVICES"], "1")

    def test_maps_logical_index_to_physical_chip(self):
        """benchsite-runner 留了物理集备份时，子进程应映射到正确的物理 chip

        benchsite-runner 的收窄是用 ASCEND_RT_VISIBLE_DEVICES 做的（逐进程、按物理
        号），而 torch_npu init 会原地覆盖 os.environ['ASCEND_RT_VISIBLE_DEVICES']，
        base_env 因此丢失原始物理可见集。BENCH_DEVICE_VISIBILITY 是它显式留的备份：
        有这个变量才说明"收窄发生在物理号层面"，此时才该把 task.device_id=0 映射
        为物理 "12"。容器层的 ASCEND_VISIBLE_DEVICES 不具备这个语义，见上一个用例。
        """
        coord = self._coordinator()
        base_env = {
            "BENCH_DEVICE_VISIBILITY": "12,13",
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
        """无 BENCH_DEVICE_VISIBILITY 时回退为相对索引"""
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
