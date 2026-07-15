#!/usr/bin/python3
# coding=utf-8

"""
测试 TaskUnit 重试机制
"""

import unittest
from unittest.mock import Mock, patch
from dataclasses import dataclass

from kernel_eval.eval.process_pool import (
    ProcessConfig,
    TaskUnit,
    ProcessPoolCoordinator,
)
from kernel_eval.base.models import CaseSpec


class TestRetryMechanism(unittest.TestCase):
    """测试重试机制"""

    def test_process_config_with_retry(self):
        """测试 ProcessConfig 包含重试配置"""
        config = ProcessConfig(
            max_retries=2,
            retry_on_timeout=True,
            retry_on_oom=True,
            retry_on_failure=False,
        )
        self.assertEqual(config.max_retries, 2)
        self.assertTrue(config.retry_on_timeout)
        self.assertTrue(config.retry_on_oom)
        self.assertFalse(config.retry_on_failure)

    def test_task_unit_with_retry_fields(self):
        """测试 TaskUnit 包含重试相关字段"""
        case = CaseSpec(
            case_id="test_case_1",
            operator="test_op",
            rel_path="test/path",
            case_num=1,
        )
        task = TaskUnit(
            operator="test_op",
            rel_path="test/path",
            cases=[case],
            device_id=0,
            retry_count=1,
            excluded_devices={0, 1},
        )
        self.assertEqual(task.retry_count, 1)
        self.assertEqual(task.excluded_devices, {0, 1})

    def test_task_unit_default_excluded_devices(self):
        """测试 TaskUnit 的 excluded_devices 默认初始化"""
        case = CaseSpec(
            case_id="test_case_1",
            operator="test_op",
            rel_path="test/path",
            case_num=1,
        )
        task = TaskUnit(
            operator="test_op",
            rel_path="test/path",
            cases=[case],
            device_id=0,
        )
        self.assertIsNotNone(task.excluded_devices)
        self.assertEqual(task.excluded_devices, set())

    def test_select_device_for_retry(self):
        """测试设备选择逻辑"""
        with patch('kernel_eval.eval.process_pool.get_config'):
            coordinator = ProcessPoolCoordinator(device_id=0)

            # 场景1：有健康的未使用卡
            healthy_cards = [0, 1, 2, 3]
            excluded = {0}
            device = coordinator._select_device_for_retry(0, excluded, healthy_cards)
            self.assertIn(device, [1, 2, 3])
            self.assertNotIn(device, excluded)

            # 场景2：所有卡都被排除
            excluded = {0, 1, 2, 3}
            device = coordinator._select_device_for_retry(0, excluded, healthy_cards)
            self.assertEqual(device, 0)  # 回退到原设备

    def test_retry_count_increment(self):
        """测试重试计数递增"""
        case = CaseSpec(
            case_id="test_case_1",
            operator="test_op",
            rel_path="test/path",
            case_num=1,
        )

        # 初始任务
        task1 = TaskUnit(
            operator="test_op",
            rel_path="test/path",
            cases=[case],
            device_id=0,
            retry_count=0,
        )

        # 重试任务
        task2 = TaskUnit(
            operator="test_op",
            rel_path="test/path",
            cases=[case],
            device_id=1,
            retry_count=task1.retry_count + 1,
            excluded_devices=task1.excluded_devices | {task1.device_id},
        )

        self.assertEqual(task2.retry_count, 1)
        self.assertEqual(task2.excluded_devices, {0})

    def test_max_retries_limit(self):
        """测试最大重试次数限制"""
        config = ProcessConfig(max_retries=2)

        # retry_count < max_retries：应该重试
        self.assertTrue(0 < config.max_retries)
        self.assertTrue(1 < config.max_retries)

        # retry_count >= max_retries：不应该重试
        self.assertFalse(2 < config.max_retries)
        self.assertFalse(3 < config.max_retries)


if __name__ == '__main__':
    unittest.main()
