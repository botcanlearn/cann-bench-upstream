#!/usr/bin/python3
# coding=utf-8

"""setup_info 的 docker 字段测试。

Why this file exists: 该字段此前硬编码为 "cake-ci / CANN 9.0.0"，镜像换代、CANN 升到
9.1.0 之后仍原样出现在每一份报告里 —— 同一份报告 environment.cann 写 9.1.0、docker 写
9.0.0，自相矛盾。没有任何测试兜着它，所以谁也没发现。这里把三条取值路径都钉住。
"""

import pytest

from kernel_eval.report import setup_info


@pytest.fixture
def no_container(monkeypatch):
    """默认按"不在容器里"跑，单个用例再按需打开。"""
    monkeypatch.setattr(setup_info, "_in_container", lambda: False)
    monkeypatch.delenv("CANN_BENCH_IMAGE", raising=False)


class TestDetectDocker:
    def test_image_env_wins(self, monkeypatch, no_container):
        """镜像注入了标识就用它，且带上实际探测到的 CANN 版本。"""
        monkeypatch.setenv("CANN_BENCH_IMAGE", "cann-bench-eval:1.0.0-ascend910b-aarch64-opsnone")
        monkeypatch.setattr(setup_info, "_get_cann_version", lambda: "9.1.0")
        assert setup_info._detect_docker() == (
            "cann-bench-eval:1.0.0-ascend910b-aarch64-opsnone / CANN 9.1.0"
        )

    def test_container_without_image_env_does_not_invent_a_name(self, monkeypatch):
        """在容器里但没注入标识 -> 只说 container，不编具体镜像名。"""
        monkeypatch.delenv("CANN_BENCH_IMAGE", raising=False)
        monkeypatch.setattr(setup_info, "_in_container", lambda: True)
        monkeypatch.setattr(setup_info, "_get_cann_version", lambda: "9.1.0")
        assert setup_info._detect_docker() == "container / CANN 9.1.0"

    def test_bare_metal_returns_none(self, monkeypatch, no_container):
        """不在容器里就如实返回 None，而不是给个假值。"""
        monkeypatch.setattr(setup_info, "_get_cann_version", lambda: "9.1.0")
        assert setup_info._detect_docker() is None

    def test_no_cann_detected_still_reports_the_image(self, monkeypatch, no_container):
        monkeypatch.setenv("CANN_BENCH_IMAGE", "some-image:tag")
        monkeypatch.setattr(setup_info, "_get_cann_version", lambda: None)
        assert setup_info._detect_docker() == "some-image:tag"

    def test_blank_image_env_is_treated_as_unset(self, monkeypatch):
        """Dockerfile 的 ARG 留空时 ENV 会是空串，不能当成镜像名用。"""
        monkeypatch.setenv("CANN_BENCH_IMAGE", "   ")
        monkeypatch.setattr(setup_info, "_in_container", lambda: True)
        monkeypatch.setattr(setup_info, "_get_cann_version", lambda: "9.1.0")
        assert setup_info._detect_docker() == "container / CANN 9.1.0"

    def test_never_reports_a_cann_version_that_contradicts_environment(self, monkeypatch):
        """docker 字段里的 CANN 与 environment.cann 必须同源 —— 这正是旧硬编码的病。"""
        monkeypatch.setenv("CANN_BENCH_IMAGE", "img:tag")
        monkeypatch.setattr(setup_info, "_get_cann_version", lambda: "9.1.0")
        info = setup_info.collect_setup_info()
        env = info["environment"]
        assert env["cann"] == "9.1.0"
        assert env["docker"] == "img:tag / CANN 9.1.0"
