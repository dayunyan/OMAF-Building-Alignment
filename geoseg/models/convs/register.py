import inspect
import torch
import torch.nn as nn
from typing import Dict, Type, Optional, Union


class LayerRegistry:
    """层注册器基类，统一管理模块注册与实例化"""

    def __init__(self):
        self._registry: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str, layer_cls: Type[nn.Module]):
        """注册新模块"""
        if name in self._registry:
            raise ValueError(f"模块名称 {name} 已存在，请勿重复注册")
        self._registry[name] = layer_cls

    def get(self, name: str) -> Type[nn.Module]:
        """获取已注册的模块类"""
        if name not in self._registry:
            raise KeyError(
                f"未注册的模块 {name}，可选模块：{list(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, cfg: Union[str, Dict], **kwargs) -> nn.Module:
        """根据配置构建模块实例
        Args:
            cfg: 模块配置，支持字符串（仅指定类型）或字典（含类型和参数）
            kwargs: 模块通用参数（会覆盖cfg中的同名参数）
        """
        # 解析配置
        if isinstance(cfg, str):
            cfg = {"type": cfg}
        assert isinstance(cfg, dict) and "type" in cfg, "cfg必须包含'type'字段"

        # 获取模块类
        layer_cls = self.get(cfg["type"])
        # 提取模块所需参数（排除self）
        param_signature = inspect.signature(layer_cls.__init__)
        required_params = list(param_signature.parameters.keys())[1:]  # 跳过self

        # 合并配置参数和外部参数（外部参数优先级更高）
        layer_params = {}
        # 1. 从cfg中提取参数
        for k, v in cfg.items():
            if k != "type" and k in required_params:
                layer_params[k] = v
        # 2. 从kwargs中提取参数（覆盖cfg）
        for k, v in kwargs.items():
            if k in required_params:
                layer_params[k] = v

        # # 检查必填参数是否齐全
        # missing_params = [p for p in required_params if p not in layer_params]
        # if missing_params:
        #     raise ValueError(f"模块 {cfg['type']} 缺少必填参数：{missing_params}")

        # 实例化模块
        return layer_cls(**layer_params)
