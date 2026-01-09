"""
基础Agent类
为所有Agent提供通用功能和接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from Base.Base import Base
from ModuleFolders.TaskConfig.TaskConfig import TaskConfig
from ModuleFolders.Cache.CacheProject import CacheProject


class BaseAgent(Base, ABC):
    """所有Agent的基类"""
    
    def __init__(self, name: str, description: str, config: TaskConfig = None):
        super().__init__()
        self.name = name
        self.description = description
        self.config = config
        self.memory = {}  # Agent的本地记忆存储
        
    @abstractmethod
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行Agent的核心任务
        
        Args:
            input_data: 输入数据字典
            
        Returns:
            输出数据字典
        """
        pass
    
    def load_memory(self, key: str) -> Optional[Any]:
        """从内存中加载数据"""
        return self.memory.get(key)
    
    def save_memory(self, key: str, value: Any) -> None:
        """保存数据到内存"""
        self.memory[key] = value
        
    def clear_memory(self) -> None:
        """清空内存"""
        self.memory.clear()
        
    def log_agent_action(self, action: str, details: str = "") -> None:
        """记录Agent执行的动作"""
        self.info(f"[{self.name}] {action}")
        if details:
            self.debug(f"[{self.name}] 详情: {details}")
