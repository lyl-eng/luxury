"""
Griptape Agent包装器
将自定义Agent包装为Griptape的Agent格式
"""

import json
from typing import Dict, Any
from griptape.structures import Agent
from griptape.drivers import OpenAiChatPromptDriver
from openai import OpenAI

from Base.Base import Base
from ModuleFolders.TaskConfig.TaskConfig import TaskConfig
from .PreprocessingAgent import PreprocessingAgent
from .TerminologyEntityAgent import TerminologyEntityAgent
from .TranslationRefinementAgent import TranslationRefinementAgent


class GriptapeAgentWrapper(Base):
    """Griptape Agent包装器基类"""
    
    def __init__(self, config: TaskConfig, agent_name: str, agent_description: str):
        super().__init__()
        self.config = config
        self.agent_name = agent_name
        self.agent_description = agent_description
        
    def _create_griptape_agent(self, system_prompt: str = None) -> Agent:
        """创建Griptape Agent"""
        # 创建自定义LLM驱动（支持DeepSeek等OpenAI兼容的API）
        prompt_driver = self._create_prompt_driver()
        
        # 创建Agent
        # 注意：新版本的Griptape Agent可能不再接受name和description参数
        agent = Agent(
            prompt_driver=prompt_driver,
        )
        
        if system_prompt:
            agent.system_prompt = system_prompt
            
        return agent
    
    def _create_prompt_driver(self) -> OpenAiChatPromptDriver:
        """创建Prompt Driver，支持DeepSeek等自定义API"""
        if not self.config:
            raise ValueError("TaskConfig未初始化")
        
        # 获取平台配置
        platform_config = self.config.get_platform_configuration("translationReq")
        api_url = platform_config.get("api_url")
        api_key = platform_config.get("api_key", "")
        model_name = platform_config.get("model_name", "deepseek-chat")
        
        # 创建OpenAI客户端（兼容DeepSeek）
        client = OpenAI(
            api_key=api_key if api_key else "none",
            base_url=api_url
        )
        
        # 创建Prompt Driver
        # 注意：新版本的OpenAiChatPromptDriver可能不支持top_p参数
        prompt_driver = OpenAiChatPromptDriver(
            model=model_name,
            client=client,
            temperature=platform_config.get("temperature", 1.0),
        )
        
        return prompt_driver


class PreprocessingGriptapeAgent(GriptapeAgentWrapper):
    """预处理Agent的Griptape包装"""
    
    def __init__(self, config: TaskConfig):
        super().__init__(
            config=config,
            agent_name="PreprocessingAgent",
            agent_description="译前预处理与结构化拆解Agent"
        )
        self.preprocessing_agent = PreprocessingAgent(config)
    
    def create_agent(self) -> Agent:
        """创建Griptape Agent"""
        system_prompt = """你是一个文本预处理专家。你的任务是：
1. 分析文本结构
2. 识别文本的领域和风格
3. 将文本按逻辑单元切分

请以JSON格式返回结果。"""
        
        agent = self._create_griptape_agent(system_prompt)
        return agent
    
    def execute_task(self, cache_project_data: str) -> Dict[str, Any]:
        """执行预处理任务（作为Griptape Task的输入处理函数）"""
        # 将JSON字符串转换回对象
        import msgspec
        cache_project = msgspec.json.decode(cache_project_data.encode())
        
        # 调用原始Agent
        result = self.preprocessing_agent.execute({"cache_project": cache_project})
        
        # 将结果转换为JSON字符串（Griptape Task需要）
        return msgspec.json.encode(result).decode()


class TerminologyGriptapeAgent(GriptapeAgentWrapper):
    """术语识别Agent的Griptape包装"""
    
    def __init__(self, config: TaskConfig):
        super().__init__(
            config=config,
            agent_name="TerminologyEntityAgent",
            agent_description="术语识别与全局一致性保障Agent"
        )
        self.terminology_agent = TerminologyEntityAgent(config)
    
    def create_agent(self) -> Agent:
        """创建Griptape Agent"""
        system_prompt = """你是一个术语识别专家。你的任务是：
1. 识别命名实体（人名、地名、机构名等）
2. 识别领域术语
3. 识别文化负载词
4. 查证术语翻译
5. 构建术语库

请以JSON格式返回结果。"""
        
        agent = self._create_griptape_agent(system_prompt)
        return agent
    
    def execute_task(self, input_data: str) -> Dict[str, Any]:
        """执行术语识别任务"""
        import msgspec
        data = msgspec.json.decode(input_data.encode())
        
        result = self.terminology_agent.execute(data)
        
        return msgspec.json.encode(result).decode()


class TranslationGriptapeAgent(GriptapeAgentWrapper):
    """翻译Agent的Griptape包装"""
    
    def __init__(self, config: TaskConfig):
        super().__init__(
            config=config,
            agent_name="TranslationRefinementAgent",
            agent_description="翻译生成与迭代优化Agent"
        )
        self.translation_agent = TranslationRefinementAgent(config)
    
    def create_agent(self) -> Agent:
        """创建Griptape Agent"""
        system_prompt = """你是一个专业的翻译专家。你的任务是：
1. 多步骤引导翻译（理解—分解—转换—润色）
2. 生成多个翻译版本并融合
3. 回译验证和自我修正

请以JSON格式返回结果。"""
        
        agent = self._create_griptape_agent(system_prompt)
        return agent
    
    def execute_task(self, input_data: str) -> Dict[str, Any]:
        """执行翻译任务"""
        import msgspec
        data = msgspec.json.decode(input_data.encode())
        
        result = self.translation_agent.execute(data)
        
        return msgspec.json.encode(result).decode()

