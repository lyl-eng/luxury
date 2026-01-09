"""
多智能体工作流管理器
基于Griptape框架实现工作流编排
"""

import json
import msgspec
from typing import Dict, Any, Optional, List
from Base.Base import Base
from .PreprocessingAgent import PreprocessingAgent
from .TerminologyEntityAgent import TerminologyEntityAgent
from .TranslationRefinementAgent import TranslationRefinementAgent
from .PlanningAgent import PlanningAgent
from .HumanCollaborationNode import HumanCollaborationNode
from .GriptapeTools import PreprocessingTool, TerminologyTool, TranslationTool
from ModuleFolders.Cache.CacheProject import CacheProject
from ModuleFolders.TaskConfig.TaskConfig import TaskConfig

# Griptape框架导入
from griptape.structures import Workflow, Agent
from griptape.tasks import ToolkitTask, PromptTask
from griptape.drivers import OpenAiChatPromptDriver
from openai import OpenAI


class WorkflowManager(Base):
    """
    多智能体工作流管理器
    使用Griptape框架编排各个Agent的工作流程
    """
    
    def __init__(self, config: TaskConfig = None):
        super().__init__()
        self.config = config
        
        # 初始化各个Agent（用于直接调用，作为备用）
        self.planning_agent = PlanningAgent(config)
        self.preprocessing_agent = PreprocessingAgent(config)
        self.terminology_agent = TerminologyEntityAgent(config)
        self.translation_agent = TranslationRefinementAgent(config)
        self.human_collab_node = HumanCollaborationNode()
        
        # 初始化Griptape工作流
        self.griptape_workflow = None
        self._init_griptape_workflow()
    
    def _create_prompt_driver(self) -> OpenAiChatPromptDriver:
        """创建Prompt Driver（用于ToolkitTask）"""
        if not self.config:
            raise ValueError("TaskConfig未初始化")
        
        # 获取平台配置
        platform_config = self.config.get_platform_configuration("translationReq")
        api_url = platform_config.get("api_url")
        api_key = platform_config.get("api_key", "")
        model_name = platform_config.get("model_name", "deepseek-chat")
        
        # 创建OpenAI客户端（兼容DeepSeek等OpenAI兼容的API）
        client = OpenAI(
            api_key=api_key if api_key else "none",
            base_url=api_url
        )
        
        # 创建Prompt Driver
        prompt_driver = OpenAiChatPromptDriver(
            model=model_name,
            client=client,
            temperature=platform_config.get("temperature", 1.0),
        )
        
        return prompt_driver
    
    def _create_griptape_agent(self, name: str, description: str, system_prompt: str = None) -> Agent:
        """创建Griptape Agent，支持DeepSeek等自定义LLM"""
        if not self.config:
            raise ValueError("TaskConfig未初始化")
        
        # 获取平台配置
        platform_config = self.config.get_platform_configuration("translationReq")
        api_url = platform_config.get("api_url")
        api_key = platform_config.get("api_key", "")
        model_name = platform_config.get("model_name", "deepseek-chat")
        
        # 创建OpenAI客户端（兼容DeepSeek等OpenAI兼容的API）
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
        
        # 创建Agent
        # 注意：新版本的Griptape Agent可能不再接受name和description参数
        # 只传递必要的参数：prompt_driver
        agent = Agent(
            prompt_driver=prompt_driver,
        )
        
        # 设置system_prompt（如果提供）
        if system_prompt:
            agent.system_prompt = system_prompt
        
        return agent
    
    def _init_griptape_workflow(self):
        """
        初始化Griptape工作流
        使用Griptape的Workflow和Task来编排多智能体工作流
        
        架构说明：
        - 使用共享状态（workflow_state）在Tools之间传递大对象（cache_project等）
        - Griptape只负责控制流程和传递小的元数据
        - Tools内部调用现有的Agent执行业务逻辑
        """
        try:
            # 初始化共享工作流状态
            # 用于在Tool之间传递大对象，避免通过LLM函数调用参数传递超大JSON
            self._workflow_state = {
                "cache_project": None,  # 核心数据对象
                "metadata": {},  # 预处理元数据
                "terminology_database": {},  # 术语库
                "memory_storage": {},  # Memory存储
                "translation_results": [],  # 翻译结果
                "did_translate": False,  # 是否完成翻译标志
            }
            
            # 创建Griptape Tools（注入共享state）
            preprocessing_tool = PreprocessingTool(self.config, self._workflow_state)
            terminology_tool = TerminologyTool(self.config, self._workflow_state)
            translation_tool = TranslationTool(self.config, self._workflow_state)
            
            # 创建Prompt Drivers（直接创建，不需要创建完整的Agent）
            # 所有Task使用相同的LLM配置，但可以通过system_prompt区分不同的任务角色
            preprocessing_prompt_driver = self._create_prompt_driver()
            terminology_prompt_driver = self._create_prompt_driver()
            translation_prompt_driver = self._create_prompt_driver()
            
            # 创建Griptape Workflow
            self.griptape_workflow = Workflow()
            
            # Task 1: 预处理任务
            # 工具从workflow_state获取cache_project，不需要LLM传递数据
            task1 = ToolkitTask(
                """你的任务是：立即调用preprocess_text工具，不要回复任何文本。

重要说明：
1. 直接调用工具即可，工具会自动获取所有需要的数据
2. 不需要传递任何参数
3. 不要询问任何问题，直接调用
4. 不要回复解释性文本，只调用工具""",
                tools=[preprocessing_tool],
                prompt_driver=preprocessing_prompt_driver
            )
            
            # Task 2: 术语识别任务
            # 工具从workflow_state获取cache_project和metadata，不需要LLM传递大对象
            task2 = ToolkitTask(
                """你的任务是：立即调用identify_terminology工具，不要回复任何文本。

重要说明：
1. 直接调用工具即可，工具会自动获取所有需要的数据
2. 不需要传递任何参数
3. 不要询问任何问题，直接调用
4. 不要回复解释性文本，只调用工具""",
                tools=[terminology_tool],
                prompt_driver=terminology_prompt_driver
            )
            
            # Task 3: 翻译任务
            # 工具从workflow_state获取所有需要的数据
            task3 = ToolkitTask(
                """你的任务是：立即调用translate_and_refine工具一次，然后直接返回结果。

重要说明：
1. 只调用工具一次即可，工具会自动获取所有需要的数据
2. 不需要传递任何参数
3. 工具返回结果后，直接输出该结果，不要重复调用
4. 不要回复解释性文本，不要询问任何问题
5. 禁止多次调用同一个工具""",
                tools=[translation_tool],
                prompt_driver=translation_prompt_driver
            )
            
            # 添加任务到工作流并设置依赖关系
            # 重要：必须按顺序执行，task2依赖task1完成，task3依赖task2完成
            self.griptape_workflow.add_task(task1)
            task2.add_parent(task1)  # task2必须等task1完成
            self.griptape_workflow.add_task(task2)
            task3.add_parent(task2)  # task3必须等task2完成
            self.griptape_workflow.add_task(task3)
            
            self.info("Griptape工作流初始化成功")
            self.info("[WorkflowManager] 工作流任务依赖关系：Task1 → Task2 → Task3")
            
        except Exception as e:
            self.error(f"Griptape工作流初始化失败: {e}", e)
            raise
    
    def _update_stage_progress(self, cache_project: CacheProject, stage: str, current: int, total: int):
        """更新当前阶段的进度信息（用于预估时间）"""
        import time
        
        if not cache_project.stats_data:
            return
        
        with cache_project.stats_data.atomic_scope():
            # 如果是新阶段，重置阶段开始时间
            if cache_project.stats_data.current_stage != stage:
                cache_project.stats_data.current_stage = stage
                cache_project.stats_data.stage_start_time = time.time()
                self.debug(f"[WorkflowManager] 进入新阶段: {stage}, 总进度={total}")
            
            # 更新进度
            cache_project.stats_data.stage_progress_current = current
            cache_project.stats_data.stage_progress_total = total
    
    def _publish_stage_with_stats(self, cache_project: CacheProject, stage: str, batch_info: str):
        """发送包含统计数据的阶段更新"""
        import time
        
        # 🔥 使用atomic_scope确保读取最新的统计数据
        if cache_project.stats_data:
            with cache_project.stats_data.atomic_scope():
                # 🔥 更新已消耗时间（确保阶段更新时也同步时间）
                cache_project.stats_data.time = time.time() - cache_project.stats_data.start_time
            update_data = cache_project.stats_data.to_dict()
        else:
            update_data = {}
        
        # 🔥 早期阶段（任务规划、文件处理、实体识别）：已翻译行数应该保持为0
        if stage in ["planning", "preprocessing", "terminology"]:
            update_data["line"] = 0
        
        # 🔥 添加阶段信息
        update_data["agent_stage"] = {
            "stage": stage,
            "batch_info": batch_info
        }
        
        self.debug(f"[WorkflowManager] 发送完整更新: stage={stage}, batch_info={batch_info}, line={update_data.get('line', 0)}/{update_data.get('total_line', 0)}, time={update_data.get('time', 0):.1f}s")
        self.emit(Base.EVENT.TASK_UPDATE, update_data)
    
    def execute_workflow(self, cache_project: CacheProject, 
                        human_intervention_callback=None,
                        progress_callback=None) -> Dict[str, Any]:
        """
        执行完整的多智能体工作流（使用Griptape框架）
        
        Args:
            cache_project: 缓存项目对象
            human_intervention_callback: 人工介入回调函数
            progress_callback: 进度回调函数 (current, total, stage, message)
            
        Returns:
            工作流执行结果
        """
        self.log_agent_action("开始执行多智能体工作流（Griptape）")
        
        workflow_result = {
            "success": False,
            "cache_project": cache_project,
            "stages": {}
        }
        
        if not self.griptape_workflow:
            self.error("Griptape工作流未初始化")
            return workflow_result
        
        try:
            # 将cache_project注入到共享状态中
            # Tools会从workflow_state获取数据，而不是通过LLM传递
            if not hasattr(self, "_workflow_state") or not isinstance(self._workflow_state, dict):
                self._workflow_state = {}
            
            self._workflow_state["cache_project"] = cache_project
            self._workflow_state["metadata"] = {}
            self._workflow_state["terminology_database"] = {}
            self._workflow_state["memory_storage"] = {}
            self._workflow_state["translation_results"] = []
            self._workflow_state["did_translate"] = False
            # 🔥 不再使用progress_callback，避免与新的阶段更新系统冲突
            # self._workflow_state["progress_callback"] = progress_callback
            self._workflow_state["human_intervention_callback"] = human_intervention_callback  # 🔥 传递人工介入回调
            
            # 调试：确认共享状态已注入
            self.info(f"[WorkflowManager] 共享状态已初始化，cache_project类型: {type(cache_project)}")
            self.info(f"[WorkflowManager] cache_project包含 {len(cache_project.files)} 个文件")
            
            # ===== 阶段0：任务规划（Planning Agent） =====
            self.info("=" * 50)
            self.info("阶段0: 任务规划与分析")
            self.info("=" * 50)
            
            # 🔥 发送UI阶段更新（包含统计数据）
            self._publish_stage_with_stats(cache_project, "planning", "分析中")
            
            # 🔥 不再使用progress_callback，避免与新的阶段更新系统冲突
            # if progress_callback:
            #     progress_callback(0, 100, "planning", "开始任务规划")
            
            # 执行规划
            self._update_stage_progress(cache_project, "planning", 0, 1)  # Planning阶段：单步操作
            planning_result = self.planning_agent.execute({
                "cache_project": cache_project
            })
            self._update_stage_progress(cache_project, "planning", 1, 1)  # Planning完成
            
            if planning_result.get("success"):
                task_analysis = planning_result.get("task_analysis", {})
                execution_plan = planning_result.get("execution_plan", {})
                resource_plan = planning_result.get("resource_plan", {})
                workflow_config = planning_result.get("workflow_config", {})
                task_memory = planning_result.get("task_memory", {})  # 获取任务元数据
                
                # ========== 详细打印Planning Agent分析结果 ==========
                self.info("")
                self.info("📊 【任务分析】")
                self.info(f"   • 文本单元数: {task_analysis['total_units']} 个")
                self.info(f"   • 平均长度: {task_analysis['avg_length']:.0f} 字符")
                self.info(f"   • 复杂度等级: {task_analysis['complexity'].upper()}")
                self.info(f"   • 文件类型: {', '.join(task_analysis['file_types'])}")
                self.info(f"   • 预计时间: {task_analysis['estimated_time']} 秒 "
                         f"({task_analysis['estimated_time']//60} 分钟)")
                
                self.info("")
                self.info("📋 【执行计划】")
                self.info(f"   • 执行模式: {execution_plan['mode'].upper()} (并行)")
                self.info(f"   • 最大并发数: {execution_plan['max_workers']} 个线程")
                self.info(f"   • 批次大小: {execution_plan['batch_size']} 个单元/批")
                self.info(f"   • 执行阶段: {' → '.join(execution_plan['stages'])}")
                self.info(f"   • 重试策略: 最多 {execution_plan['retry_policy']['max_retries']} 次, "
                         f"退避算法={execution_plan['retry_policy']['backoff']}")
                
                # ========== 打印chunk策略分配 ==========
                chunk_strategies = task_memory.get("chunk_strategies", [])
                if chunk_strategies:
                    self.info("")
                    self.info("🎯 【翻译策略分配】")
                    strategy_counts = {}
                    for chunk_info in chunk_strategies:
                        strategy = chunk_info["strategy"]
                        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                    
                    strategy_names = {"literal": "直译", "free": "意译", "stylized": "风格化"}
                    for strategy, count in strategy_counts.items():
                        strategy_cn = strategy_names.get(strategy, strategy)
                        self.info(f"   • {strategy_cn}策略: {count} 个批次")
                    
                    # 显示前3个批次的策略作为示例
                    self.info("")
                    self.info("   示例（前3个批次）：")
                    for i, chunk_info in enumerate(chunk_strategies[:3], 1):
                        strategy_cn = strategy_names.get(chunk_info["strategy"], chunk_info["strategy"])
                        self.info(f"   批次{i}: {strategy_cn} - {chunk_info['reason'][:50]}...")
                
                self.info("")
                self.info("💰 【资源评估】")
                self.info(f"   • 预计Token消耗: {resource_plan['estimated_tokens']:,} tokens")
                self.info(f"   • 预计成本: ${resource_plan['estimated_cost']:.2f} USD")
                self.info(f"   • 预计内存使用: {resource_plan['memory_usage']:.1f} MB")
                self.info(f"   • 预计API调用: {resource_plan['api_calls']:,} 次")
                if resource_plan.get("strategy_breakdown"):
                    self.info("   • 各策略API调用分布:")
                    for strategy, info in resource_plan["strategy_breakdown"].items():
                        strategy_cn = {"literal": "直译", "free": "意译", "stylized": "风格化"}.get(strategy, strategy)
                        self.info(f"     - {strategy_cn}: {info['api_calls']} 次 ({info['chunks']} 批次)")
                
                self.info("")
                self.info("⚙️  【工作流配置】")
                self.info(f"   • 启用预处理: {'是' if workflow_config['enable_preprocessing'] else '否'}")
                self.info(f"   • 启用术语识别: {'是' if workflow_config['enable_terminology'] else '否'}")
                self.info(f"   • 启用翻译: {'是' if workflow_config['enable_translation'] else '否'}")
                self.info(f"   • 并行翻译: {'是' if workflow_config['parallel_translation'] else '否'}")
                self.info(f"   • 最大并发翻译数: {workflow_config['max_concurrent_translations']}")
                self.info(f"   • 人工审核: {'启用' if workflow_config['enable_human_review'] else '禁用'}")
                if workflow_config['enable_human_review']:
                    self.info(f"   • 审核阈值: {workflow_config['review_threshold']*100:.0f}% 质量分")
                
                self.info("")
                self.info("✅ 任务规划完成！准备执行工作流...")
                self.info("=" * 50)
                
                # 将规划结果存入共享状态
                self._workflow_state["planning_result"] = planning_result
                self._workflow_state["execution_plan"] = execution_plan
                self._workflow_state["workflow_config"] = workflow_config
                self._workflow_state["task_memory"] = task_memory  # 🔥 存储任务元数据（chunk策略、实体数据库等）
                
                # 🔥 发送UI阶段更新：规划完成（包含统计数据）
                self._publish_stage_with_stats(cache_project, "planning", "完成")
            else:
                self.warning("⚠️  任务规划失败，使用默认配置继续执行")
            
            # 🔥 不再使用progress_callback，避免与新的阶段更新系统冲突
            # if progress_callback:
            #     progress_callback(100, 100, "planning", "任务规划完成")
            
            # 构建初始输入（只是触发工作流，不包含大数据）
            initial_input = "开始执行多智能体翻译工作流。"
            
            self.info("=" * 50)
            self.info("开始执行Griptape工作流")
            self.info("=" * 50)
            
            # 执行Griptape工作流
            # Griptape的run方法会按顺序执行所有Task
            workflow_output = self.griptape_workflow.run(initial_input)
            
            self.info("=" * 50)
            self.info("Griptape工作流执行完成")
            self.info("=" * 50)
            
            # 从共享状态中提取结果
            # 不再从workflow_output解析（那只是LLM的文本输出）
            # 实际结果在workflow_state中
            updated_cache_project = self._workflow_state.get("cache_project", cache_project)
            did_translate = bool(self._workflow_state.get("did_translate"))
            translated_count = len(self._workflow_state.get("translation_results", []) or [])
            
            if did_translate and translated_count > 0:
                workflow_result["success"] = True
                workflow_result["cache_project"] = updated_cache_project
                workflow_result["stages"] = {
                    "preprocess": self._workflow_state.get("metadata", {}),
                    "terminology_count": len(self._workflow_state.get("terminology_database", {}) or {}),
                    "translated_count": translated_count,
                }
                
                self.info("\n" + "=" * 60)
                self.info("🎉 多智能体翻译工作流执行成功")
                self.info("=" * 60)
                self.info(f"✓ 预处理: 领域={self._workflow_state.get('metadata', {}).get('domain')}, 风格={self._workflow_state.get('metadata', {}).get('style')}")
                self.info(f"✓ 术语识别: {len(self._workflow_state.get('terminology_database', {}) or {})} 个术语")
                self.info(f"✓ 翻译完成: {translated_count} 个文本单元")
                self.info("=" * 60 + "\n")
            else:
                # 必须使用Griptape工作流模式，不允许回退
                error_msg = f"Griptape工作流未产生有效翻译结果（did_translate={did_translate}, translated_count={translated_count}）。请检查工具调用是否成功。"
                self.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.log_agent_action("多智能体工作流执行完成")
            
        except Exception as e:
            self.error(f"Griptape工作流执行失败: {e}", e)
            # 必须使用Griptape工作流模式，不允许回退
            error_msg = f"Griptape工作流执行失败，必须修复错误才能继续。错误详情: {e}"
            self.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        return workflow_result
    
    def _extract_results_from_griptape_output(self, workflow_output, original_cache_project: CacheProject, recursion_depth: int = 0) -> Optional[Dict[str, Any]]:
        """从Griptape工作流输出中提取结果"""
        # 防止递归死循环
        if recursion_depth > 3:
            self.error(f"递归深度超过限制 ({recursion_depth})，停止提取")
            return None
        
        try:
            # Griptape的输出通常是TextArtifact或包含Artifact的列表
            # 我们需要解析JSON格式的结果
            
            output_text = None
            if hasattr(workflow_output, 'value'):
                output_text = str(workflow_output.value)
            elif hasattr(workflow_output, 'output'):
                output_text = str(workflow_output.output)
            elif isinstance(workflow_output, str):
                output_text = workflow_output
            else:
                output_text = str(workflow_output)
            
            # 清理输出文本中的空字节
            if output_text:
                output_text = output_text.replace('\x00', '').strip()
            
            self.info(f"Griptape输出文本长度: {len(output_text) if output_text else 0}")
            self.debug(f"Griptape输出文本前500字符: {output_text[:500] if output_text else 'None'}...")  # 记录前500字符用于调试
            
            # 尝试提取JSON部分
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                # 清理字符串中的空字节和其他无效字符
                json_str = json_str.replace('\x00', '').strip()
                
                try:
                    result = json.loads(json_str)
                except json.JSONDecodeError as e:
                    self.error(f"JSON解析失败: {e}, JSON字符串: {json_str[:200]}...")
                    # 尝试从工作流的tasks中提取（增加递归深度）
                    return self._extract_from_tasks(original_cache_project, recursion_depth + 1)
                
                # 转换cache_project
                if "cache_project" in result:
                    cache_project_data = result["cache_project"]
                    if isinstance(cache_project_data, str):
                        try:
                            # 清理字符串中的空字节
                            cache_project_data = cache_project_data.replace('\x00', '').strip()
                            # 如果字符串是JSON格式，直接解析
                            if cache_project_data.startswith('{') or cache_project_data.startswith('['):
                                result["cache_project"] = msgspec.json.decode(
                                    cache_project_data.encode('utf-8'),
                                    type=CacheProject
                                )
                            else:
                                # 如果不是JSON格式，尝试作为普通字符串解析
                                result["cache_project"] = msgspec.json.decode(
                                    cache_project_data.encode('utf-8'),
                                type=CacheProject
                            )
                        except Exception as e:
                            self.error(f"解析cache_project字符串失败: {e}, 数据长度: {len(cache_project_data)}, 前100字符: {cache_project_data[:100]}")
                            # 如果解析失败，尝试从工作流的tasks中提取（增加递归深度）
                            return self._extract_from_tasks(original_cache_project, recursion_depth + 1)
                    elif isinstance(cache_project_data, dict):
                        try:
                            result["cache_project"] = msgspec.json.decode(
                                json.dumps(cache_project_data).encode(),
                                type=CacheProject
                            )
                        except Exception as e:
                            self.error(f"解析cache_project字典失败: {e}")
                            result["cache_project"] = original_cache_project
                    else:
                        result["cache_project"] = original_cache_project
                else:
                    result["cache_project"] = original_cache_project
                
                # 检查是否真正执行了翻译（必须有cache_project且内容有变化）
                # 如果cache_project没有变化，说明没有真正执行翻译
                if "success" not in result:
                    # 检查cache_project是否被更新（通过比较文件数量或内容）
                    if "cache_project" in result:
                        cache_project_updated = result["cache_project"]
                        # 检查是否有翻译结果（检查是否有translated_text）
                        has_translation = False
                        if hasattr(cache_project_updated, 'files'):
                            for file in cache_project_updated.files:
                                if hasattr(file, 'items'):
                                    for item in file.items:
                                        if hasattr(item, 'translated_text') and item.translated_text:
                                            has_translation = True
                                            break
                                if has_translation:
                                    break
                        
                        result["success"] = has_translation
                    else:
                        result["success"] = False
                
                # 如果success为False，说明没有真正执行翻译，返回None
                if not result.get("success", False):
                    self.warning("Griptape工作流输出中没有有效的翻译结果")
                    return None
                
                return result
            else:
                # 如果没有找到JSON，尝试从工作流的tasks中提取（增加递归深度）
                return self._extract_from_tasks(original_cache_project, recursion_depth + 1)
        except Exception as e:
            self.error(f"解析Griptape输出失败: {e}", e)
            return None
    
    def _extract_from_tasks(self, original_cache_project: CacheProject, recursion_depth: int = 0) -> Optional[Dict[str, Any]]:
        """从工作流的tasks中提取结果"""
        # 防止递归死循环
        if recursion_depth > 3:
            self.error(f"递归深度超过限制 ({recursion_depth})，停止从tasks提取")
            return None
        
        try:
            if hasattr(self.griptape_workflow, 'tasks') and self.griptape_workflow.tasks:
                    # 获取最后一个task的输出
                last_task = self.griptape_workflow.tasks[-1]
                if last_task:
                    # 尝试多种方式获取输出
                    task_output = None
                    if hasattr(last_task, 'output') and last_task.output:
                        task_output = last_task.output
                    elif hasattr(last_task, 'output_text') and last_task.output_text:
                        task_output = last_task.output_text
                    elif hasattr(last_task, 'output_value') and last_task.output_value:
                        task_output = last_task.output_value
                    
                    if task_output:
                        # 检查是否是同一个对象，避免无限递归
                        if hasattr(self, '_last_extracted_output') and self._last_extracted_output is task_output:
                            self.warning("检测到重复的输出对象，停止递归提取")
                            return None
                        self._last_extracted_output = task_output
                        
                        self.info(f"从最后一个task提取输出，类型: {type(task_output)}")
                        return self._extract_results_from_griptape_output(task_output, original_cache_project, recursion_depth + 1)
                    else:
                        self.warning(f"最后一个task没有输出，task类型: {type(last_task)}, 属性: {dir(last_task)}")
            
            # 如果都失败了，返回None（让调用者处理）
            self.warning("无法从Griptape工作流中提取结果")
            return None
        except Exception as e:
            self.error(f"从tasks提取结果失败: {e}", e)
            import traceback
            self.error(f"详细错误: {traceback.format_exc()}")
            return None
    
    def _execute_fallback_workflow(self, cache_project: CacheProject,
                                  human_intervention_callback=None) -> Dict[str, Any]:
        """
        回退工作流：直接调用Agent（当Griptape执行失败时使用）
        包含人机协作节点
        """
        self.info("使用回退模式：直接调用Agent")
        
        workflow_result = {
            "success": False,
            "cache_project": cache_project,
            "stages": {}
        }
        
        try:
            # 阶段1: 译前预处理（文件处理）
            self.info("=" * 60)
            self.info("📄 阶段1: 文件处理")
            self.info("=" * 60)
            # 🔥 发送UI阶段更新（包含统计数据）
            file_count = len(cache_project.files) if cache_project.files else 1
            self._update_stage_progress(cache_project, "preprocessing", 0, file_count)  # Preprocessing阶段：基于文件数
            self._publish_stage_with_stats(cache_project, "preprocessing", "处理中")
            preprocessing_result = self.preprocessing_agent.execute({"cache_project": cache_project})
            self._update_stage_progress(cache_project, "preprocessing", file_count, file_count)  # Preprocessing完成
            if not preprocessing_result.get("success"):
                return workflow_result
            
            cache_project = preprocessing_result["cache_project"]
            metadata = preprocessing_result.get("metadata", {})
            
            # 阶段2: 术语识别（实体翻译）
            self.info("\n" + "=" * 60)
            self.info("📚 阶段2: 实体翻译（术语识别）")
            self.info("=" * 60)
            # 🔥 发送UI阶段更新（包含统计数据）
            self._publish_stage_with_stats(cache_project, "terminology", "识别中")
            terminology_result = self.terminology_agent.execute({
                "cache_project": cache_project,
                "metadata": metadata
            })
            if not terminology_result.get("success"):
                return workflow_result
            
            cache_project = terminology_result["cache_project"]
            terminology_db = terminology_result.get("terminology_database", {})
            memory_storage = terminology_result.get("memory_storage", {})
            
            # 人机协作节点1: 术语审核（如果需要）
            if human_intervention_callback:
                first_terms = self._get_first_occurrence_terms(terminology_db)
                if first_terms:
                    self.info(f"发现 {len(first_terms)} 个首次出现的术语，请求人工审核")
                    review_result = self._request_term_review(first_terms, human_intervention_callback)
                    if review_result:
                        self._update_terminology_from_review(terminology_db, review_result)
                        # 更新cache_project中的术语库
                        cache_project.extra["terminology_database"] = terminology_db
            
            # 阶段3: 翻译
            translation_result = self.translation_agent.execute({
                "cache_project": cache_project,
                "terminology_database": terminology_db,
                "memory_storage": memory_storage,
                "human_intervention_callback": human_intervention_callback  # 🔥 传递人工介入回调
            })
            if not translation_result.get("success"):
                return workflow_result
            
            cache_project = translation_result["cache_project"]
            translation_results = translation_result.get("translation_results", [])
            
            # 人机协作节点2: 翻译审核（如果需要）
            if human_intervention_callback:
                # 检查是否有需要审核的翻译错误
                error_items = []
                for result in translation_results:
                    # 这里可以根据质量评估结果判断是否需要审核
                    # 简化处理：检查是否有回译发现的问题
                    if result.get("status") != "success":
                        error_items.append(result)
                
                if error_items:
                    self.info(f"发现 {len(error_items)} 个需要审核的翻译项")
                    review_result = self._request_translation_review(error_items, human_intervention_callback)
                    if review_result and review_result.get("action") == "retranslate":
                        # 如果需要重新翻译，可以在这里处理
                        self.info("用户要求重新翻译部分内容")
            
            workflow_result["success"] = True
            workflow_result["cache_project"] = cache_project
            workflow_result["stages"] = {
                "preprocessing": preprocessing_result,
                "terminology": terminology_result,
                "translation": translation_result
            }
            
        except Exception as e:
            self.error(f"回退工作流执行失败: {e}", e)
        
        return workflow_result
    
    def _check_human_intervention(self, stage: str, stage_result: Dict, 
                                  callback) -> Optional[Dict]:
        """检查是否需要人工介入"""
        # 这里可以根据stage_result判断是否需要人工介入
        return None
    
    def _get_first_occurrence_terms(self, terminology_db: Dict) -> List[Dict]:
        """
        获取首次出现的术语（需要人工审核）
        返回格式: [{"term": "术语", "info": {...}}]
        """
        first_terms = []
        for term, info in terminology_db.items():
            # 检查是否需要人工审核：高优先级且未验证，或者是命名实体
            if (info.get("priority") == "high" and not info.get("verified_by_human")) or \
               (info.get("category") == "named_entity" and not info.get("verified_by_human")):
                first_terms.append({
                    "term": term,
                    "info": info
                })
        return first_terms[:10]  # 限制数量，避免一次审核太多
    
    def _request_term_review(self, terms: List[Dict], callback) -> Optional[Dict]:
        """请求术语审核"""
        if callback:
            return callback("terminology_review", {"terms": terms})
        return None
    
    def _update_terminology_from_review(self, terminology_db: Dict, review_result: Dict) -> None:
        """根据审核结果更新术语库"""
        if "approved_terms" in review_result:
            for term_info in review_result["approved_terms"]:
                term = term_info.get("term")
                if term in terminology_db:
                    terminology_db[term]["verified_by_human"] = True
                    if "translation" in term_info:
                        terminology_db[term]["translation"] = term_info["translation"]
    
    def _request_translation_review(self, error_items: List[Dict], callback) -> Optional[Dict]:
        """请求翻译审核"""
        if callback:
            return callback("translation_review", {"error_items": error_items})
        return None
    
    def log_agent_action(self, action: str, details: str = "") -> None:
        """记录工作流动作"""
        self.info(f"[WorkflowManager] {action}")
        if details:
            self.debug(f"[WorkflowManager] 详情: {details}")
