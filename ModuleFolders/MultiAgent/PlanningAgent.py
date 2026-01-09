"""
Planning Agent (规划Agent)
负责任务规划、资源调度和流程控制
"""

import json
from typing import Dict, Any, List, Optional
from .BaseAgent import BaseAgent
from ModuleFolders.Cache.CacheProject import CacheProject


class PlanningAgent(BaseAgent):
    """
    Planning Agent: 规划与调度Agent
    功能：
    1. 分析翻译任务复杂度
    2. 制定执行计划（串行/并行、分批大小等）
    3. 动态调整工作流（跳过/重试某些阶段）
    4. 监控各Agent执行状态
    5. 决策是否需要人工介入
    """
    
    def __init__(self, config=None):
        super().__init__(
            name="PlanningAgent",
            description="任务规划与流程控制Agent",
            config=config
        )
        
        self.execution_plan = {}  # 执行计划
        self.agent_status = {}  # Agent状态跟踪
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行规划任务
        """
        try:
            self.log_agent_action("开始执行任务规划")
            
            cache_project: CacheProject = input_data.get("cache_project")
            if not cache_project:
                self.error("未找到cache_project数据")
                return {"success": False, "error": "缺少cache_project"}
            
            # 1. 分析任务复杂度
            task_analysis = self._analyze_task_complexity(cache_project)
            self.info(f"任务分析完成: {task_analysis}")
            
            # 2. 细粒度分析文本块 - 为每个chunk打上策略标签
            self.info("正在进行文本块细粒度分析...")
            chunk_strategies = self._analyze_chunks_and_assign_strategies(cache_project)
            self.info(f"文本块分析完成: {len(chunk_strategies)} 个批次已分配策略")
            
            # 3. 制定执行计划
            execution_plan = self._create_execution_plan(task_analysis)
            self.info(f"执行计划: {execution_plan}")
            
            # 4. 评估资源需求
            resource_plan = self._estimate_resources(task_analysis, chunk_strategies)
            self.info(f"资源评估: {resource_plan}")
            
            # 5. 确定工作流配置
            workflow_config = self._configure_workflow(execution_plan, resource_plan)
            self.info(f"工作流配置: {workflow_config}")
            
            # 6. 构建Task Memory（任务元数据）
            task_memory = {
                "chunk_strategies": chunk_strategies,  # 每个chunk的翻译策略
                "terminology_database": {},  # 将由TerminologyAgent填充
                "style_guide": self._determine_style_guide(cache_project),  # 文体风格指南
                "entity_database": {},  # 实体数据库（用于一致性检查）
            }
            
            self.log_agent_action("任务规划完成", 
                                 f"预计处理 {task_analysis['total_units']} 个单元，"
                                 f"已为 {len(chunk_strategies)} 个批次分配翻译策略")
            
            return {
                "success": True,
                "cache_project": cache_project,
                "task_analysis": task_analysis,
                "execution_plan": execution_plan,
                "resource_plan": resource_plan,
                "workflow_config": workflow_config,
                "task_memory": task_memory,  # 新增：任务元数据
            }
        except Exception as e:
            self.error(f"任务规划执行失败: {e}", e)
            return {"success": False, "error": str(e)}
    
    def _analyze_task_complexity(self, cache_project: CacheProject) -> Dict[str, Any]:
        """
        分析任务复杂度
        
        Returns:
            {
                "total_units": 总文本单元数,
                "avg_length": 平均文本长度,
                "complexity": "simple" | "medium" | "complex",
                "file_types": 文件类型列表,
                "estimated_time": 预计时间（秒）
            }
        """
        from ModuleFolders.Cache.CacheItem import TranslationStatus
        
        total_units = 0
        total_length = 0
        file_types = set()
        
        for file_path, cache_file in cache_project.files.items():
            file_types.add(cache_file.file_project_type)
            for item in cache_file.items:
                if item.translation_status == TranslationStatus.UNTRANSLATED:
                    total_units += 1
                    total_length += len(item.source_text)
        
        avg_length = total_length / total_units if total_units > 0 else 0
        
        # 评估复杂度
        if total_units < 50 and avg_length < 100:
            complexity = "simple"
            estimated_time = total_units * 2  # 每单元约2秒
        elif total_units < 300 and avg_length < 500:
            complexity = "medium"
            estimated_time = total_units * 5  # 每单元约5秒
        else:
            complexity = "complex"
            estimated_time = total_units * 10  # 每单元约10秒
        
        return {
            "total_units": total_units,
            "avg_length": avg_length,
            "complexity": complexity,
            "file_types": list(file_types),
            "estimated_time": estimated_time,
        }
    
    def _create_execution_plan(self, task_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        制定执行计划
        
        Returns:
            {
                "mode": "parallel" | "serial",  # 并行/串行
                "batch_size": 批次大小,
                "max_workers": 最大并发数,
                "stages": ["preprocess", "terminology", "translate"],  # 需要执行的阶段
                "skip_stages": [],  # 可跳过的阶段
                "retry_policy": {"max_retries": 3, "backoff": "exponential"}
            }
        """
        complexity = task_analysis["complexity"]
        total_units = task_analysis["total_units"]
        
        if complexity == "simple":
            return {
                "mode": "parallel",
                "batch_size": min(total_units, 50),
                "max_workers": 5,
                "stages": ["preprocess", "terminology", "translate"],
                "skip_stages": [],
                "retry_policy": {"max_retries": 2, "backoff": "linear"}
            }
        elif complexity == "medium":
            return {
                "mode": "parallel",
                "batch_size": min(total_units, 100),
                "max_workers": 10,
                "stages": ["preprocess", "terminology", "translate"],
                "skip_stages": [],
                "retry_policy": {"max_retries": 3, "backoff": "exponential"}
            }
        else:  # complex
            return {
                "mode": "parallel",
                "batch_size": min(total_units, 200),
                "max_workers": 15,
                "stages": ["preprocess", "terminology", "translate"],
                "skip_stages": [],
                "retry_policy": {"max_retries": 5, "backoff": "exponential"}
            }
    
    def _estimate_resources(self, task_analysis: Dict[str, Any], chunk_strategies: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        评估资源需求（基于chunk策略的精确估算）
        
        Args:
            task_analysis: 任务分析结果
            chunk_strategies: 每个chunk的翻译策略（如果为None，使用粗略估算）
        
        Returns:
            {
                "estimated_tokens": 预计token消耗,
                "estimated_cost": 预计成本（美元）,
                "memory_usage": 预计内存使用（MB）,
                "api_calls": 预计API调用次数,
                "strategy_breakdown": 各策略的API调用分布
            }
        """
        total_units = task_analysis["total_units"]
        avg_length = task_analysis["avg_length"]
        
        if not chunk_strategies:
            # 粗略估算（向后兼容）
            tokens_per_unit = avg_length * 2
            estimated_tokens = total_units * tokens_per_unit
            api_calls = total_units * 3
            strategy_breakdown = {}
        else:
            # 精确估算：基于chunk策略
            # 新流程：步骤1（批量翻译）+ 步骤2（批量回译验证）
            # 每个chunk的API调用次数：1次批量翻译 + 2次回译验证（回译+修正） = 3次/chunk
            
            num_chunks = len(chunk_strategies)
            api_calls = num_chunks * 3  # 每个chunk：1次翻译 + 1次回译 + 1次修正（如需要）
            
            # 根据策略统计
            strategy_counts = {}
            for chunk_info in chunk_strategies:
                strategy = chunk_info["strategy"]
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            strategy_breakdown = {
                strategy: {
                    "chunks": count,
                    "api_calls": count * 3  # 每个chunk 3次API调用
                }
                for strategy, count in strategy_counts.items()
            }
            
            # Token估算（基于平均长度和chunk数）
            tokens_per_unit = avg_length * 2
        estimated_tokens = total_units * tokens_per_unit
        
        # DeepSeek价格约 $0.27 / 1M tokens (输入) + $1.1 / 1M tokens (输出)
        # 假设输入:输出 = 1:1.5
        input_tokens = estimated_tokens * 0.4
        output_tokens = estimated_tokens * 0.6
        estimated_cost = (input_tokens / 1_000_000 * 0.27) + (output_tokens / 1_000_000 * 1.1)
        
        # 内存使用（粗略估算）
        memory_usage = total_units * 0.1  # 每单元约0.1MB
        
        return {
            "estimated_tokens": int(estimated_tokens),
            "estimated_cost": round(estimated_cost, 2),
            "memory_usage": round(memory_usage, 1),
            "api_calls": api_calls,
            "strategy_breakdown": strategy_breakdown
        }
    
    def _configure_workflow(self, execution_plan: Dict[str, Any], resource_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        配置工作流参数
        
        Returns:
            {
                "enable_preprocessing": True/False,
                "enable_terminology": True/False,
                "enable_translation": True/False,
                "parallel_translation": True/False,
                "max_concurrent_translations": int,
                "enable_human_review": True/False,
                "review_threshold": 0.8  # 质量低于此阈值触发人工审核
            }
        """
        stages = execution_plan["stages"]
        max_workers = execution_plan["max_workers"]
        
        return {
            "enable_preprocessing": "preprocess" in stages,
            "enable_terminology": "terminology" in stages,
            "enable_translation": "translate" in stages,
            "parallel_translation": execution_plan["mode"] == "parallel",
            "max_concurrent_translations": max_workers,
            "enable_human_review": True,  # 🔥 启用人工审核
            "review_threshold": 0.8,  # 评分低于8.0（满分10）时触发人工审核
        }
    
    def update_agent_status(self, agent_name: str, status: str, progress: float = 0.0):
        """
        更新Agent执行状态
        
        Args:
            agent_name: Agent名称
            status: 状态（"pending", "running", "completed", "failed"）
            progress: 进度（0.0 - 1.0）
        """
        self.agent_status[agent_name] = {
            "status": status,
            "progress": progress,
            "updated_at": self._get_current_time()
        }
        self.info(f"[{agent_name}] 状态更新: {status} ({progress*100:.1f}%)")
    
    def should_intervene(self, agent_name: str, quality_score: float) -> bool:
        """
        判断是否需要人工介入
        
        Args:
            agent_name: Agent名称
            quality_score: 质量评分（0.0 - 1.0）
            
        Returns:
            True if需要人工介入
        """
        # 质量评分低于阈值时触发人工介入
        if quality_score < 0.7:
            self.warning(f"[{agent_name}] 质量评分过低 ({quality_score:.2f})，建议人工介入")
            return True
        return False
    
    def _get_current_time(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _analyze_chunks_and_assign_strategies(self, cache_project: CacheProject) -> List[Dict[str, Any]]:
        """
        细粒度分析每个文本块，为每个chunk分配翻译策略
        
        分析维度：
        1. 文本长度和复杂度
        2. 句子结构（简单句/复合句/长难句）
        3. 专业术语密度
        4. 文体风格（正式/非正式/文学性）
        
        策略类型：
        - "literal": 直译（技术文档、法律文本）
        - "free": 意译（小说、对话）
        - "stylized": 风格化（文学作品、营销文案）
        
        Returns:
            List of {
                "chunk_index": 批次索引,
                "strategy": "literal" | "free" | "stylized",
                "complexity": "simple" | "medium" | "complex",
                "style": "formal" | "informal" | "literary",
                "terminology_density": 0.0-1.0,  # 术语密度
                "avg_sentence_length": 平均句子长度,
                "reason": "选择该策略的原因"
            }
        """
        from ModuleFolders.Cache.CacheItem import TranslationStatus
        import re
        
        chunk_strategies = []
        chunk_index = 0
        
        # 获取配置
        if self.config:
            limit_type = "token" if getattr(self.config, 'tokens_limit_switch', False) else "line"
            limit_count = getattr(self.config, 'tokens_limit', 500) if limit_type == "token" else getattr(self.config, 'lines_limit', 15)
        else:
            limit_type = "line"
            limit_count = 15
        
        # 遍历所有文件
        for file_path, cache_file in cache_project.files.items():
            items = [item for item in cache_file.items if item.translation_status == TranslationStatus.UNTRANSLATED]
            
            if not items:
                continue
            
            # 模拟分块逻辑（与TranslationRefinementAgent一致）
            current_chunk, current_length, chunk_chars = [], 0, 0
            MAX_CHUNK_CHARS = 6000
            
            for item in items:
                item_length = item.token_count if limit_type == "token" else 1
                source_text_length = len(item.source_text)
                
                # 🔥 【智能分块策略】
                is_extreme_long = source_text_length > MAX_CHUNK_CHARS
                
                # 极端超长文本单独成chunk
                if is_extreme_long:
                    if current_chunk:
                        strategy_info = self._analyze_chunk_strategy(current_chunk, chunk_index)
                        chunk_strategies.append(strategy_info)
                        chunk_index += 1
                    
                    strategy_info = self._analyze_chunk_strategy([item], chunk_index)
                    chunk_strategies.append(strategy_info)
                    chunk_index += 1
                    current_chunk, current_length, chunk_chars = [], 0, 0
                    continue
                
                # 智能打包：按总字符数限制
                if current_chunk and (chunk_chars + source_text_length > MAX_CHUNK_CHARS):
                    strategy_info = self._analyze_chunk_strategy(current_chunk, chunk_index)
                    chunk_strategies.append(strategy_info)
                    chunk_index += 1
                    current_chunk, current_length, chunk_chars = [], 0, 0
                
                current_chunk.append(item)
                current_length += item_length
                chunk_chars += source_text_length
            
            # 处理最后一个chunk
            if current_chunk:
                strategy_info = self._analyze_chunk_strategy(current_chunk, chunk_index)
                chunk_strategies.append(strategy_info)
                chunk_index += 1
        
        return chunk_strategies
    
    def _analyze_chunk_strategy(self, chunk: List, chunk_index: int) -> Dict[str, Any]:
        """
        分析单个chunk并决定翻译策略
        
        Args:
            chunk: CacheItem列表
            chunk_index: 批次索引
            
        Returns:
            策略信息字典
        """
        import re
        
        # 收集chunk的所有文本
        texts = [item.source_text for item in chunk]
        combined_text = " ".join(texts)
        
        # 1. 计算平均句子长度
        sentences = re.split(r'[.!?。！？]+', combined_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        
        # 2. 计算专业术语密度（简单启发式：大写单词、特殊符号）
        words = combined_text.split()
        technical_words = sum(1 for w in words if w and (w[0].isupper() or '_' in w or '-' in w)) if words else 0
        terminology_density = technical_words / len(words) if words else 0
        
        # 3. 判断文体风格
        formal_indicators = len(re.findall(r'\b(therefore|thus|furthermore|moreover|whereas|hereby)\b', combined_text, re.I))
        informal_indicators = len(re.findall(r'\b(gonna|wanna|yeah|ok|hey)\b', combined_text, re.I))
        literary_indicators = len(re.findall(r'[，。！？—…""''；：]', combined_text))  # 中文标点
        
        if formal_indicators > informal_indicators:
            style = "formal"
        elif literary_indicators > len(combined_text) * 0.05:  # 中文标点占比>5%
            style = "literary"
        else:
            style = "informal"
        
        # 4. 评估复杂度
        if avg_sentence_length < 50 and terminology_density < 0.1:
            complexity = "simple"
        elif avg_sentence_length < 150 and terminology_density < 0.3:
            complexity = "medium"
        else:
            complexity = "complex"
        
        # 5. 决定翻译策略
        if terminology_density > 0.3 or style == "formal":
            # 高术语密度或正式文体 → 直译
            strategy = "literal"
            reason = f"高术语密度({terminology_density:.2f})或正式文体，选择直译策略"
        elif style == "literary" or complexity == "complex":
            # 文学性或复杂文本 → 风格化
            strategy = "stylized"
            reason = f"文学性文体或复杂句式，选择风格化策略"
        else:
            # 默认 → 意译
            strategy = "free"
            reason = f"普通对话或叙述性文本，选择意译策略"
        
        return {
            "chunk_index": chunk_index,
            "strategy": strategy,
            "complexity": complexity,
            "style": style,
            "terminology_density": round(terminology_density, 2),
            "avg_sentence_length": round(avg_sentence_length, 1),
            "reason": reason,
            "text_sample": texts[0][:50] + "..." if texts else ""  # 前50字符作为样本
        }
    
    def _determine_style_guide(self, cache_project: CacheProject) -> Dict[str, Any]:
        """
        确定整体文体风格指南
        
        Returns:
            {
                "overall_style": "formal" | "informal" | "literary",
                "tone": "professional" | "casual" | "artistic",
                "preferences": {
                    "use_honorifics": bool,  # 是否使用敬语
                    "preserve_formatting": bool,  # 是否保留格式
                    "maintain_rhythm": bool,  # 是否保持韵律（文学作品）
                }
            }
        """
        from ModuleFolders.Cache.CacheItem import TranslationStatus
        import re
        
        # 收集所有未翻译文本的样本
        all_texts = []
        for cache_file in cache_project.files.values():
            for item in cache_file.items:
                if item.translation_status == TranslationStatus.UNTRANSLATED:
                    all_texts.append(item.source_text)
                    if len(all_texts) >= 50:  # 采样50个文本单元
                        break
            if len(all_texts) >= 50:
                break
        
        if not all_texts:
            return {
                "overall_style": "informal",
                "tone": "casual",
                "preferences": {
                    "use_honorifics": False,
                    "preserve_formatting": True,
                    "maintain_rhythm": False,
                }
            }
        
        combined_sample = " ".join(all_texts[:20])  # 只分析前20个
        
        # 分析整体风格
        formal_score = len(re.findall(r'\b(therefore|thus|furthermore|moreover|whereas|hereby)\b', combined_sample, re.I))
        informal_score = len(re.findall(r'\b(gonna|wanna|yeah|ok|hey)\b', combined_sample, re.I))
        literary_score = len(re.findall(r'[，。！？—…""''；：]', combined_sample))
        
        if formal_score > max(informal_score, literary_score):
            overall_style = "formal"
            tone = "professional"
            use_honorifics = True
        elif literary_score > max(formal_score, informal_score):
            overall_style = "literary"
            tone = "artistic"
            use_honorifics = False
        else:
            overall_style = "informal"
            tone = "casual"
            use_honorifics = False
        
        return {
            "overall_style": overall_style,
            "tone": tone,
            "preferences": {
                "use_honorifics": use_honorifics,
                "preserve_formatting": True,  # 默认保留格式
                "maintain_rhythm": overall_style == "literary",
            }
        }
