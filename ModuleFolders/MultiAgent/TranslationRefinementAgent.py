"""
翻译与优化Agent (Agent 2)
负责翻译生成与迭代优化
"""

import re
import json
import copy
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from .BaseAgent import BaseAgent
from ModuleFolders.LLMRequester.LLMRequester import LLMRequester
from ModuleFolders.Cache.CacheProject import CacheProject
from ModuleFolders.Cache.CacheItem import CacheItem, TranslationStatus
from ModuleFolders.RequestLimiter.RequestLimiter import RequestLimiter
from ModuleFolders.ResponseExtractor.ResponseExtractor import ResponseExtractor


class TranslationRefinementAgent(BaseAgent):
    """
    Agent 2: 翻译与优化Agent
    功能：
    1. 多步骤引导翻译（理解—分解—转换—润色）
    2. 多版本生成与融合（直译版、意译版、风格化版）
    3. 回译验证与自我修正（TEaR框架）
    """
    
    def __init__(self, config=None):
        super().__init__(
            name="TranslationRefinementAgent",
            description="翻译生成与迭代优化Agent",
            config=config
        )
        
        self.llm_requester = LLMRequester()
        self.translation_versions = {}  # 存储多版本翻译
        self.request_limiter = RequestLimiter()  # 添加请求限制器
        self._current_cache_project = None  # 当前处理的cache_project
        
        # 初始化 DB Manager
        try:
            from ModuleFolders.Cache.DatabaseManager import DatabaseManager
            self.db_manager = DatabaseManager()
        except ImportError:
            self.db_manager = None

        # 配置请求限制器
        if self.config:
            rpm_limit = getattr(self.config, 'rpm_limit', 60)
            tpm_limit = getattr(self.config, 'tpm_limit', 10000)
            self.request_limiter.set_limit(tpm_limit, rpm_limit)
    
    def _get_atom_id(self, cache_project, file_path, row_index) -> Optional[int]:
        """[DB] 获取缓存项对应的数据库原子ID"""
        if hasattr(cache_project, 'db_atom_map'):
            file_map = cache_project.db_atom_map.get(file_path)
            if file_map:
                return file_map.get(row_index)
        return None
    
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
                self.debug(f"[TranslationAgent] 进入新阶段: {stage}, 总进度={total}")
            
            # 更新进度
            cache_project.stats_data.stage_progress_current = current
            cache_project.stats_data.stage_progress_total = total
    
    def _publish_stage_with_stats(self, cache_project: CacheProject, stage: str, batch_info: str):
        """发送包含统计数据的阶段更新（与WorkflowManager保持一致）"""
        from Base.Base import Base
        import time
        
        # 🔥 使用atomic_scope确保读取最新的统计数据
        if cache_project.stats_data:
            with cache_project.stats_data.atomic_scope():
                # 🔥 更新已消耗时间（确保阶段更新时也同步时间）
                cache_project.stats_data.time = time.time() - cache_project.stats_data.start_time
            update_data = cache_project.stats_data.to_dict()
        else:
            update_data = {}
        
        # 添加阶段信息
        update_data["agent_stage"] = {
            "stage": stage,
            "batch_info": batch_info
        }
        
        start_time_val = update_data.get('start_time', 0)
        time_val = update_data.get('time', 0)
        self.debug(f"[TranslationAgent] 发送阶段更新: stage={stage}, batch_info={batch_info}, line={update_data.get('line', 0)}/{update_data.get('total_line', 0)}, time={time_val:.1f}s, start_time={start_time_val:.0f}, active_llm={update_data.get('active_llm_calls', 0)}")
        self.emit(Base.EVENT.TASK_UPDATE, update_data)
    
    def _update_token_stats(self, prompt_tokens: int, completion_tokens: int):
        """更新token统计并发送UI更新事件（与原TaskExecutor保持一致）"""
        if not self._current_cache_project or not self._current_cache_project.stats_data:
            return
        
        from Base.Base import Base
        import time
        
        # 🔥 使用atomic_scope确保线程安全
        with self._current_cache_project.stats_data.atomic_scope():
            # 🔥 更新总token数（prompt + completion）
            if prompt_tokens or completion_tokens:
                self._current_cache_project.stats_data.token += (prompt_tokens or 0) + (completion_tokens or 0)
            
            # 🔥 更新completion_tokens（用于成本计算）
            if completion_tokens:
                self._current_cache_project.stats_data.total_completion_tokens += completion_tokens
            
            # 更新请求计数
            self._current_cache_project.stats_data.total_requests += 1
            
            # 🔥 更新已消耗时间（与原TaskExecutor保持一致）
            self._current_cache_project.stats_data.time = time.time() - self._current_cache_project.stats_data.start_time
            
            # 🔥 立即发送UI更新事件，确保token统计实时更新
            stats_dict = self._current_cache_project.stats_data.to_dict()
        
        # 在atomic_scope外发送事件
        self.emit(Base.EVENT.TASK_UPDATE, stats_dict)
    
    def _update_line_stats(self, row_count: int):
        """更新已翻译行数统计并发送UI更新事件（与原TaskExecutor保持一致）"""
        if not self._current_cache_project or not self._current_cache_project.stats_data:
            return
        
        from Base.Base import Base
        import time
        
        # 🔥 使用atomic_scope确保线程安全
        with self._current_cache_project.stats_data.atomic_scope():
            # 🔥 更新已翻译行数
            self._current_cache_project.stats_data.line += row_count
            
            # 🔥 更新已消耗时间（与原TaskExecutor保持一致）
            self._current_cache_project.stats_data.time = time.time() - self._current_cache_project.stats_data.start_time
            
            # 🔥 立即发送UI更新事件，确保行数统计实时更新
            stats_dict = self._current_cache_project.stats_data.to_dict()
        
        # 在atomic_scope外发送事件
        # 每5次更新打印一次日志，避免刷屏
        if not hasattr(self, '_line_update_count'):
            self._line_update_count = 0
        self._line_update_count += 1
        if self._line_update_count % 5 == 1:
            self.debug(f"[TranslationAgent] 更新行数统计: +{row_count}, total={stats_dict.get('line', 0)}/{stats_dict.get('total_line', 0)}, time={stats_dict.get('time', 0):.1f}s, active_llm={stats_dict.get('active_llm_calls', 0)}")
        self.emit(Base.EVENT.TASK_UPDATE, stats_dict)
    
    def _llm_request_with_tracking(self, messages, system_prompt, platform_config):
        """
        包装LLM请求，自动追踪活跃调用数
        
        Returns:
            (skip, response_think, response_content, prompt_tokens, completion_tokens)
        """
        if not self._current_cache_project or not self._current_cache_project.stats_data:
            return self.llm_requester.sent_request(messages, system_prompt, platform_config)
        
        from Base.Base import Base
        
        try:
            # 🔥 调用前：增加活跃LLM调用计数并立即发送事件
            with self._current_cache_project.stats_data.atomic_scope():
                self._current_cache_project.stats_data.active_llm_calls += 1
                stats_dict = self._current_cache_project.stats_data.to_dict()
            self.emit(Base.EVENT.TASK_UPDATE, stats_dict)
            
            # 执行LLM请求
            result = self.llm_requester.sent_request(messages, system_prompt, platform_config)
            
            return result
        finally:
            # 🔥 调用后：减少活跃LLM调用计数并立即发送事件
            with self._current_cache_project.stats_data.atomic_scope():
                self._current_cache_project.stats_data.active_llm_calls = max(0, self._current_cache_project.stats_data.active_llm_calls - 1)
                stats_dict = self._current_cache_project.stats_data.to_dict()
            self.emit(Base.EVENT.TASK_UPDATE, stats_dict)
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行翻译和优化任务
        
        Args:
            input_data: 包含cache_project、terminology_database等的字典
            
        Returns:
            包含翻译结果和优化信息的字典
        """
        self.log_agent_action("开始执行翻译与优化")
        
        cache_project: CacheProject = input_data.get("cache_project")
        terminology_db = input_data.get("terminology_database", {})
        memory_storage = input_data.get("memory_storage", {})
        progress_callback = input_data.get("progress_callback")  # 获取进度回调
        planning_result = input_data.get("planning_result", {})  # 获取规划结果
        task_memory = input_data.get("task_memory", {})  # 获取任务元数据（包含chunk策略和实体数据库）
        human_intervention_callback = input_data.get("human_intervention_callback", None)  # 🔥 获取人工介入回调
        
        # 🔥 保存cache_project引用，用于token统计
        self._current_cache_project = cache_project
        
        # 🔥 保存human_intervention_callback引用，用于人工审核
        self._human_intervention_callback = human_intervention_callback
        
        if not cache_project:
            self.error("未找到cache_project数据")
            return {"success": False, "error": "缺少cache_project"}
        
        # 🔥 使用与原TaskExecutor相同的批量翻译策略
        # 生成chunks（每个chunk包含多行文本，而不是单行）
        translation_chunks, context_chunks, file_paths = self._prepare_translation_chunks(cache_project)
        
        total_chunks = len(translation_chunks)
        
        # 统计总文本单元数（用于进度显示）
        total_units = sum(len(chunk) for chunk in translation_chunks)
        self.info(f"批量翻译模式：{total_units} 个文本单元，分为 {total_chunks} 个批次")
        
        # 🔥 使用与原TaskExecutor相同的并发数计算策略
        # 不需要降级系数，因为我们现在使用批量翻译（多行合并为一个chunk）
        if planning_result and "execution_plan" in planning_result:
            max_workers = planning_result["execution_plan"].get("max_workers", 10)
        elif self.config and hasattr(self.config, 'actual_thread_counts'):
            # 直接使用配置的线程数（与原TaskExecutor一致）
            max_workers = self.config.actual_thread_counts
            self.info(f"使用配置的线程数: {max_workers} (与原翻译方法一致)")
        else:
            max_workers = 10  # 默认值
        
        # 批量翻译模式下，并发数等于chunk数量
        max_workers = min(max_workers, total_units)  # 不超过总chunk数
        
        self.info("=" * 60)
        self.info(f"开始批量翻译：{total_units} 个文本单元，分为 {total_chunks} 个批次")
        self.info(f"并发线程数: {max_workers}")
        self.info("=" * 60)
        
        # 🔥 发送UI阶段更新：开始批量翻译
        self._update_stage_progress(cache_project, "translating", 0, total_chunks)  # 使用chunk数量而不是unit数量
        self._publish_stage_with_stats(cache_project, "translating", "翻译中")
        
        # 发送初始进度
        if progress_callback:
            progress_callback(0, total_units, "translation", "开始批量翻译")
        
        results = []
        completed_units = 0  # 已完成的文本单元数
        all_chunks_data = []  # 🔥 收集所有批次的数据（用于统一人工审核和实体检查）
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有批量翻译任务
            future_to_chunk = {
                executor.submit(
                    self._translate_chunk,  # 改为批量翻译
                    chunk, 
                    context_chunk,
                    file_path,
                    idx, 
                    total_chunks, 
                    terminology_db, 
                    memory_storage,
                    task_memory,  # 传递任务元数据（包含chunk策略）
                    progress_callback,
                    completed_units  # 传递已完成数用于进度更新
                ): (chunk, idx) 
                for idx, (chunk, context_chunk, file_path) in enumerate(zip(translation_chunks, context_chunks, file_paths), 1)
            }
            
            # 按完成顺序收集结果
            completed_chunks = 0  # 🔥 追踪已完成的chunk数量
            for future in as_completed(future_to_chunk):
                chunk, chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    if result and result.get("success"):
                        results.extend(result.get("translated_items", []))
                        all_chunks_data.append(result)  # 🔥 保存批次数据
                        chunk_size = len(chunk)
                        completed_units += chunk_size
                        completed_chunks += 1  # 🔥 增加已完成chunk数
                        
                        # 🔥 更新阶段进度（基于chunk数量）
                        self._update_stage_progress(cache_project, "translating", completed_chunks, total_chunks)
                        
                        # 更新进度
                        if progress_callback:
                            progress_callback(
                                completed_units, 
                                total_units, 
                                "translation", 
                                f"已翻译 {completed_units}/{total_units} 个单元 (批次 {completed_chunks}/{total_chunks})"
                            )
                except Exception as exc:
                    self.error(f"翻译批次 {chunk_idx} 失败: {exc}", exc)
        
        # 🔥 ========== 统一人工审核（所有批次完成后） ==========
        self.info("\n" + "="*60)
        self.info("所有批次回译验证完成，开始统一人工审核...")
        self.info("="*60)
        
        # 获取工作流配置
        workflow_config = planning_result.get("workflow_config", {})
        
        all_chunks_data = self._unified_human_review(all_chunks_data, terminology_db, task_memory, workflow_config)
        
        # 🔥 ========== 统一实体一致性检查（人工审核后） ==========
        self.info("\n" + "="*60)
        self.info("开始统一实体一致性检查...")
        self.info("="*60)
        entity_database = task_memory.get("entity_database", {})
        all_chunks_data = self._unified_entity_check(all_chunks_data, terminology_db, entity_database)
        
        # 🔥 ========== 更新缓存（所有检查完成后） ==========
        for chunk_data in all_chunks_data:
            chunk_items = chunk_data.get("chunk_items", [])
            translated_texts = chunk_data.get("translated_texts", [])
            for item, translated_text in zip(chunk_items, translated_texts):
                if translated_text:
                    self._update_cache_item(item, translated_text)
        
        self.log_agent_action("翻译与优化完成", f"成功翻译 {len(results)} 个单元")
        
        return {
            "success": True,
            "cache_project": cache_project,
            "translation_results": results
        }
    
    def _prepare_translation_chunks(self, cache_project: CacheProject):
        """
        准备翻译批次（chunks）- 使用与原TaskExecutor相同的批量策略
        
        Returns:
            chunks: List[List[CacheItem]] - 批次列表，每个批次包含多个文本单元
            context_chunks: List[List[CacheItem]] - 上下文批次
            file_paths: List[str] - 文件路径列表
        """
        from ModuleFolders.TaskConfig.TaskType import TaskType
        
        # 获取批量翻译配置（与原TaskExecutor一致）
        if self.config:
            limit_type = "token" if getattr(self.config, 'tokens_limit_switch', False) else "line"
            limit_count = getattr(self.config, 'tokens_limit', 500) if limit_type == "token" else getattr(self.config, 'lines_limit', 15)
            previous_line_count = getattr(self.config, 'pre_line_counts', 3)
        else:
            limit_type = "line"
            limit_count = 15  # 默认每批15行
            previous_line_count = 3
        
        chunks, context_chunks, file_paths = [], [], []
        
        for file_path, cache_file in cache_project.files.items():
            # 筛选未翻译的条目
            items = [item for item in cache_file.items if item.translation_status == TranslationStatus.UNTRANSLATED]
            
            if not items:
                continue
            
            current_chunk, current_length = [], 0
            chunk_start_idx = 0
            
            for i, item in enumerate(items):
                # 计算item长度（按行或按token）
                item_length = item.token_count if limit_type == "token" else 1
                source_text_length = len(item.source_text)
                
                # 🔥 【智能分块策略】
                # 策略：按总字符数分块，而不是固定行数
                # - 极端超长文本（>6000字符）：单独成chunk
                # - 普通文本：累计不超过6000字符/chunk
                MAX_CHUNK_CHARS = 6000
                is_extreme_long = source_text_length > MAX_CHUNK_CHARS
                
                # 记录chunk起始索引
                if not current_chunk:
                    chunk_start_idx = i
                    chunk_chars = 0  # 跟踪当前chunk的总字符数
                
                # 🔥 极端超长文本（>6000字符）单独成chunk
                if is_extreme_long:
                    # 先提交当前chunk（如果有）
                    if current_chunk:
                        chunks.append(current_chunk)
                        context_chunk = self._generate_context_chunk(items, previous_line_count, chunk_start_idx)
                        context_chunks.append(context_chunk)
                        file_paths.append(file_path)
                    
                    # 极端超长文本单独成chunk
                    chunks.append([item])
                    context_chunk = self._generate_context_chunk(items, previous_line_count, i)
                    context_chunks.append(context_chunk)
                    file_paths.append(file_path)
                    
                    self.debug(f"  ⚡ 极端超长文本 (第{i+1}项, {source_text_length}字符)，单独成chunk")
                    
                    # 重置
                    current_chunk, current_length, chunk_chars = [], 0, 0
                    chunk_start_idx = -1
                    continue
                
                # 🔥 智能打包：按总字符数限制
                # 如果加入当前item会超过MAX_CHUNK_CHARS，先提交当前chunk
                if current_chunk and (chunk_chars + source_text_length > MAX_CHUNK_CHARS):
                    chunks.append(current_chunk)
                    context_chunk = self._generate_context_chunk(items, previous_line_count, chunk_start_idx)
                    context_chunks.append(context_chunk)
                    file_paths.append(file_path)
                    
                    # 重置
                    current_chunk, current_length, chunk_chars = [], 0, 0
                    chunk_start_idx = i
                
                # 添加当前item到chunk
                current_chunk.append(item)
                current_length += item_length
                chunk_chars += source_text_length
            
            # 处理最后一个chunk
            if current_chunk:
                chunks.append(current_chunk)
                context_chunk = self._generate_context_chunk(items, previous_line_count, chunk_start_idx)
                context_chunks.append(context_chunk)
                file_paths.append(file_path)
        
        return chunks, context_chunks, file_paths
    
    def _generate_context_chunk(self, all_items: List[CacheItem], previous_count: int, start_idx: int) -> List[CacheItem]:
        """生成上下文chunk（与原CacheManager.generate_previous_chunks一致）"""
        if previous_count <= 0 or start_idx <= 0:
            return []
        
        from_idx = max(0, start_idx - previous_count)
        to_idx = start_idx
        
        return all_items[from_idx:to_idx]
    
    def _translate_chunk(self, chunk: List[CacheItem], context_chunk: List[CacheItem], 
                         file_path: str, chunk_idx: int, total_chunks: int,
                         terminology_db: Dict, memory_storage: Dict, task_memory: Dict,
                         progress_callback=None, completed_units: int = 0) -> Optional[Dict[str, Any]]:
        """
        批量翻译一个chunk（多行文本）- 基于PlanningAgent策略的智能翻译
        
        Args:
            chunk: 待翻译的文本单元列表（10-15个CacheItem）
            context_chunk: 上下文单元列表
            file_path: 文件路径
            chunk_idx: 当前批次序号
            total_chunks: 总批次数
            terminology_db: 术语库
            memory_storage: 记忆存储
            task_memory: 任务元数据（包含chunk策略和实体数据库）
            progress_callback: 进度回调
            completed_units: 已完成的单元数
            
        Returns:
            翻译结果字典
        """
        try:
            chunk_size = len(chunk)
            self.info(f"\n{'='*60}")
            self.info(f"[{chunk_idx}/{total_chunks}] 正在批量翻译 {chunk_size} 个文本单元...")
            self.info(f"{'='*60}")
            
            # 构建批量翻译的prompt（合并所有文本）
            source_texts = [item.source_text for item in chunk]
            context_texts = [item.source_text for item in context_chunk] if context_chunk else []
            
            # 获取当前chunk的翻译策略（从PlanningAgent的分析结果）
            chunk_strategies = task_memory.get("chunk_strategies", [])
            chunk_strategy_info = chunk_strategies[chunk_idx - 1] if chunk_idx - 1 < len(chunk_strategies) else None
            strategy = chunk_strategy_info["strategy"] if chunk_strategy_info else "free"  # 默认意译
            
            self.info(f"  📋 翻译策略: {strategy} ({chunk_strategy_info['reason'] if chunk_strategy_info else '默认意译'})")
            
            # ========== 步骤1+2合并: 基于策略的批量翻译（多步骤引导） ==========
            # 根据PlanningAgent的策略，直接执行对应的翻译方式，不需要生成多个版本
            self.info(f"  → 步骤1: 批量{strategy}翻译（多步骤引导: 理解→分解→转换→润色）...")
            
            translated_texts = self._strategy_based_batch_translation(
                source_texts, context_texts, strategy, terminology_db, memory_storage
            )
            
            # 🔥 【关键检查】严格验证返回行数
            if translated_texts and len(translated_texts) != chunk_size:
                self.error(f"  ❌ 致命错误：返回行数不匹配！原文{chunk_size}行，译文{len(translated_texts)}行")
                self.error(f"  → 这会导致后续所有内容错位，必须重新翻译整个batch")
                self.warning(f"  → 触发完全重译...")
                # 强制逐行重译整个batch
                translated_texts = self._fallback_translate_one_by_one(
                    source_texts, context_texts, strategy, terminology_db, memory_storage
                )
            
            # 🔥 Fallback机制：如果批量翻译失败或部分失败，对缺失的行进行单独重试
            if not translated_texts:
                self.warning(f"  ⚠ 批量翻译完全失败，尝试逐行翻译...")
                translated_texts = self._fallback_translate_one_by_one(
                    source_texts, context_texts, strategy, terminology_db, memory_storage
                )
            else:
                # 🔥 补齐列表长度（如果返回数量不足）
                while len(translated_texts) < chunk_size:
                    translated_texts.append("")
                
                # 🔥 检查每一行，对以下情况进行补充翻译：
                # 1. 空字符串
                # 2. 严重截断（译文长度 < 原文长度 * 0.3 且原文长度 > 100）
                problem_indices = []
                for i, (src, trans) in enumerate(zip(source_texts[:chunk_size], translated_texts[:chunk_size])):
                    if not trans or trans.strip() == "":
                        problem_indices.append((i, "空"))
                    elif len(src) > 100 and len(trans) < len(src) * 0.3:
                        # 译文严重截断（长原文但译文太短）
                        problem_indices.append((i, f"截断({len(trans)}/{len(src)})"))
                
                if problem_indices:
                    self.warning(f"  ⚠ 批量翻译部分失败: {len(problem_indices)} 行需要重试...")
                    
                    for i, reason in problem_indices:
                        self.warning(f"    → 正在单独翻译第 {i+1} 行（{reason}）...")
                        single_translation = self._translate_single_line(
                            source_texts[i], context_texts, strategy, terminology_db, memory_storage
                        )
                        if single_translation and single_translation.strip():
                            # 检查单行翻译是否也被截断
                            if len(source_texts[i]) > 100 and len(single_translation) < len(source_texts[i]) * 0.3:
                                self.warning(f"    ⚠ 第 {i+1} 行单独翻译也被截断 ({len(single_translation)}/{len(source_texts[i])})")
                            translated_texts[i] = single_translation
                            self.info(f"    ✓ 第 {i+1} 行翻译完成 (长度: {len(single_translation)})")
                        else:
                            # 如果单独翻译也失败，使用原文标记
                            translated_texts[i] = f"[翻译失败]{source_texts[i]}"
                            self.error(f"    ✗ 第 {i+1} 行翻译失败，保留原文")
                    
                    self.info(f"  ✓ 补充翻译完成")
            
            # 最终检查
            if len(translated_texts) != chunk_size:
                self.error(f"  ✗ 批次 {chunk_idx} 翻译失败：无法完成所有行的翻译")
                return {"success": False}
            
            self.info(f"  ✓ 策略翻译完成: {len(translated_texts)} 行")
            
            # ========== 步骤2: 批量回译验证（不进行人工审核，只返回评分数据） ==========
            self.info(f"  → 步骤2: 批量回译验证（TEaR: 批量回译→批量评估→批量修正）...")
            verified_texts, quality_scores, back_translations = self._batch_tear_verification_with_scores(
                source_texts, translated_texts, terminology_db
            )
            
            if not verified_texts or len(verified_texts) != chunk_size:
                self.warning(f"  ⚠ 回译验证失败，使用原译文")
                verified_texts = translated_texts
                quality_scores = [8.0] * chunk_size  # 默认评分
                back_translations = [""] * chunk_size
            
            self.info(f"  ✓ 回译验证完成: {len(verified_texts)} 行")
            translated_texts = verified_texts
            
            # [DB] Phase 3: 记录评估轨迹 (Evaluate Trace) + 更新 examination 字段
            if self.db_manager and self._current_cache_project:
                try:
                    for idx, (score, back_trans) in enumerate(zip(quality_scores, back_translations)):
                        if idx < len(chunk):
                            item = chunk[idx]
                            atom_id = self._get_atom_id(self._current_cache_project, file_path, item.row_index)
                            if atom_id:
                                # 确定警告级别
                                if score >= 8.0:
                                    warning_level = "low"
                                elif score >= 6.0:
                                    warning_level = "medium"
                                else:
                                    warning_level = "high"
                                
                                # 记录评估轨迹
                                self.db_manager.add_trace(
                                    atom_id=atom_id,
                                    agent_role="QualityAssessor",
                                    action_type="evaluate",
                                    content=f"Quality Score: {score}",
                                    quality_report={
                                        "score": score,
                                        "back_translation": back_trans,
                                        "status": "pass" if score >= 8.0 else "needs_refinement"
                                    }
                                )
                                
                                # 更新原子的 examination 字段（质量检查信息）
                                examination = {
                                    "back_translation": back_trans,
                                    "warning_level": warning_level,
                                    "semantic_similarity": score / 10.0,  # 转换为0-1范围
                                    "issues": [] if score >= 8.0 else ["需要润色"],
                                    "algorithm": "backtranslation"
                                }
                                self.db_manager.update_atom_examination(atom_id, examination)
                except Exception as e:
                    self.error(f"[DB] 记录评估轨迹失败: {e}")

            # 更新缓存
            translated_items = []
            for item, translated_text in zip(chunk, translated_texts):
                if translated_text:
                    self._update_cache_item(item, translated_text)
                    translated_items.append({
                        "source": item.source_text,
                        "translated": translated_text,
                        "status": "success"
                    })

                    # [DB] Phase 3: 记录初翻轨迹 (Draft Trace)
                    if self.db_manager and self._current_cache_project:
                        atom_id = self._get_atom_id(self._current_cache_project, file_path, item.row_index)
                        if atom_id:
                            self.db_manager.add_trace(
                                atom_id=atom_id,
                                agent_role="Translator",
                                action_type="draft",
                                content=translated_text,
                                meta_data={"strategy": strategy}
                            )
                            # 同时更新原子的翻译结果
                            self.db_manager.update_atom_translation(
                                atom_id=atom_id,
                                translated_text=translated_text,
                                status_code=1  # 已初翻
                            )
            
            # 🔥 更新行数统计（每完成一个chunk就更新）
            self._update_line_stats(chunk_size)
            
            self.info(f"✓ 批次 {chunk_idx} 完整翻译流程完成: {chunk_size} 个单元")
            self.info(f"{'='*60}\n")
        
            return {
                "success": True,
                "translated_items": translated_items,
                "chunk_size": chunk_size,
                "source_texts": source_texts,  # 🔥 返回原文
                "translated_texts": translated_texts,  # 🔥 返回译文
                "quality_scores": quality_scores,  # 🔥 返回评分
                "back_translations": back_translations,  # 🔥 返回回译
                "chunk_items": chunk,  # 🔥 返回CacheItem用于后续更新
                "file_path": file_path # 🔥 记录文件路径
            }
                
        except Exception as e:
            self.error(f"翻译批次 {chunk_idx} 失败: {e}", e)
            return {"success": False}
    
    def _translate_single_unit(self, unit: Dict, idx: int, total_units: int, terminology_db: Dict, memory_storage: Dict) -> Optional[Dict[str, Any]]:
        """
        翻译单个文本单元（用于并行调用）
        
        Args:
            unit: 翻译单元
            idx: 当前序号
            total_units: 总数
            terminology_db: 术语库
            memory_storage: 记忆存储
            
        Returns:
            翻译结果字典
        """
        try:
            self.info(f"\n{'='*60}")
            self.info(f"[{idx}/{total_units}] 正在翻译...")
            self.info(f"{'='*60}")
            self.info(f"原文: {unit['source_text'][:200]}{'...' if len(unit['source_text']) > 200 else ''}")
            self.info(f"-" * 60)
            
            # 1. 多步骤引导翻译
            translated_text = self._multi_step_translation(unit, terminology_db, memory_storage)
            
            # 2. 多版本生成与融合
            if translated_text:
                optimized_text = self._multi_version_fusion(unit, translated_text, terminology_db, memory_storage)
            else:
                optimized_text = translated_text
            
            # 3. 回译验证与自我修正（TEaR）
            if optimized_text:
                final_text = self._tear_verification(unit, optimized_text, terminology_db)
            else:
                final_text = optimized_text
            
            # 更新缓存
            if final_text:
                self._update_cache_item(unit["item"], final_text)
                
                # 输出翻译结果
                self.info(f"译文: {final_text[:200]}{'...' if len(final_text) > 200 else ''}")
                self.info(f"✓ 翻译完成 [{idx}/{total_units}]")
                self.info(f"{'='*60}\n")
                
                return {
                    "item_id": unit["item_id"],
                    "source": unit["source_text"],
                    "translated": final_text,
                    "status": "success"
                }
            else:
                self.warning(f"翻译单元 {unit['item_id']} 返回空结果")
                return None
                
        except Exception as e:
            self.error(f"翻译单元 {unit['item_id']} 失败: {e}", e)
            return None
    
    def _multi_step_batch_translation(self, source_texts: List[str], context_texts: List[str],
                                      terminology_db: Dict, memory_storage: Dict) -> Optional[List[str]]:
        """
        批量多步骤翻译（一次API调用翻译多行）
        使用与原TranslatorTask相同的textarea格式和ResponseExtractor解析
        
        Args:
            source_texts: 待翻译文本列表
            context_texts: 上下文文本列表
            terminology_db: 术语库
            memory_storage: 记忆存储
            
        Returns:
            翻译结果列表
        """
        self.info("  → 步骤1: 批量多步骤翻译（理解→分解→转换→润色）...")
        
        # 构建批量翻译提示词（✅ 传递source_texts用于动态筛选）
        terminology_prompt = self._build_terminology_prompt(terminology_db, source_texts)
        memory_context = self._build_memory_context(memory_storage)
        
        # 【关键】使用与原TranslatorTask相同的system_prompt格式
        system_prompt = f"""你是一位专业的翻译专家。请按照以下步骤进行翻译：

步骤1 - 理解：分析原文的语义、语境和风格
步骤2 - 分解：对于长难句，先识别主干成分和从句层级
步骤3 - 转换：将原文转换为目标语言，保持语义准确
步骤4 - 润色：优化译文，确保流畅自然

{terminology_prompt}
{memory_context}

【重要】输出格式要求：
- 必须使用<textarea>标签包裹所有译文
- 每行译文前必须加上序号（如1. 2. 3.）
- 不要添加任何额外的标题、前缀或说明文字
- 格式示例：
<textarea>
1.第一行译文
2.第二行译文
3.第三行译文
</textarea>"""
        
        # 【关键】构建source_text_dict（与原TranslatorTask完全相同）
        source_text_dict = {str(i): text for i, text in enumerate(source_texts)}
        
        # 【关键】使用与原PromptBuilder.build_source_text相同的逻辑构建原文
        numbered_lines = []
        for index, line in enumerate(source_texts):
            # 检查是否为多行文本
            if "\n" in line:
                lines = line.split("\n")
                numbered_text = f"{index + 1}.[\n"
                total_lines = len(lines)
                for sub_index, sub_line in enumerate(lines):
                    # 仅当只有一个尾随空格时才去除
                    sub_line = sub_line[:-1] if re.match(r'.*[^ ] $', sub_line) else sub_line
                    numbered_text += f'"{index + 1}.{total_lines - sub_index}.,{sub_line}",\n'
                numbered_text = numbered_text.rstrip('\n').rstrip(',')
                numbered_text += f"\n]"
                numbered_lines.append(numbered_text)
            else:
                # 单行文本直接添加序号
                numbered_lines.append(f"{index + 1}.{line}")
        
        source_text = "\n".join(numbered_lines)
        
        # 【关键】构建上下文（与原方法相同）
        context_str = "\n".join(context_texts[-3:]) if context_texts else ""
        context_prefix = f"###上文内容\n{context_str}\n" if context_str else ""
        
        # 【关键】使用与原方法相同的textarea标签格式
        user_prompt = f"""{context_prefix}###待翻译文本
<textarea>
{source_text}
</textarea>

###译文输出格式（必须严格遵守）
<textarea>
（在这里输出带序号的译文，每行一个序号）
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            # 等待RequestLimiter允许发送请求
            if not self._wait_for_limiter(messages, system_prompt):
                self.warning("  ⚠ RequestLimiter检查失败")
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 【调试】打印LLM原始响应（前1000字符）
                self.debug(f"  [调试] LLM原始响应（前1000字符）：\n{response_content[:1000]}")
                
                # 【关键】使用ResponseExtractor提取翻译结果（与原TranslatorTask完全相同）
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                
                # 【调试】打印解析后的字典
                if response_dict:
                    self.debug(f"  [调试] ResponseExtractor解析后字典键: {list(response_dict.keys())}")
                    self.debug(f"  [调试] 第一个译文示例: {list(response_dict.values())[0][:100] if response_dict else 'None'}...")
                else:
                    self.warning(f"  [调试] ResponseExtractor解析返回None或空字典")
                
                # 【关键】去除数字序号前缀（与原方法相同）
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                # 【关键】与原方法一致：只取我们需要的键，忽略多余的键
                # ResponseExtractor的generate_text_by_newlines会自动处理多余的译文
                if response_dict:
                    translated_texts = []
                    missing_keys = []
                    
                    for i in range(len(source_texts)):
                        key = str(i)
                        if key in response_dict:
                            translated_texts.append(response_dict[key])
                        else:
                            missing_keys.append(key)
                            translated_texts.append("")  # 缺失的键用空字符串填充
                    
                    # 【调试】如果译文数量不等于原文数量，打印详细信息
                    if len(response_dict) != len(source_texts):
                        self.debug(f"  [调试] 译文数量({len(response_dict)})≠原文数量({len(source_texts)})，已自动处理")
                        self.debug(f"  [调试] response_dict键: {list(response_dict.keys())}")
                        if len(response_dict) > len(source_texts):
                            extra_keys = [k for k in response_dict.keys() if int(k) >= len(source_texts)]
                            self.debug(f"  [调试] 多余的键(已忽略): {extra_keys}")
                    
                    if missing_keys:
                        self.warning(f"  ⚠ 部分译文缺失: 键{missing_keys}不存在")
                    
                    # 只要获取到了部分译文就返回（与原方法一致）
                    if any(translated_texts):
                        self.info(f"  ✓ 批量翻译成功: {len([t for t in translated_texts if t])} 行")
                        return translated_texts
                    else:
                        self.warning(f"  ⚠ 所有译文为空")
                        return None
                else:
                    self.warning(f"  ⚠ ResponseExtractor返回空字典")
                    return None
            else:
                self.warning("  ⚠ LLM返回为空或被跳过")
                return None
                
        except Exception as e:
            self.error(f"  ✗ 批量翻译失败: {e}", e)
            return None
    
    def _multi_step_translation(self, unit: Dict, terminology_db: Dict, memory_storage: Dict) -> Optional[str]:
        """
        多步骤引导翻译
        将翻译任务拆分为"理解—分解—转换—润色"阶段
        """
        self.info("  → 步骤1: 多步骤引导翻译...")
        
        source_text = unit["source_text"]
        context = unit.get("context", [])
        
        # 构建多步骤提示词（✅ 传递单个source_text作为列表）
        terminology_prompt = self._build_terminology_prompt(terminology_db, [source_text])
        memory_context = self._build_memory_context(memory_storage)
        
        system_prompt = f"""你是一位专业的翻译专家。请按照以下步骤进行翻译：

步骤1 - 理解：分析原文的语义、语境和风格
步骤2 - 分解：对于长难句，先识别主干成分和从句层级
步骤3 - 转换：将原文转换为目标语言，保持语义准确
步骤4 - 润色：优化译文，确保流畅自然

{terminology_prompt}
{memory_context}

请直接输出最终译文，不要输出中间步骤。"""
        
        # 构建上下文
        context_text = "\n".join(context[-3:]) if context else ""
        user_content = f"""请翻译以下文本：

上下文：
{context_text}

待翻译文本：
{source_text}"""
        
        messages = [{"role": "user", "content": user_content}]
        
        try:
            # 🔥 等待RequestLimiter允许发送请求
            if not self._wait_for_limiter(messages, system_prompt):
                self.warning("RequestLimiter检查失败或超时，跳过多步骤翻译")
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            
            # 调试日志：显示Agent翻译使用的配置（仅首次）
            if not hasattr(self, '_logged_config'):
                self._logged_config = True
                self.info("=" * 60)
                self.info("[Agent 翻译配置]")
                self.info(f"平台: {platform_config.get('target_platform', 'unknown')}")
                self.info(f"API URL: {platform_config.get('api_url', 'N/A')}")
                self.info(f"模型: {platform_config.get('model_name', 'N/A')}")
                self.info("=" * 60)
            
            skip, response_think, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 提取译文（去除可能的步骤标记）
                translated = self._extract_translation(response_content)
                self.info(f"  ✓ 初步译文: {translated[:100]}{'...' if len(translated) > 100 else ''}")
                return translated
        except Exception as e:
            self.error(f"多步骤翻译失败: {e}", e)
        
        return None
    
    def _multi_version_fusion(self, unit: Dict, initial_translation: str, 
                             terminology_db: Dict, memory_storage: Dict) -> Optional[str]:
        """
        多版本生成与融合
        生成多种风格的译文版本，然后智能评选与融合
        """
        self.info("  → 步骤2: 多版本生成与融合...")
        
        source_text = unit["source_text"]
        
        # 生成多个版本
        versions = {}
        
        # 版本1：直译版
        versions["literal"] = self._generate_version(source_text, initial_translation, "literal", terminology_db)
        
        # 版本2：意译版
        versions["free"] = self._generate_version(source_text, initial_translation, "free", terminology_db)
        
        # 版本3：风格化版（根据memory中的风格）
        style = memory_storage.get("style", "neutral")
        versions["stylized"] = self._generate_version(source_text, initial_translation, f"stylized_{style}", terminology_db)
        
        # 智能评选与融合
        best_version = self._select_and_fuse_versions(source_text, versions, terminology_db)
        
        # 保存版本信息
        self.translation_versions[unit["item_id"]] = versions
        
        if best_version:
            self.info(f"  ✓ 融合后译文: {best_version[:100]}{'...' if len(best_version) > 100 else ''}")
        
        return best_version
    
    def _generate_version(self, source_text: str, initial_translation: str, 
                        version_type: str, terminology_db: Dict) -> Optional[str]:
        """
        生成特定版本的翻译
        使用textarea格式和ResponseExtractor（与原方法相同）
        """
        version_prompts = {
            "literal": "请提供直译版本，尽可能贴近原文结构",
            "free": "请提供意译版本，注重流畅性和自然度",
            "stylized_formal": "请提供正式风格的翻译版本",
            "stylized_informal": "请提供非正式风格的翻译版本"
        }
        
        prompt_instruction = version_prompts.get(version_type, "请提供翻译版本")
        
        system_prompt = f"""你是一位专业的翻译专家。{prompt_instruction}。

{self._build_terminology_prompt(terminology_db, [source_text])}

【重要】输出格式要求：
- 必须使用<textarea>标签包裹译文
- 译文前必须加上序号"1."
- 不要添加任何额外的标题、前缀或说明文字
- 格式示例：<textarea>
1.译文内容
</textarea>"""
        
        # 【关键】使用textarea格式（单行）
        source_text_dict = {"0": source_text}
        user_prompt = f"""###待翻译文本
<textarea>
1.{source_text}
</textarea>

###译文输出格式（必须严格遵守）
<textarea>
1.
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return initial_translation
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 【关键】使用ResponseExtractor解析
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict and "0" in response_dict:
                    return response_dict["0"]
                else:
                    # 降级为简单提取
                    return self._extract_translation(response_content)
        except Exception as e:
            self.debug(f"生成{version_type}版本失败: {e}")
        
        return initial_translation
    
    def _select_and_fuse_versions(self, source_text: str, versions: Dict[str, str], 
                                  terminology_db: Dict) -> str:
        """
        智能评选与融合多个版本
        使用textarea格式和ResponseExtractor（与原方法相同）
        """
        versions_text = "\n".join([f"{k}: {v}" for k, v in versions.items() if v])
        
        system_prompt = f"""你是一位专业的翻译评估专家。请评估以下多个翻译版本，并融合生成最佳译文。

评估标准：
1. 语义准确性
2. 流畅性
3. 风格一致性
4. 术语使用规范性

【重要】输出格式要求：
- 必须使用<textarea>标签包裹译文
- 译文前必须加上序号"1."
- 不要添加任何额外的标题、前缀或说明文字"""
        
        # 【关键】使用textarea格式（单行）
        source_text_dict = {"0": source_text}
        user_prompt = f"""原文：
<textarea>
1.{source_text}
</textarea>

翻译版本：
{versions_text}

请评估并融合生成最佳译文：
<textarea>
1.
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return list(versions.values())[0] if versions else ""
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 【关键】使用ResponseExtractor解析
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict and "0" in response_dict:
                    return response_dict["0"]
                else:
                    fused = self._extract_translation(response_content)
                    return fused if fused else list(versions.values())[0]
        except Exception as e:
            self.debug(f"版本融合失败: {e}")
        
        return list(versions.values())[0] if versions else ""
    
    def _tear_verification(self, unit: Dict, translated_text: str, 
                          terminology_db: Dict) -> str:
        """
        回译验证与自我修正（TEaR框架）
        TEaR: Translate, Estimate, and Refine
        """
        self.info("  → 步骤3: 回译验证与自我修正...")
        
        source_text = unit["source_text"]
        
        # 步骤1: Estimate - 回译并评估
        back_translation = self._back_translate(translated_text)
        if back_translation:
            self.info(f"  ✓ 回译结果: {back_translation[:100]}{'...' if len(back_translation) > 100 else ''}")
        
        estimate_result = self._estimate_quality(source_text, translated_text, back_translation)
        
        # 步骤2: Refine - 如果发现问题，进行修正
        if estimate_result.get("needs_refinement", False):
            issues = estimate_result.get("issues", [])
            self.info(f"  ⚠ 发现质量问题: {', '.join(issues[:3])}")
            refined_text = self._refine_translation(source_text, translated_text, 
                                                   estimate_result, terminology_db)
            if refined_text:
                self.info(f"  ✓ 修正后译文: {refined_text[:100]}{'...' if len(refined_text) > 100 else ''}")
            return refined_text
        else:
            score = estimate_result.get("score", 0)
            self.info(f"  ✓ 质量评分: {score}/100 (无需修正)")
        
        return translated_text
    
    def _back_translate(self, translated_text: str) -> Optional[str]:
        """
        将译文回译到源语言
        使用textarea格式和ResponseExtractor（与原方法相同）
        """
        source_lang = self.config.source_language if self.config else "chinese"
        target_lang = self.config.target_language if self.config else "english"
        
        system_prompt = f"""请将以下{target_lang}文本回译为{source_lang}。

【重要】输出格式要求：
- 必须使用<textarea>标签包裹回译结果
- 回译结果前必须加上序号"1."
- 不要添加任何额外的标题、前缀或说明文字"""
        
        # 【关键】使用textarea格式（单行）
        source_text_dict = {"0": translated_text}
        user_prompt = f"""请回译以下文本：
<textarea>
1.{translated_text}
</textarea>

###回译输出格式（必须严格遵守）
<textarea>
1.
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 【关键】使用ResponseExtractor解析
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict and "0" in response_dict:
                    return response_dict["0"]
                else:
                    return self._extract_translation(response_content)
        except Exception as e:
            self.debug(f"回译失败: {e}")
        
        return None
    
    def _estimate_quality(self, source_text: str, translated_text: str, 
                         back_translation: Optional[str]) -> Dict[str, Any]:
        """
        评估翻译质量（TEaR的Estimate步骤）
        """
        if not back_translation:
            return {"needs_refinement": False, "issues": []}
        
        system_prompt = """你是一位专业的翻译质量评估专家。请比较原文和回译文，评估翻译质量。

评估维度：
1. 语义偏差
2. 逻辑错误
3. 信息遗漏
4. 术语一致性

请以JSON格式返回评估结果：
{
    "needs_refinement": true/false,
    "issues": ["问题1", "问题2"],
    "score": 0-100
}"""
        
        messages = [{
            "role": "user",
            "content": f"原文：{source_text}\n\n译文：{translated_text}\n\n回译文：{back_translation}\n\n请评估翻译质量："
        }]
        
        try:
            # 🔥 等待RequestLimiter允许发送请求
            if not self._wait_for_limiter(messages, system_prompt):
                self.debug("RequestLimiter检查失败，跳过质量评估")
                return {"needs_refinement": False, "issues": [], "score": 80}
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 尝试解析JSON
                try:
                    json_start = response_content.find("{")
                    json_end = response_content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_content[json_start:json_end]
                        result = json.loads(json_str)
                        return result
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            self.debug(f"质量评估失败: {e}")
        
        return {"needs_refinement": False, "issues": [], "score": 80}
    
    def _refine_translation(self, source_text: str, translated_text: str, 
                           estimate_result: Dict, terminology_db: Dict) -> str:
        """
        修正翻译（TEaR的Refine步骤）
        使用textarea格式和ResponseExtractor（与原方法相同）
        """
        issues = estimate_result.get("issues", [])
        issues_text = "\n".join(issues) if issues else "无明显问题"
        
        system_prompt = f"""你是一位专业的翻译修正专家。请根据评估结果修正以下译文。

评估发现的问题：
{issues_text}

{self._build_terminology_prompt(terminology_db, [source_text])}

【重要】输出格式要求：
- 必须使用<textarea>标签包裹修正后的译文
- 译文前必须加上序号"1."
- 不要添加任何额外的标题、前缀或说明文字"""
        
        # 【关键】使用textarea格式（单行）
        source_text_dict = {"0": source_text}
        user_prompt = f"""原文：
<textarea>
1.{source_text}
</textarea>

原译文：{translated_text}

请修正译文：
<textarea>
1.
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return translated_text
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                # 【关键】使用ResponseExtractor解析
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict and "0" in response_dict:
                    return response_dict["0"]
                else:
                    refined = self._extract_translation(response_content)
                return refined if refined else translated_text
        except Exception as e:
            self.error(f"翻译修正失败: {e}")
        
        return translated_text
    
    def _build_terminology_prompt(self, terminology_db: Dict, source_texts: List[str] = None) -> str:
        """
        构建术语表提示词（包含实体和专业术语）
        ✅ 【关键改进】采用原方法的动态筛选策略，只包含本批次文本中出现的术语
        
        Args:
            terminology_db: 完整术语库
            source_texts: 本批次的原文列表（用于筛选）
        
        Returns:
            术语表提示词
        """
        if not terminology_db:
            return ""
        
        # ✅ 【关键】动态筛选：只包含本批次文本中实际出现的术语
        if source_texts:
            # 将所有原文合并并转为小写（用于匹配）
            combined_text = " ".join(source_texts).lower()
            
            # 筛选出现在本批次中的术语
            filtered_terms = []
            for term, info in terminology_db.items():
                term_lower = term.lower()
                # 如果术语出现在本批次文本中
                if term_lower in combined_text:
                    translation = info.get("translation", "")
                    if translation:
                        filtered_terms.append({
                            "term": term,
                            "translation": translation,
                            "type": info.get("type", "term"),
                            "info": info.get("info", "")
                        })
            
            # 如果没有筛选到任何术语，返回空
            if not filtered_terms:
                return ""
        else:
            # 如果没有提供source_texts，使用所有术语（向后兼容）
            filtered_terms = []
            for term, info in list(terminology_db.items())[:50]:
                translation = info.get("translation", "")
                if translation:
                    filtered_terms.append({
                        "term": term,
                        "translation": translation,
                        "type": info.get("type", "term"),
                        "info": info.get("info", "")
                    })
        
        # ✅ 使用原方法的表格格式（更清晰）
        if self.config and self.config.target_language in ("chinese_simplified", "chinese_traditional"):
            prompt = "\n###术语表\n原文|译文|备注\n"
        else:
            prompt = "\n###Glossary\nOriginal Text|Translation|Remarks\n"
        
        # 添加筛选后的术语
        for item in filtered_terms:
            info_text = item["info"] if item["info"] else " "
            prompt += f"{item['term']}|{item['translation']}|{info_text}\n"
        
        return prompt
    
    def _build_terminology_prompt_for_backtranslation(self, terminology_db: Dict, translated_texts: List[str]) -> str:
        """
        构建回译用的术语表提示词
        ✅ 筛选标准：检查术语的**译文**是否在translated_texts中出现
        
        Args:
            terminology_db: 完整术语库
            translated_texts: 本批次的译文列表（用于筛选）
        
        Returns:
            术语表提示词（反向：译文→原文）
        """
        if not terminology_db or not translated_texts:
            return ""
        
        # 将所有译文合并并转为小写（用于匹配）
        combined_text = " ".join(translated_texts).lower()
        
        # 筛选：检查术语的translation是否出现在译文中
        filtered_terms = []
        for term, info in terminology_db.items():
            translation = info.get("translation", "")
            if translation and translation.lower() in combined_text:
                filtered_terms.append({
                    "term": term,
                    "translation": translation,
                    "info": info.get("info", "")
                })
        
        if not filtered_terms:
            return ""
        
        # ✅ 使用表格格式（反向：译文→原文，用于回译）
        if self.config and self.config.target_language in ("chinese_simplified", "chinese_traditional"):
            prompt = "\n###术语表（回译参考）\n译文|原文|备注\n"
        else:
            prompt = "\n###Glossary (Back-translation Reference)\nTranslation|Original Text|Remarks\n"
        
        # 添加筛选后的术语（注意顺序：译文在前，原文在后）
        for item in filtered_terms:
            info_text = item["info"] if item["info"] else " "
            prompt += f"{item['translation']}|{item['term']}|{info_text}\n"
        
        return prompt
    
    def _build_memory_context(self, memory_storage: Dict) -> str:
        """构建Memory上下文"""
        context_parts = []
        
        domain = memory_storage.get("domain", "")
        style = memory_storage.get("style", "")
        
        if domain:
            context_parts.append(f"文本领域：{domain}")
        if style:
            context_parts.append(f"文本风格：{style}")
        
        return "\n".join(context_parts) if context_parts else ""
    
    def _extract_translation(self, response: str) -> str:
        """从LLM响应中提取译文"""
        # 去除可能的标记和说明
        lines = response.strip().split("\n")
        # 取第一行或最长的行作为译文
        translation = max(lines, key=len).strip()
        # 去除可能的引号
        translation = translation.strip('"').strip("'")
        return translation
    
    def _update_cache_item(self, item: CacheItem, translated_text: str) -> None:
        """更新缓存项"""
        item.translated_text = translated_text
        item.translation_status = TranslationStatus.TRANSLATED
    
    def _wait_for_limiter(self, messages: list, system_prompt: str, timeout: int = 300) -> bool:
        """
        等待RequestLimiter允许发送请求（参考原TaskExecutor的实现）
        
        Args:
            messages: 消息列表
            system_prompt: 系统提示词
            timeout: 超时时间（秒）
            
        Returns:
            True if 可以发送请求, False if 超时
        """
        import time
        from Base.Base import Base
        
        # 计算Token消耗
        tokens_consume = self.request_limiter.calculate_tokens(messages, system_prompt)
        
        # 等待限制器允许
        start_time = time.time()
        while True:
            # 检测是否收到停止翻译事件
            if Base.work_status == Base.STATUS.STOPING:
                return False
            
            # 检查是否超时
            if time.time() - start_time >= timeout:
                self.warning(f"等待RequestLimiter超时（{timeout}秒），跳过当前请求")
                return False
            
            # 检查RPM和TPM限制
            if self.request_limiter.check_limiter(tokens_consume):
                return True
            
            # 如果以上条件都不符合，则间隔1秒再次检查
            time.sleep(1)
    
    def _batch_multi_version_fusion(self, source_texts: List[str], initial_translations: List[str],
                                   terminology_db: Dict, memory_storage: Dict) -> Optional[List[str]]:
        """
        批量多版本生成与融合
        使用与批量翻译相同的textarea格式，分别批量生成3个版本，然后批量融合
        """
        self.info(f"    → 批量生成3个版本（直译/意译/风格化）...")
        
        # ===== 批量生成版本1：直译版 =====
        literal_versions = self._batch_generate_version(source_texts, initial_translations, "literal", terminology_db)
        if not literal_versions:
            self.warning("    ⚠ 直译版本批量生成失败")
            literal_versions = initial_translations
        
        # ===== 批量生成版本2：意译版 =====
        free_versions = self._batch_generate_version(source_texts, initial_translations, "free", terminology_db)
        if not free_versions:
            self.warning("    ⚠ 意译版本批量生成失败")
            free_versions = initial_translations
        
        # ===== 批量生成版本3：风格化版 =====
        style = memory_storage.get("style", "neutral")
        stylized_versions = self._batch_generate_version(source_texts, initial_translations, f"stylized_{style}", terminology_db)
        if not stylized_versions:
            self.warning("    ⚠ 风格化版本批量生成失败")
            stylized_versions = initial_translations
        
        self.info(f"    ✓ 3个版本批量生成完成")
        
        # ===== 批量智能融合 =====
        self.info(f"    → 批量智能融合3个版本...")
        fused_texts = self._batch_fuse_versions(
            source_texts, literal_versions, free_versions, stylized_versions, terminology_db
        )
        
        if not fused_texts:
            self.warning("    ⚠ 批量融合失败，使用初始译文")
            return initial_translations
        
        self.info(f"    ✓ 批量融合完成: {len(fused_texts)} 行")
        return fused_texts
    
    def _batch_generate_version(self, source_texts: List[str], initial_translations: List[str],
                               version_type: str, terminology_db: Dict) -> Optional[List[str]]:
        """
        批量生成特定版本的翻译（使用textarea格式）
        """
        version_prompts = {
            "literal": "请提供直译版本，尽可能贴近原文结构",
            "free": "请提供意译版本，注重流畅性和自然度",
            "stylized_formal": "请提供正式风格的翻译版本",
            "stylized_informal": "请提供非正式风格的翻译版本",
            "stylized_neutral": "请提供中性风格的翻译版本"
        }
        
        prompt_instruction = version_prompts.get(version_type, "请提供翻译版本")
        
        system_prompt = f"""你是一位专业的翻译专家。{prompt_instruction}。

{self._build_terminology_prompt(terminology_db, [source_text])}

【重要】输出格式要求：
- 必须使用<textarea>标签包裹所有译文
- 每行译文前必须加上序号（如1. 2. 3.）
- 不要添加任何额外的标题、前缀或说明文字"""
        
        # 构建source_text_dict（与批量翻译相同）
        source_text_dict = {str(i): text for i, text in enumerate(source_texts)}
        
        # 构建带序号的原文（与批量翻译相同）
        numbered_lines = []
        for index, line in enumerate(source_texts):
            if "\n" in line:
                lines = line.split("\n")
                numbered_text = f"{index + 1}.[\n"
                total_lines = len(lines)
                for sub_index, sub_line in enumerate(lines):
                    sub_line = sub_line[:-1] if re.match(r'.*[^ ] $', sub_line) else sub_line
                    numbered_text += f'"{index + 1}.{total_lines - sub_index}.,{sub_line}",\n'
                numbered_text = numbered_text.rstrip('\n').rstrip(',')
                numbered_text += f"\n]"
                numbered_lines.append(numbered_text)
            else:
                numbered_lines.append(f"{index + 1}.{line}")
        
        source_text = "\n".join(numbered_lines)
        
        # 构建user_prompt
        user_prompt = f"""###待翻译文本
<textarea>
{source_text}
</textarea>

###译文输出格式（必须严格遵守）
<textarea>
（在这里输出带序号的译文）
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict:
                    translated_texts = []
                    for i in range(len(source_texts)):
                        key = str(i)
                        if key in response_dict:
                            translated_texts.append(response_dict[key])
                        else:
                            translated_texts.append("")
                    
                    if any(translated_texts):
                        return translated_texts
        except Exception as e:
            self.debug(f"批量生成{version_type}版本失败: {e}")
        
        return None
    
    def _batch_fuse_versions(self, source_texts: List[str], literal_versions: List[str],
                           free_versions: List[str], stylized_versions: List[str],
                           terminology_db: Dict) -> Optional[List[str]]:
        """
        批量智能融合多个版本（使用textarea格式）
        """
        system_prompt = f"""你是一位专业的翻译评估专家。请评估以下多个翻译版本，并融合生成最佳译文。

评估标准：
1. 语义准确性
2. 流畅性
3. 风格一致性
4. 术语使用规范性

【重要】输出格式要求：
- 必须使用<textarea>标签包裹所有译文
- 每行译文前必须加上序号（如1. 2. 3.）
- 不要添加任何额外的标题、前缀或说明文字"""
        
        # 构建批量融合的输入
        source_text_dict = {str(i): text for i, text in enumerate(source_texts)}
        
        # 构建带版本的原文
        numbered_blocks = []
        for i, (src, lit, free, sty) in enumerate(zip(source_texts, literal_versions, free_versions, stylized_versions)):
            block = f"""{i + 1}.原文: {src}
   直译版: {lit}
   意译版: {free}
   风格化版: {sty}"""
            numbered_blocks.append(block)
        
        versions_text = "\n\n".join(numbered_blocks)
        
        user_prompt = f"""###多版本翻译结果
{versions_text}

###请评估并融合，输出最佳译文
<textarea>
（在这里输出带序号的最佳译文）
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, _, _ = self.llm_requester.sent_request(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            if not skip and response_content:
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict:
                    fused_texts = []
                    for i in range(len(source_texts)):
                        key = str(i)
                        if key in response_dict:
                            fused_texts.append(response_dict[key])
                        else:
                            fused_texts.append("")
                    
                    if any(fused_texts):
                        return fused_texts
        except Exception as e:
            self.debug(f"批量融合失败: {e}")
        
        return None
    
    def _unified_human_review(self, all_chunks_data: List[Dict], terminology_db: Dict, task_memory: Dict, workflow_config: Dict = None) -> List[Dict]:
        """
        统一人工审核（所有批次完成后）
        
        Args:
            all_chunks_data: 所有批次的数据列表
            terminology_db: 术语库
            task_memory: 任务元数据
            workflow_config: 工作流配置
            
        Returns:
            更新后的批次数据列表
        """
        if workflow_config is None:
            workflow_config = {}
            
        enable_human_review = workflow_config.get("enable_human_review", False)
        review_threshold = workflow_config.get("review_threshold", 0.8)
        
        self.info(f"人工审核检查: enable_human_review={enable_human_review}, callback={self._human_intervention_callback is not None}")
        
        if not enable_human_review or not self._human_intervention_callback:
            self.info("    ⚠ 人工审核未启用或回调不可用，跳过人工审核")
            return all_chunks_data
        
        # 🔥 收集全文的评分数据
        all_scores_with_index = []  # [(global_index, chunk_idx, local_idx, score, source, translated, back)]
        global_index = 0
        
        for chunk_idx, chunk_data in enumerate(all_chunks_data):
            source_texts = chunk_data.get("source_texts", [])
            translated_texts = chunk_data.get("translated_texts", [])
            quality_scores = chunk_data.get("quality_scores", [])
            back_translations = chunk_data.get("back_translations", [])
            chunk_items = chunk_data.get("chunk_items", []) # 获取原始CacheItem列表
            
            file_path = chunk_data.get("file_path", "") # 🔥 从 chunk_data 获取 file_path
            
            for local_idx, (src, trans, score, back) in enumerate(zip(source_texts, translated_texts, quality_scores, back_translations)):
                # 获取对应的CacheItem信息
                row_index = -1
                if local_idx < len(chunk_items):
                    # file_path = chunk_items[local_idx].file_path # CacheItem 没有 file_path
                    row_index = chunk_items[local_idx].row_index

                all_scores_with_index.append({
                    "global_index": global_index,
                    "chunk_idx": chunk_idx,
                    "local_idx": local_idx,
                    "score": score,
                    "source_text": src,
                    "translated_text": trans,
                    "back_translation": back,
                    "file_path": file_path,
                    "row_index": row_index
                })
                global_index += 1
        
        # 🔥 按评分排序，找出最低的3个
        all_scores_with_index.sort(key=lambda x: x["score"])
        lowest_3 = all_scores_with_index[:min(3, len(all_scores_with_index))]
        
        self.info(f"    🔔 启用人工审核：展示评分最低的3个供审核（测试模式）")
        
        # 准备审核项目
        review_items = []
        for item in lowest_3:
            # 准备上下文（前后各1行）
            global_idx = item["global_index"]
            context_before = all_scores_with_index[global_idx - 1]["source_text"] if global_idx > 0 else ""
            context_after = all_scores_with_index[global_idx + 1]["source_text"] if global_idx < len(all_scores_with_index) - 1 else ""
            
            review_items.append({
                "index": global_idx,  # 全局索引
                "chunk_idx": item["chunk_idx"],
                "local_idx": item["local_idx"],
                "source_text": item["source_text"],
                "translated_text": item["translated_text"],
                "back_translation": item["back_translation"],
                "score": item["score"],
                "context_before": context_before,
                "context_after": context_after,
                "file_path": item.get("file_path"),
                "row_index": item.get("row_index")
            })
            self.info(f"      全局行{global_idx+1}: 评分 {item['score']:.1f}/10（最低3个之一）")
        
        # 调用人工审核
        review_result = self._human_intervention_callback(
            "batch_translation_review",
            {
                "review_items": review_items,
                "review_threshold": review_threshold
            }
        )
        
        # 处理人工审核结果
        if review_result and review_result.get("review_results"):
            self.info(f"    ✓ 人工审核完成，应用用户决策...")
            
            for user_decision in review_result["review_results"]:
                global_idx = user_decision["index"]
                action = user_decision["action"]
                
                # 找到对应的批次和本地索引
                target_item = None
                for item in review_items:
                    if item["index"] == global_idx:
                        target_item = item
                        break
                
                if not target_item:
                    continue
                
                chunk_idx = target_item["chunk_idx"]
                local_idx = target_item["local_idx"]
                chunk_data = all_chunks_data[chunk_idx]
                
                if action == "accept":
                    self.info(f"      全局行{global_idx+1}: 用户接受 ✅")
                    # 无需修改
                    
                elif action == "custom":
                    # 使用用户自定义翻译
                    custom_translation = user_decision["translation"]
                    chunk_data["translated_texts"][local_idx] = custom_translation
                    self.info(f"      全局行{global_idx+1}: 使用用户自定义翻译 ✏️")
                    
                    # [DB] 记录人工修改 (Human Edit Trace)
                    if self.db_manager and self._current_cache_project:
                        atom_id = self._get_atom_id(self._current_cache_project, target_item["file_path"], target_item["row_index"])
                        if atom_id:
                            self.db_manager.add_trace(
                                atom_id=atom_id,
                                agent_role="Human",
                                action_type="human_edit",
                                content=custom_translation,
                                meta_data={"note": "User custom translation in review dialog"}
                            )
                            # 更新原子翻译结果
                            self.db_manager.update_atom_translation(
                                atom_id=atom_id,
                                translated_text=custom_translation,
                                status_code=3  # 已人工审核
                            )
                    
                elif action == "retranslate":
                    # 需要LLM重新翻译
                    self.info(f"      全局行{global_idx+1}: 标记为需要重新翻译 🔄")
                    source_text = target_item["source_text"]
                    # 简化：直接使用单行翻译
                    new_translation = self._translate_single_line(
                        source_text, [], "literal", terminology_db, {}
                    )
                    if new_translation:
                        chunk_data["translated_texts"][local_idx] = new_translation
                        self.info(f"      全局行{global_idx+1}: 重新翻译完成")
                        
                        # [DB] 记录重新翻译 (Refine Trace)
                        if self.db_manager and self._current_cache_project:
                            atom_id = self._get_atom_id(self._current_cache_project, target_item["file_path"], target_item["row_index"])
                            if atom_id:
                                self.db_manager.add_trace(
                                    atom_id=atom_id,
                                    agent_role="Translator",
                                    action_type="refine",
                                    content=new_translation,
                                    meta_data={"reason": "User requested retranslation"}
                                )
                                # 更新原子翻译结果
                                self.db_manager.update_atom_translation(
                                    atom_id=atom_id,
                                    translated_text=new_translation,
                                    status_code=2  # 已润色
                                )
                    else:
                        self.warning(f"      全局行{global_idx+1}: 重新翻译失败，保留原译文")
        else:
            self.info(f"    ⚠ 用户取消审核，继续自动流程")
        
        return all_chunks_data
    
    def _unified_entity_check(self, all_chunks_data: List[Dict], terminology_db: Dict, entity_database: Dict) -> List[Dict]:
        """
        统一实体一致性检查（人工审核后）
        
        Args:
            all_chunks_data: 所有批次的数据列表
            terminology_db: 术语库
            entity_database: 实体数据库
            
        Returns:
            更新后的批次数据列表
        """
        for chunk_idx, chunk_data in enumerate(all_chunks_data):
            source_texts = chunk_data.get("source_texts", [])
            translated_texts = chunk_data.get("translated_texts", [])
            
            self.info(f"  → 检查批次 {chunk_idx+1}/{len(all_chunks_data)}...")
            
            # 🔥 发送UI阶段更新：一致性检查阶段（仅第一次）
            if chunk_idx == 0 and self._current_cache_project and not hasattr(self, '_entity_check_stage_sent'):
                self._update_stage_progress(self._current_cache_project, "entity_check", 0, 1)
                self._publish_stage_with_stats(self._current_cache_project, "entity_check", "检查中")
                self._entity_check_stage_sent = True
            
            checked_texts = self._check_entity_consistency(
                source_texts, translated_texts, terminology_db, entity_database
            )
            
            # [DB] Phase 3: 记录最终一致性检查轨迹 (Final Trace)
            # 仅记录被实体检查修改过的行
            if self.db_manager and self._current_cache_project:
                chunk_items = chunk_data.get("chunk_items", [])
                file_path = chunk_data.get("file_path", "")
                
                for idx, (old_text, new_text) in enumerate(zip(translated_texts, checked_texts)):
                    if old_text != new_text and idx < len(chunk_items):
                        item = chunk_items[idx]
                        atom_id = self._get_atom_id(self._current_cache_project, file_path, item.row_index)
                        if atom_id:
                            self.db_manager.add_trace(
                                atom_id=atom_id,
                                agent_role="ConsistencyChecker",
                                action_type="final",
                                content=new_text,
                                meta_data={"reason": "Entity consistency check", "before": old_text}
                            )
                            # 更新原子为最终译文
                            self.db_manager.update_atom_translation(
                                atom_id=atom_id,
                                translated_text=new_text,
                                status_code=4  # 已完成
                            )

            chunk_data["translated_texts"] = checked_texts
        
        # 完成一致性检查
        if self._current_cache_project:
            self._update_stage_progress(self._current_cache_project, "entity_check", 1, 1)
        
        self.info(f"  ✓ 所有批次实体一致性检查完成")
        return all_chunks_data
    
    def _batch_tear_verification_with_scores(self, source_texts: List[str], translated_texts: List[str],
                                terminology_db: Dict) -> tuple[Optional[List[str]], List[float], List[str]]:
        """
        批量回译验证（返回评分数据，不进行人工审核）
        
        Returns:
            (verified_texts, quality_scores, back_translations)
        """
        # 🔥 发送UI阶段更新：回译评估阶段（仅第一次）
        if self._current_cache_project and not hasattr(self, '_backtranslation_stage_sent'):
            self._update_stage_progress(self._current_cache_project, "backtranslation", 0, 3)
            self._publish_stage_with_stats(self._current_cache_project, "backtranslation", "回译中")
            self._backtranslation_stage_sent = True
        
        # ===== 批量回译 =====
        self.info(f"    → 批量回译...")
        self._update_stage_progress(self._current_cache_project, "backtranslation", 1, 3)
        back_translations = self._batch_back_translate(translated_texts, terminology_db)
        
        if not back_translations:
            self.warning("    ⚠ 批量回译失败，跳过TEaR验证")
            return translated_texts, [8.0] * len(translated_texts), [""] * len(translated_texts)
        
        self.info(f"    ✓ 批量回译完成: {len(back_translations)} 行")
        
        # ===== 批量质量评估 =====
        self.info(f"    → 批量质量评估...")
        self._update_stage_progress(self._current_cache_project, "backtranslation", 2, 3)
        needs_refinement, quality_scores = self._batch_estimate_quality(source_texts, translated_texts, back_translations)
        
        # 统计需要修正的数量
        refine_count = sum(1 for need in needs_refinement if need)
        
        # ===== 批量修正（仅对需要修正的） =====
        self._update_stage_progress(self._current_cache_project, "backtranslation", 3, 3)
        if refine_count == 0:
            self.info(f"    ✓ 所有译文质量良好，无需修正")
            return translated_texts, quality_scores, back_translations
        
        self.info(f"    → 批量修正 {refine_count} 行...")
        refined_texts = self._batch_refine_translation(
            source_texts, translated_texts, back_translations, needs_refinement, terminology_db
        )
        
        if not refined_texts:
            self.warning("    ⚠ 批量修正失败，使用原译文")
            return translated_texts, quality_scores, back_translations
        
        self.info(f"    ✓ 批量修正完成")
        return refined_texts, quality_scores, back_translations
    
    def _batch_tear_verification(self, source_texts: List[str], translated_texts: List[str],
                                terminology_db: Dict, human_intervention_callback=None) -> Optional[List[str]]:
        """
        批量回译验证（TEaR: Translate, Estimate, and Refine）
        
        Args:
            source_texts: 原文列表
            translated_texts: 译文列表
            terminology_db: 术语库
            human_intervention_callback: 人工介入回调函数
        """
        # 🔥 发送UI阶段更新：回译评估阶段（仅第一次）
        if self._current_cache_project and not hasattr(self, '_backtranslation_stage_sent'):
            # 回译阶段：总共3步（回译、评估、修正）
            self._update_stage_progress(self._current_cache_project, "backtranslation", 0, 3)
            self._publish_stage_with_stats(self._current_cache_project, "backtranslation", "回译中")
            self._backtranslation_stage_sent = True
        
        # ===== 批量回译 =====
        self.info(f"    → 批量回译...")
        self._update_stage_progress(self._current_cache_project, "backtranslation", 1, 3)  # 第1步
        back_translations = self._batch_back_translate(translated_texts, terminology_db)
        
        if not back_translations:
            self.warning("    ⚠ 批量回译失败，跳过TEaR验证")
            return translated_texts
        
        self.info(f"    ✓ 批量回译完成: {len(back_translations)} 行")
        
        # ===== 批量质量评估 =====
        self.info(f"    → 批量质量评估...")
        self._update_stage_progress(self._current_cache_project, "backtranslation", 2, 3)  # 第2步
        needs_refinement, quality_scores = self._batch_estimate_quality(source_texts, translated_texts, back_translations)
        
        # 统计需要修正的数量
        refine_count = sum(1 for need in needs_refinement if need)
        # 注意：详细评分已在 _batch_estimate_quality 中显示
        
        # ===== 人在回路：人工审核（如果启用） =====
        if human_intervention_callback:
            # 获取工作流配置
            workflow_config = {}
            if hasattr(self, '_workflow_state') and self._workflow_state:
                workflow_config = self._workflow_state.get("workflow_config", {})
            
            enable_human_review = workflow_config.get("enable_human_review", False)
            review_threshold = workflow_config.get("review_threshold", 0.8)
            
            if enable_human_review:
                # 准备需要审核的项目
                review_items = []
                
                if refine_count > 0:
                    # 情况1：有评分低于7.0的行，审核这些行
                    self.info(f"    🔔 启用人工审核：{refine_count} 行评分低于阈值 {review_threshold*10:.0f}/10")
                    
                    for i, (need, score) in enumerate(zip(needs_refinement, quality_scores)):
                        if need:  # score < 7.0
                            # 准备上下文（前后各1行）
                            context_before = source_texts[i-1] if i > 0 else ""
                            context_after = source_texts[i+1] if i < len(source_texts) - 1 else ""
                            
                            review_items.append({
                                "index": i,
                                "source_text": source_texts[i],
                                "translated_text": translated_texts[i],
                                "back_translation": back_translations[i],
                                "score": score,
                                "context_before": context_before,
                                "context_after": context_after
                            })
                else:
                    # 情况2：所有评分都很高，但为了测试/展示，选择评分最低的3个
                    self.info(f"    🔔 启用人工审核：所有译文评分良好，展示评分最低的3个供审核（测试模式）")
                    
                    # 按评分排序，找出最低的3个
                    scored_items = [(i, score) for i, score in enumerate(quality_scores)]
                    scored_items.sort(key=lambda x: x[1])  # 按评分从低到高排序
                    lowest_3 = scored_items[:min(3, len(scored_items))]
                    
                    for i, score in lowest_3:
                        context_before = source_texts[i-1] if i > 0 else ""
                        context_after = source_texts[i+1] if i < len(source_texts) - 1 else ""
                        
                        review_items.append({
                            "index": i,
                            "source_text": source_texts[i],
                            "translated_text": translated_texts[i],
                            "back_translation": back_translations[i],
                            "score": score,
                            "context_before": context_before,
                            "context_after": context_after
                        })
                        self.info(f"      行{i+1}: 评分 {score:.1f}/10（最低3个之一）")
                
                if review_items:
                    # 调用人工审核
                    review_result = human_intervention_callback(
                        "batch_translation_review",
                        {
                            "review_items": review_items,
                            "review_threshold": review_threshold
                        }
                    )
                    
                    # 处理人工审核结果
                    if review_result and review_result.get("review_results"):
                        translated_texts = self._apply_human_review_results(
                            source_texts, translated_texts, back_translations, 
                            needs_refinement, quality_scores, review_result["review_results"],
                            terminology_db
                        )
                        self.info(f"    ✓ 人工审核完成，已应用用户决策")
                        self._update_stage_progress(self._current_cache_project, "backtranslation", 3, 3)
                        return translated_texts
                    else:
                        self.info(f"    ⚠ 用户取消审核，继续自动流程")
        
        # ===== 批量修正（仅对需要修正的） =====
        self._update_stage_progress(self._current_cache_project, "backtranslation", 3, 3)  # 第3步
        if refine_count == 0:
            self.info(f"    ✓ 所有译文质量良好，无需修正")
            return translated_texts
        
        self.info(f"    → 批量修正 {refine_count} 行...")
        refined_texts = self._batch_refine_translation(
            source_texts, translated_texts, back_translations, needs_refinement, terminology_db
        )
        
        if not refined_texts:
            self.warning("    ⚠ 批量修正失败，使用原译文")
            return translated_texts
        
        self.info(f"    ✓ 批量修正完成")
        return refined_texts
    
    def _apply_human_review_results(self, source_texts: List[str], translated_texts: List[str],
                                    back_translations: List[str], needs_refinement: List[bool],
                                    quality_scores: List[float], review_results: List[Dict],
                                    terminology_db: Dict) -> List[str]:
        """
        应用人工审核结果
        
        Args:
            source_texts: 原文列表
            translated_texts: 译文列表
            back_translations: 回译列表
            needs_refinement: 需要修正的标记
            quality_scores: 质量评分列表
            review_results: 用户审核结果 [{"index": int, "action": str, "translation": str/None}]
            terminology_db: 术语库
            
        Returns:
            更新后的译文列表
        """
        result_texts = list(translated_texts)  # 复制译文列表
        
        # 构建索引到审核结果的映射
        review_map = {item["index"]: item for item in review_results}
        
        # 收集需要LLM重新翻译的行
        to_retranslate_indices = []
        
        for i, need in enumerate(needs_refinement):
            if not need:  # 不需要修正的行，保持原样
                continue
            
            if i in review_map:
                review = review_map[i]
                action = review["action"]
                
                if action == "accept":
                    # 用户接受当前译文，无需修改
                    self.info(f"      行{i+1}: 用户接受 ✅")
                    pass
                
                elif action == "custom":
                    # 用户提供了自定义翻译
                    custom_translation = review["translation"]
                    result_texts[i] = custom_translation
                    self.info(f"      行{i+1}: 使用用户自定义翻译 ✏️")
                
                elif action == "retranslate":
                    # 需要LLM重新翻译
                    to_retranslate_indices.append(i)
                    self.info(f"      行{i+1}: 标记为需要重新翻译 🔄")
        
        # 批量重新翻译
        if to_retranslate_indices:
            self.info(f"    → 正在重新翻译 {len(to_retranslate_indices)} 行...")
            
            to_retranslate_sources = [source_texts[i] for i in to_retranslate_indices]
            to_retranslate_translations = [translated_texts[i] for i in to_retranslate_indices]
            to_retranslate_backs = [back_translations[i] for i in to_retranslate_indices]
            
            # 使用_batch_refine_translation重新翻译
            needs_refinement_subset = [True] * len(to_retranslate_indices)
            refined_subset = self._batch_refine_translation(
                to_retranslate_sources,
                to_retranslate_translations,
                to_retranslate_backs,
                needs_refinement_subset,
                terminology_db
            )
            
            if refined_subset:
                # 将重新翻译的结果放回原位置
                for idx, refined_text in zip(to_retranslate_indices, refined_subset):
                    result_texts[idx] = refined_text
                self.info(f"    ✓ 重新翻译完成")
            else:
                self.warning(f"    ⚠ 重新翻译失败，保留原译文")
        
        return result_texts
    
    def _extract_by_line_number(self, text: str) -> Dict[int, str]:
        """
        鲁棒的按行号提取文本 (辅助方法)
        解决ResponseExtractor按顺序提取导致的错位问题
        """
        results = {}
        # 移除textarea标签
        text = re.sub(r'<textarea.*?>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'</textarea>', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        # 按 "换行+数字+点/顿号" 分割
        # 使用split保留分隔符中的数字
        parts = re.split(r'(?:^|\n)\s*(\d+)[.、]\s*', text)
        
        # split结果: [前缀, 数字1, 内容1, 数字2, 内容2, ...]
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                try:
                    num_str = parts[i]
                    content = parts[i+1].strip()
                    idx = int(num_str) - 1 # 转为0-based索引
                    if idx >= 0:
                        results[idx] = content
                except:
                    pass
        return results
    
    def _batch_back_translate(self, translated_texts: List[str], terminology_db: Dict) -> Optional[List[str]]:
        """
        批量回译（使用textarea格式）
        
        Args:
            translated_texts: 译文列表
            terminology_db: 术语库（确保回译时使用相同的实体翻译）
        
        Returns:
            回译结果列表
        """
        source_lang = self.config.source_language if self.config else "chinese"
        target_lang = self.config.target_language if self.config else "english"
        
        # 构建术语提示（✅ 传递translated_texts用于筛选，检查译文中是否包含术语的翻译）
        terminology_prompt = self._build_terminology_prompt_for_backtranslation(terminology_db, translated_texts)
        
        system_prompt = f"""你是一位专业的回译专家。请将以下{target_lang}文本精确回译为{source_lang}。

{terminology_prompt}

🔥【强制要求-术语表严格遵守】🔥
- 术语表中列出的所有译文，必须回译为对应的原文术语，绝不允许替换或改写
- 这是强制性要求，不可违反
- 例如：如果术语表规定"磷脂酰肌醇"必须回译为"phosphatidylinositol"，则绝对不能回译为"phospholipid inositol"或其他任何变体
- 例如：如果术语表规定"Beclin"必须回译为"Beclin"，则必须保持不变
- 例如：如果术语表规定"自噬"必须回译为"autophagy"，则绝对不能回译为"self-phagocytosis"或其他任何替代词
- 术语表的回译规则优先级最高，高于任何语言习惯或同义词

【回译目的】
- 回译是为了验证正向翻译的准确性
- 如果回译无法还原原文术语，说明正向翻译可能有误
- 因此，术语的回译必须100%准确

【重要】输出格式要求：
- 必须使用<textarea>标签包裹所有回译结果
- 每行回译前必须加上序号（如1. 2. 3.）
- 不要添加任何额外的标题、前缀或说明文字"""
        
        source_text_dict = {str(i): text for i, text in enumerate(translated_texts)}
        
        # 构建带序号的译文
        numbered_lines = []
        for index, line in enumerate(translated_texts):
            if "\n" in line:
                lines = line.split("\n")
                numbered_text = f"{index + 1}.[\n"
                total_lines = len(lines)
                for sub_index, sub_line in enumerate(lines):
                    sub_line = sub_line[:-1] if re.match(r'.*[^ ] $', sub_line) else sub_line
                    numbered_text += f'"{index + 1}.{total_lines - sub_index}.,{sub_line}",\n'
                numbered_text = numbered_text.rstrip('\n').rstrip(',')
                numbered_text += f"\n]"
                numbered_lines.append(numbered_text)
            else:
                numbered_lines.append(f"{index + 1}.{line}")
        
        translated_text = "\n".join(numbered_lines)
        
        user_prompt = f"""###请回译以下文本
<textarea>
{translated_text}
</textarea>

###回译输出格式（必须严格遵守）
<textarea>
（在这里输出带序号的回译结果）
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 🔥 使用鲁棒的按行号提取，解决错位问题
                extracted_map = self._extract_by_line_number(response_content)
                
                if extracted_map:
                    back_translations = []
                    for i in range(len(translated_texts)):
                        # 按索引获取，如果缺失则为空
                        back_translations.append(extracted_map.get(i, ""))
                    
                    if any(back_translations):
                        return back_translations
        except Exception as e:
            self.debug(f"批量回译失败: {e}")
        
        return None
    
    def _batch_estimate_quality(self, source_texts: List[str], translated_texts: List[str],
                               back_translations: List[str]) -> tuple[List[bool], List[float]]:
        """
        🔥 批量评估翻译质量（带详细评分）
        返回: (needs_refinement, quality_scores)
            - needs_refinement: List[bool]，每个元素表示对应行是否需要修正
            - quality_scores: List[float]，每个元素表示对应行的质量评分(1-10)
        """
        system_prompt = """你是一位专业的翻译质量评估专家。请比较原文和回译文，为每行翻译打分（1-10分）。

评估维度：
1. 语义准确性（40%）：回译是否准确还原了原文的含义
2. 信息完整性（30%）：回译是否保留了原文的所有关键信息
3. 逻辑一致性（20%）：回译是否逻辑通顺，无矛盾
4. 整体流畅性（10%）：回译是否自然流畅

🔥【重要评估原则-术语容忍】🔥
- 如果回译与原文的差异仅在于专有名词、术语的表述不同，但语义等价，应给予高分
- 例如：原文"phosphatidylinositol"，回译为"phospholipid inositol"，虽然词不同，但语义相近，应给8-9分
- 例如：原文"autophagy"，回译为"self-eating"，虽然词不同，但都指代自噬，应给8-9分
- 术语的精确性由正向翻译的术语表保证，回译只需验证语义准确性

【评分标准】
- 9-10分：完美，语义准确，信息完整
- 8分：优秀，仅有微小的术语表述差异
- 7分：良好，语义基本准确，有小瑕疵
- 6分：及格，语义大致正确，但有明显不足
- 5分以下：需要修正，存在语义偏差、信息遗漏或逻辑错误

【重要】输出格式要求：
- 必须使用<textarea>标签包裹所有评估结果
- 每行格式：序号. 评分：X.X（如：1. 评分：9.5 或 2. 评分：7.0）
- 评分必须是1.0到10.0之间的数字，必须包含小数点
- 不要输出"0"或"0.0"这样的无效评分
- 不要添加"分"字或其他说明文字"""
        
        # 构建批量评估的输入
        comparison_blocks = []
        for i, (src, trans, back) in enumerate(zip(source_texts, translated_texts, back_translations)):
            block = f"{i + 1}.原文: {src[:100]}{'...' if len(src) > 100 else ''}\n   回译: {back[:100]}{'...' if len(back) > 100 else ''}"
            comparison_blocks.append(block)
        
        comparison_text = "\n\n".join(comparison_blocks)
        
        user_prompt = f"""###翻译质量评估（为每行打分1-10分）
{comparison_text}

###评估结果输出格式（必须严格遵守）
<textarea>
1. 评分：9.5
2. 评分：8.0
3. 评分：7.5
4. 评分：6.0
5. 评分：9.0
...（按此格式输出所有行的评分）
</textarea>

【重要提示】
- 每行必须包含"评分："两个字
- 评分必须是1.0-10.0之间的数字
- 不要输出0或0.0
- 示例正确格式：1. 评分：9.5（正确）
- 示例错误格式：1. 9.5（错误）、1. 评分：0（错误）、1. 评分：9.5分（错误）"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return [False] * len(source_texts), [8.0] * len(source_texts)  # 默认都不需要修正，默认8分
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 🔥 使用鲁棒的按行号提取，解决错位问题
                extracted_map = self._extract_by_line_number(response_content)
                
                if extracted_map:
                    needs_refinement = []
                    quality_scores = []  # 存储质量分数
                    
                    for i in range(len(source_texts)):
                        if i in extracted_map:
                            # 🔥 优先解析"评分："格式
                            raw_response = extracted_map[i].strip()
                            score_str = ""
                            
                            try:
                                # 方法1：查找"评分："或"评分:"后面的数字
                                if '评分：' in raw_response:
                                    score_str = raw_response.split('评分：')[-1].strip()
                                elif '评分:' in raw_response:
                                    score_str = raw_response.split('评分:')[-1].strip()
                                # 方法2：查找英文"score:"或"Score:"
                                elif 'score：' in raw_response.lower():
                                    score_str = raw_response.lower().split('score：')[-1].strip()
                                elif 'score:' in raw_response.lower():
                                    score_str = raw_response.lower().split('score:')[-1].strip()
                                # 方法3：如果没有前缀，直接当作数字
                                else:
                                    score_str = raw_response
                                
                                # 清理可能的干扰字符
                                score_str = score_str.replace('分', '').replace('/10', '').replace(' ', '').strip()
                                
                                # 如果是空的或以"."开头，说明解析失败
                                if not score_str or score_str.startswith('.'):
                                    raise ValueError(f"Invalid score format: '{score_str}'")
                                
                                # 🔥 如果只有整数（如"9"），自动添加".0"
                                if '.' not in score_str:
                                    score_str = score_str + '.0'
                                
                                score = float(score_str)
                                
                                # 🔥 分数范围检查：必须在1-10之间
                                if score < 1.0 or score > 10.0:
                                    self.warning(f"    ⚠ 行{i+1}评分异常({score})，超出1-10范围，使用默认值8.0")
                                    score = 8.0
                                
                                quality_scores.append(score)
                                # 7分以下需要修正
                                needs_refinement.append(score < 7.0)
                            except (ValueError, Exception) as e:
                                # 显示原始响应内容以便调试
                                self.warning(f"    ⚠ 无法解析行{i+1}的评分'{raw_response[:100]}'，使用默认值8.0")
                                quality_scores.append(8.0)  # 默认8分
                                needs_refinement.append(False)
                        else:
                            self.warning(f"    ⚠ 行{i+1}未找到评分，使用默认值8.0")
                            quality_scores.append(8.0)  # 默认8分
                            needs_refinement.append(False)
                    
                    # 🔥 显示详细评分
                    need_refine_count = sum(needs_refinement)
                    self.info(f"    ✓ 评估完成: {need_refine_count}/{len(source_texts)} 行需要修正")
                    
                    # 🔥 显示策略：
                    # - 如果<=10行：显示所有行的评分
                    # - 如果>10行：显示需要修正的行 + 前3行示例（让用户知道确实打分了）
                    show_all = len(source_texts) <= 10
                    
                    if show_all:
                        # 显示所有评分
                        for i, (score, needs_refine) in enumerate(zip(quality_scores, needs_refinement)):
                            status_icon = "⚠" if needs_refine else "✅"
                            status_text = "需修正" if needs_refine else "良好"
                            self.info(f"      行{i+1}: 评分：{score:.1f}/10 {status_icon} {status_text}")
                    else:
                        # 显示需要修正的行 + 前3行示例
                        shown_count = 0
                        for i, (score, needs_refine) in enumerate(zip(quality_scores, needs_refinement)):
                            should_show = needs_refine or (shown_count < 3 and not needs_refine)
                            
                            if should_show:
                                status_icon = "⚠" if needs_refine else "✅"
                                status_text = "需修正" if needs_refine else "良好"
                                self.info(f"      行{i+1}: 评分：{score:.1f}/10 {status_icon} {status_text}")
                                if not needs_refine:
                                    shown_count += 1
                        
                        if need_refine_count == 0 and len(source_texts) > 10:
                            self.info(f"      ... (其余{len(source_texts)-3}行评分均为良好，已省略)")
                    
                    return needs_refinement, quality_scores
        except Exception as e:
            self.debug(f"批量质量评估失败: {e}")
        
        return [False] * len(source_texts), [8.0] * len(source_texts)
    
    def _batch_refine_translation(self, source_texts: List[str], translated_texts: List[str],
                                 back_translations: List[str], needs_refinement: List[bool],
                                 terminology_db: Dict) -> Optional[List[str]]:
        """
        批量修正翻译（仅修正需要修正的行）
        """
        # 收集需要修正的行
        to_refine_indices = [i for i, need in enumerate(needs_refinement) if need]
        
        if not to_refine_indices:
            return translated_texts
        
        to_refine_sources = [source_texts[i] for i in to_refine_indices]
        to_refine_translations = [translated_texts[i] for i in to_refine_indices]
        to_refine_backs = [back_translations[i] for i in to_refine_indices]
        
        system_prompt = f"""你是一位专业的翻译修正专家。请根据原文和回译结果修正以下译文。

{self._build_terminology_prompt(terminology_db, to_refine_sources)}

🔥【强制要求-术语表必须严格遵守】🔥
- 如果原文中出现术语表中的任何术语，修正后的译文必须使用术语表中指定的翻译
- 绝对不允许用其他翻译替代术语表中的术语
- 这是强制性要求，不可违反
- 例如：如果术语表规定"phosphatidylinositol"必须翻译为"磷脂酰肌醇"，则修正时必须使用这个翻译
- 例如：如果术语表规定"Beclin"必须翻译为"Beclin"，则修正时不能改成"贝克林"
- 术语表的翻译优先级最高，即使回译结果显示有差异，也必须保持术语表规定的译法

【修正原则】
- 如果回译与原文的差异是由于术语翻译不一致导致的，不要修正译文，因为术语已经是正确的
- 只修正真正的语义错误、语法错误或流畅性问题
- 修正时必须保持术语表规定的所有术语翻译不变

【重要】输出格式要求：
- 必须使用<textarea>标签包裹所有修正后的译文
- 每行修正译文前必须加上序号（如1. 2. 3.）
- 不要添加任何额外的标题、前缀或说明文字"""
        
        # 构建批量修正的输入
        refine_blocks = []
        for i, (src, trans, back) in enumerate(zip(to_refine_sources, to_refine_translations, to_refine_backs)):
            block = f"{i + 1}. 原文: {src}\n   原译文: {trans}\n   回译: {back}"
            refine_blocks.append(block)
        
        refine_text = "\n\n".join(refine_blocks)
        
        user_prompt = f"""###请修正以下译文
{refine_text}

###【严格要求】输出格式
你必须只输出修正后的纯译文，不要输出"原文:"、"原译文:"、"回译:"、"修正后译文:"等标签。
格式如下：
<textarea>
1. （第1行修正后的纯中文译文）
2. （第2行修正后的纯中文译文）
</textarea>

例如，如果修正后译文是"这是翻译结果"，正确输出是：
<textarea>
1. 这是翻译结果
</textarea>

错误示例（不要这样输出）：
<textarea>
1. 原文:xxx 原译文:xxx 回译:xxx 修正后译文:这是翻译结果
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            if not self._wait_for_limiter(messages, system_prompt):
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                source_text_dict = {str(i): text for i, text in enumerate(to_refine_sources)}
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                if response_dict:
                    # 将修正结果填回原列表
                    refined_texts = translated_texts.copy()
                    for i, idx in enumerate(to_refine_indices):
                        key = str(i)
                        if key in response_dict and response_dict[key]:
                            refined_text = response_dict[key]
                            # 🔥 【关键】清理可能残留的前缀标签
                            refined_text = self._clean_refine_response(refined_text)
                            refined_texts[idx] = refined_text
                    
                    return refined_texts
        except Exception as e:
            self.error(f"批量修正失败: {e}", e)
        
        return None
    
    def _clean_refine_response(self, text: str) -> str:
        """
        清理修正响应中可能残留的前缀标签
        防止"原文:xxx 原译文:xxx 回译:xxx 修正后译文:xxx"这种格式被输出
        """
        if not text:
            return text
        
        # 如果包含"修正后译文:"，提取后面的内容
        if "修正后译文:" in text:
            text = text.split("修正后译文:")[-1].strip()
        elif "修正后译文：" in text:
            text = text.split("修正后译文：")[-1].strip()
        
        # 清理可能残留的其他标签
        prefixes_to_remove = [
            "原文:", "原文：",
            "原译文:", "原译文：",
            "回译:", "回译：",
            "译文:", "译文：",
        ]
        
        # 检查开头是否有这些前缀
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 如果整行都是"原文:xxx 原译文:xxx"格式，尝试提取最后一个有意义的部分
        if any(p in text for p in ["原文:", "原译文:", "回译:"]):
            # 使用正则表达式分割
            import re
            parts = re.split(r'(?:原文|原译文|回译|修正后译文)[：:]', text)
            if parts:
                # 取最后一个非空部分
                for part in reversed(parts):
                    part = part.strip()
                    if part and not any(p in part for p in ["原文", "原译文", "回译"]):
                        return part
        
        return text
    
    def _fallback_translate_one_by_one(self, source_texts: List[str], context_texts: List[str],
                                       strategy: str, terminology_db: Dict, memory_storage: Dict) -> List[str]:
        """
        Fallback机制：逐行翻译（当批量翻译完全失败时）
        """
        self.warning(f"  → 开始逐行Fallback翻译，共{len(source_texts)}行...")
        translated_texts = []
        
        for i, source_text in enumerate(source_texts):
            self.debug(f"    → 翻译第{i+1}/{len(source_texts)}行...")
            translation = self._translate_single_line(
                source_text, context_texts, strategy, terminology_db, memory_storage
            )
            if translation:
                translated_texts.append(translation)
            else:
                # 如果单行翻译也失败，保留原文标记
                translated_texts.append(f"[翻译失败]{source_text}")
                self.error(f"    ✗ 第{i+1}行翻译失败")
        
        success_count = sum(1 for t in translated_texts if not t.startswith("[翻译失败]"))
        self.info(f"  ✓ 逐行翻译完成: {success_count}/{len(source_texts)} 行成功")
        return translated_texts
    
    def _translate_single_line(self, source_text: str, context_texts: List[str],
                               strategy: str, terminology_db: Dict, memory_storage: Dict) -> Optional[str]:
        """
        翻译单行文本（用于fallback或补充翻译）
        """
        # 构建术语提示
        terminology_prompt = self._build_terminology_prompt(terminology_db, [source_text])
        
        # 🔥 检测是否为参考文献
        is_reference = any(keyword in source_text.lower() for keyword in [
            'et al.', 'doi:', 'http://', 'https://', 'pubmed', 'pmid:', 
            'journal', 'proc.', 'vol.', 'pp.', 'issn'
        ]) or (len(source_text) > 500 and source_text.count(',') > 5)
        
        # 为参考文献添加特殊说明
        reference_instruction = ""
        if is_reference:
            reference_instruction = """
【参考文献翻译要求】
⚠️ 这是参考文献内容，需要翻译！不要直接输出英文原文！
- 必须翻译：文章标题、期刊名称、会议名称
- 保留不变：作者姓名、年份、DOI、URL、卷号页码
- 翻译示例：
  原文: Brown, W.J. et al. (1995) Role for phosphatidylinositol 3-kinase in lysosomal enzyme transport. Nature 377, 525–528.
  译文: Brown, W.J. 等人 (1995) 磷脂酰肌醇3-激酶在溶酶体酶运输中的作用。《自然》377, 525–528。"""
        
        # 简化的system_prompt（单行翻译不需要复杂的多步骤引导）
        system_prompt = f"""你是一位专业的翻译专家。
        
{terminology_prompt}
{reference_instruction}

【重要】输出格式：
- 直接输出译文，不要添加序号、标签或其他说明文字
- 必须翻译成中文，不要直接输出英文原文"""
        
        # 构建上下文
        context_str = "\n".join(context_texts[-3:]) if context_texts else ""
        context_prefix = f"###上文内容\n{context_str}\n\n" if context_str else ""
        
        user_prompt = f"""{context_prefix}###待翻译文本
{source_text}

###译文"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            # 等待RequestLimiter
            if not self._wait_for_limiter(messages, system_prompt):
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 简单提取：去除前后空白和可能的引号
                translation = response_content.strip().strip('"').strip("'")
                return translation if translation else None
            else:
                return None
        except Exception as e:
            self.error(f"单行翻译失败: {e}", e)
            return None
    
    def _strategy_based_batch_translation(self, source_texts: List[str], context_texts: List[str],
                                              strategy: str, terminology_db: Dict, memory_storage: Dict) -> Optional[List[str]]:
        """
        基于策略的批量翻译（合并原步骤1和2）
        根据PlanningAgent分配的策略，直接执行对应的翻译方式
        
        Args:
            source_texts: 待翻译文本列表
            context_texts: 上下文文本列表
            strategy: 翻译策略 ("literal" | "free" | "stylized")
            terminology_db: 术语库
            memory_storage: 记忆存储
        
        Returns:
            翻译结果列表
        """
        # 构建术语提示（✅ 传递source_texts用于动态筛选）
        terminology_prompt = self._build_terminology_prompt(terminology_db, source_texts)
        
        # 根据策略构建不同的system_prompt
        if strategy == "literal":
            # 直译策略：强调术语准确、保持原文结构
            strategy_instruction = """直译策略：
- 保持原文的句子结构和表达方式
- **严格遵守术语库中的专业术语和实体翻译，不得更改**
- 优先保证准确性和术语一致性，其次考虑流畅性
- 适用于技术文档、法律文本等正式内容"""
        elif strategy == "stylized":
            # 风格化策略：强调艺术性、韵律感
            strategy_instruction = """风格化策略：
- 注重译文的艺术性和文学美感
- 保持原文的韵律、节奏和情感
- **术语库中的人名、地名等专有名词必须使用固定翻译**
- 可以适当调整句式以符合目标语言习惯
- 适用于文学作品、诗歌、营销文案"""
        else:  # free (默认)
            # 意译策略：强调自然流畅
            strategy_instruction = """意译策略：
- 注重译文的自然流畅性
- 符合目标语言的表达习惯
- **术语库中的专有名词和关键术语必须使用固定翻译**
- 准确传达原文的意思，可灵活调整表达方式
- 适用于对话、叙述性文本等日常内容"""
        
        # 🔥 检测是否包含参考文献
        has_references = any(
            any(keyword in text.lower() for keyword in [
                'et al.', 'doi:', 'http://', 'https://', 'pubmed', 'pmid:',
                'journal', 'proc.', 'vol.', 'pp.', 'issn', 'references'
            ]) or (len(text) > 500 and text.count(',') > 5)
            for text in source_texts
        )
        
        reference_instruction = ""
        if has_references:
            reference_instruction = """
【参考文献翻译要求】
⚠️ 如果文本中包含参考文献，必须翻译！不要直接输出英文原文！
- 必须翻译：文章标题、期刊名称、会议名称
- 保留不变：作者姓名、年份、DOI、URL、卷号页码
- 翻译示例：
  原文: Brown, W.J. et al. (1995) Role for phosphatidylinositol 3-kinase in lysosomal enzyme transport. Nature 377, 525–528.
  译文: Brown, W.J. 等人 (1995) 磷脂酰肌醇3-激酶在溶酶体酶运输中的作用。《自然》377, 525–528。
"""
        
        # 构建system_prompt（强制要求遵守术语表）
        system_prompt = f"""你是一位专业的翻译专家，你的任务是把原文翻译成中文，逐行翻译，不要合并，保持原来的格式。

请按照以下步骤进行翻译：
步骤1 - 理解：分析原文的语义、语境和风格
步骤2 - 分解：对于长难句，先识别主干成分和从句层级
步骤3 - 转换：将原文转换为目标语言，保持语义准确
步骤4 - 润色：优化译文，确保流畅自然

{strategy_instruction}

{terminology_prompt}

🔥【强制要求-术语表遵守】🔥
- 如果原文中出现术语表中的任何术语，必须使用术语表中指定的翻译
- 绝对不允许用其他翻译替代术语表中的术语
- 例如：如果术语表规定"Beclin"必须翻译为"贝可林"，则不能翻译为"Beclin"、"贝克林"或其他任何译法
- 例如：如果术语表规定"phosphatidylinositol"必须翻译为"磷脂酰肌醇"，则不能翻译为"磷脂肌醇"或其他任何译法
{reference_instruction}
【重要】输出格式要求：
- 逐行翻译，不要合并，原文有{len(source_texts)}行，译文也必须有{len(source_texts)}行
- 输出的翻译顺序标号必须和输入一一对应：输入1.对应输出1.，输入2.对应输出2.，依此类推
- 必须使用<textarea>标签包裹所有译文
- 每行译文前必须加上序号（如1. 2. 3.）
- 序号必须从1到{len(source_texts)}连续，不要跳过
- 即使是很短的行也不要与其他行合并
- 必须翻译成中文，不要直接输出英文原文
- 不要自动添加书名号《》、引号""或其他原文没有的标点符号
- 不要添加任何解释性文字，如"（音译为主）"、"（可加注说明）"、"（注：...）"等
- 只输出纯粹的翻译结果，不要加任何注释或说明
- 如果原文是"scientific reports"，只翻译为"科学报告"，不要翻译为"《科学报告》"

格式示例：
<textarea>
1.第一行译文
2.第二行译文
3.第三行译文
</textarea>"""
        
        # 构建source_text_dict（使用与原方法相同的格式）
        source_text_dict = {str(i): text for i, text in enumerate(source_texts)}
        
        # 构建待翻译文本（使用与原TranslatorTask相同的格式）
        numbered_lines = []
        for index, line in enumerate(source_texts):
            if "\n" in line:
                sub_lines = line.split("\n")
                formatted_line = f"{index + 1}.{sub_lines[0]}"
                for sub_line in sub_lines[1:]:
                    formatted_line += f"\n{sub_line}"
                numbered_lines.append(formatted_line)
            else:
                numbered_lines.append(f"{index + 1}.{line}")
        
        source_text = "\n".join(numbered_lines)
        
        # 构建上下文
        context_str = "\n".join(context_texts[-3:]) if context_texts else ""
        
        # 构建user_prompt（与原方法一致，使用textarea标签）
        context_prefix = f"###上文内容\n{context_str}\n" if context_str else ""
        user_prompt = f"""{context_prefix}###待翻译文本（共{len(source_texts)}行）
<textarea>
{source_text}
</textarea>

###译文输出格式（必须严格遵守）
⚠️ 原文有{len(source_texts)}行，译文也必须有{len(source_texts)}行，序号从1到{len(source_texts)}
⚠️ 输出的翻译顺序标号必须和输入一一对应：输入第1行对应输出第1行，输入第2行对应输出第2行，依此类推
⚠️ 不要合并多行，不要跳过任何行，不要改变顺序
⚠️ 不要自动添加书名号《》或其他标点符号
<textarea>
1. （第1行译文）
2. （第2行译文）
...
{len(source_texts)}. （第{len(source_texts)}行译文）
</textarea>"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        try:
            # 等待RequestLimiter允许发送请求
            if not self._wait_for_limiter(messages, system_prompt):
                self.warning(f"  ⚠ RequestLimiter检查失败")
                return None
            
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 使用ResponseExtractor提取翻译结果（与原TranslatorTask完全相同）
                response_dict = ResponseExtractor.text_extraction(self, source_text_dict, response_content)
                
                # 去除数字序号前缀（与原方法相同）
                response_dict = ResponseExtractor.remove_numbered_prefix(self, response_dict)
                
                # 只取我们需要的键，忽略多余的键（与原方法一致）
                if response_dict:
                    translated_texts = []
                    for i in range(len(source_texts)):
                        key = str(i)
                        if key in response_dict:
                            translated_texts.append(response_dict[key])
                        else:
                            translated_texts.append("")  # 缺失的键用空字符串填充
                    
                    # 只要有部分译文就返回
                    if any(translated_texts):
                        non_empty_count = sum(1 for t in translated_texts if t)
                        self.info(f"  ✓ 批量{strategy}翻译成功: {non_empty_count}/{len(translated_texts)} 行")
                        return translated_texts
                    else:
                        self.warning(f"  ⚠ 所有译文均为空")
                        return None
                else:
                    self.warning(f"  ⚠ ResponseExtractor未能解析任何结果")
                    return None
            else:
                self.warning("  ⚠ LLM返回为空")
                return None
        except Exception as e:
            self.error(f"批量{strategy}翻译失败: {e}", e)
            return None
    
    def _check_entity_consistency(self, source_texts: List[str], translated_texts: List[str],
                                  terminology_db: Dict, entity_database: Dict) -> List[str]:
        """
        检查并修正实体一致性问题
        
        确保：
        1. 人名、地名等专有名词翻译一致
        2. 术语库中的术语翻译一致
        3. 跨批次的实体翻译保持统一
        
        Args:
            source_texts: 原文列表
            translated_texts: 译文列表
            terminology_db: 术语库
            entity_database: 实体数据库（会被更新）
        
        Returns:
            修正后的译文列表
        """
        # 🔥 发送UI阶段更新：一致性检查阶段（仅第一次）
        if self._current_cache_project and not hasattr(self, '_entity_check_stage_sent'):
            # 一致性检查：单步操作
            self._update_stage_progress(self._current_cache_project, "entity_check", 0, 1)
            self._publish_stage_with_stats(self._current_cache_project, "entity_check", "检查中")
            self._entity_check_stage_sent = True
        
        import re

        def _find_actual_entity_rendering(entity: str, expected_translation: str, text: str) -> Dict[str, str]:
            """
            试图从译文中找出“实体实际呈现”为何，便于日志定位问题：
            1) 优先检测是否保留了原文实体（大小写不敏感）
            2) 否则从期望译文里提取中文子串（2-5字）去译文里找命中窗口
            3) 仍找不到则返回未知
            """
            if not text:
                return {"actual": "", "hint": "译文为空"}

            # 1) 是否保留原文实体
            try:
                m = re.search(re.escape(entity), text, flags=re.IGNORECASE)
            except Exception:
                m = None
            if m:
                start = max(0, m.start() - 25)
                end = min(len(text), m.end() + 25)
                snippet = text[start:end]
                return {"actual": snippet, "hint": "疑似保留原文实体（未按术语表翻译）"}

            # 2) 从期望译文里抽中文子串做“命中窗口”
            zh = re.sub(r"[^\u4e00-\u9fff]+", "", expected_translation or "")
            if zh:
                # 生成2-5字子串，按长度优先，避免太短误命中
                subs = []
                max_len = min(5, len(zh))
                min_len = 2 if len(zh) >= 2 else 1
                for L in range(max_len, min_len - 1, -1):
                    for i in range(0, len(zh) - L + 1):
                        subs.append(zh[i:i + L])
                seen = set()
                uniq_subs = []
                for s in subs:
                    if s not in seen:
                        seen.add(s)
                        uniq_subs.append(s)

                for s in uniq_subs:
                    idx = text.find(s)
                    if idx != -1:
                        start = max(0, idx - 25)
                        end = min(len(text), idx + len(s) + 25)
                        snippet = text[start:end]
                        return {"actual": snippet, "hint": f"命中期望译文关键词片段: {s}"}

            return {"actual": "", "hint": "未找到明显对应片段（可能被改写/省略/同义替换）"}
        
        # 从术语库中提取实体映射
        entity_mappings = {}
        # ✅ 注意：terminology_db 的 key 可能就是术语本身；同时要过滤空term，避免 "" in text 永远为真导致误报
        if isinstance(terminology_db, dict):
            for k, term_info in terminology_db.items():
                if not isinstance(term_info, dict):
                    continue
                raw_term = (term_info.get("term") or k or "")
                raw_translation = (term_info.get("translation") or "")
                term = str(raw_term).strip()
                translation = str(raw_translation).strip()
                if not term or not translation:
                    continue
                # 额外清理：防止术语库里残留Markdown标记影响匹配
                translation = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", translation).strip()
                translation = re.sub(r"_{1,2}(.+?)_{1,2}", r"\1", translation).strip()
                if not translation:
                    continue
                entity_mappings[term] = translation
        
        # 输出实体/术语统计
        if entity_mappings:
            self.debug(f"  → 正在检查 {len(entity_mappings)} 个实体/术语的一致性...")
            # 显示前5个实体作为示例
            sample_entities = list(entity_mappings.items())[:5]
            for entity, trans in sample_entities:
                self.debug(f"    • {entity} → {trans}")
            if len(entity_mappings) > 5:
                self.debug(f"    ... 以及其他 {len(entity_mappings) - 5} 个")
        else:
            self.debug(f"  → 未发现实体/术语，跳过一致性检查")
            return translated_texts
        
        # 检查并修正每一行译文
        corrected_texts = []
        inconsistency_details = []  # 存储详细的不一致信息
        entities_verified = 0  # 统计验证通过的实体数量
        entities_auto_fixed = 0  # 🔥 新增：统计自动修正的实体数量
        entity_check_log = []  # 记录每个实体的检查情况
        
        for line_idx, (source_text, translated_text) in enumerate(zip(source_texts, translated_texts)):
            if not translated_text:
                corrected_texts.append(translated_text)
                continue
            
            corrected_text = translated_text
            line_entities_found = []  # 本行找到的实体
            line_entities_replaced = []  # 本行强制替换的实体
            line_entities_missing = []  # 本行缺失但无法替换的实体
            
            # 🔥 强制替换：按照术语表强制替换实体翻译
            for entity, expected_translation in entity_mappings.items():
                # 如果原文中有该实体（不区分大小写）
                if entity.lower() in source_text.lower():
                    # 检查译文中是否已有正确翻译（模糊匹配）
                    normalized_translation = re.sub(r'[\s\-–—]+', '', expected_translation.lower())
                    normalized_text = re.sub(r'[\s\-–—]+', '', corrected_text.lower())
                    
                    if normalized_translation in normalized_text or expected_translation.lower() in corrected_text.lower():
                        # 译文中已包含正确的翻译
                        entities_verified += 1
                        line_entities_found.append(f"{entity}→{expected_translation}✓")
                        
                        # 更新实体数据库
                        if entity not in entity_database:
                            entity_database[entity] = {
                                "translation": expected_translation,
                                "occurrences": 1,
                                "source": "terminology_db"
                            }
                        else:
                            entity_database[entity]["occurrences"] += 1
                    else:
                        # 🔥 策略1：自动修正保留的原文实体
                        # 仅当译文里仍出现原文实体（如 Beclin / Autophagy）且期望译文不同，才直接替换为期望译文
                        if expected_translation and expected_translation != entity:
                            try:
                                before = corrected_text
                                corrected_text = re.sub(re.escape(entity), expected_translation, corrected_text, flags=re.IGNORECASE)
                                if corrected_text != before:
                                    entities_auto_fixed += 1  # 🔥 计数自动修正
                                    entities_verified += 1
                                    line_entities_replaced.append(f"{entity}→{expected_translation}✓(自动修正)")
                                    line_entities_found.append(f"{entity}→{expected_translation}✓(自动修正)")
                                    self.debug(f"    ✅ [行{line_idx+1}] 自动修正保留的原文实体: {entity} → {expected_translation}")
                                    continue
                            except Exception:
                                pass

                        # 仍无法修正：记录为需要重新翻译/人工关注
                        
                        # 先尝试找到原文中entity的确切位置模式
                        # 使用正则表达式查找entity在原文中的所有匹配位置
                        entity_pattern = re.compile(re.escape(entity), re.IGNORECASE)
                        matches = list(entity_pattern.finditer(source_text))
                        
                        if matches:
                            # 尝试智能替换：查找译文中对应位置的可能错误翻译
                            # 简单策略：如果译文长度接近原文，按比例定位
                            # 复杂策略：使用LLM重新翻译这一行（成本较高）
                            
                            # 这里使用简单策略：全局搜索可能的错误翻译并替换
                            # 例如：如果entity是"Beclin"，expected是"Beclin"，但译文中是"贝克林"
                            # 我们需要找到"贝克林"并替换为"Beclin"
                            
                            # 由于不知道LLM具体把entity翻译成了什么，我们采用强制插入策略
                            # 在第一次出现相关内容的地方插入正确翻译
                            
                            # 更简单的策略：记录为需要重新翻译的行
                            line_entities_missing.append(f"{entity}→{expected_translation}❌")
                            actual_info = _find_actual_entity_rendering(entity, expected_translation, corrected_text)
                            # 控制日志长度，避免一条过长刷屏
                            actual_snippet = (actual_info.get("actual") or "")
                            if len(actual_snippet) > 160:
                                actual_snippet = actual_snippet[:160] + "..."
                            inconsistency_details.append({
                                "line": line_idx + 1,
                                "entity": entity,
                                "expected": expected_translation,
                                "actual_entity": actual_snippet,
                                "actual_hint": actual_info.get("hint", ""),
                                "source": source_text[:80] + "..." if len(source_text) > 80 else source_text,
                                "translation": corrected_text[:80] + "..." if len(corrected_text) > 80 else corrected_text,
                                "action": "需要重新翻译"
                            })
                        else:
                            # 原文中没找到entity（理论上不应该发生）
                            pass
            
            # 记录本行的检查结果
            if line_entities_found or line_entities_missing or line_entities_replaced:
                entity_check_log.append({
                    "line": line_idx + 1,
                    "found": line_entities_found,
                    "missing": line_entities_missing,
                    "replaced": line_entities_replaced
                })
            
            corrected_texts.append(corrected_text)
        
        # 🔥 输出完整的检查结果
        inconsistencies_found = len(inconsistency_details)
        
        # 显示自动修正统计
        if entities_auto_fixed > 0:
            self.info(f"  ✅ 自动修正了 {entities_auto_fixed} 处保留的原文实体")
        
        if inconsistencies_found > 0:
            self.warning(f"  ⚠ 发现 {inconsistencies_found} 处无法自动修正的实体一致性问题，"
                        f"{entities_verified} 个实体翻译正确（含 {entities_auto_fixed} 个自动修正）")
            
            # 🔥 显示所有不一致的详细信息（不限制数量）
            self.warning(f"  → 【无法自动修正的问题列表】共 {inconsistencies_found} 处：")
            for i, detail in enumerate(inconsistency_details, 1):
                self.warning(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                self.warning(f"    问题 {i}/{inconsistencies_found}")
                self.warning(f"    【行号】: {detail['line']}")
                self.warning(f"    【原文实体】: '{detail['entity']}'")
                self.warning(f"    【期望译文】: '{detail['expected']}'")
                if detail.get("actual_entity") or detail.get("actual_hint"):
                    self.warning(f"    【译文中实体呈现】: {detail.get('actual_entity', '')}")
                    self.warning(f"    【判定依据】: {detail.get('actual_hint', '')}")
                self.warning(f"    【原文片段】: {detail['source']}")
                self.warning(f"    【实际译文】: {detail['translation']}")
                self.warning(f"    【处理方式】: {detail.get('action', '未处理')}")
            self.warning(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # 统计最常出现问题的实体
            problem_entities = {}
            for detail in inconsistency_details:
                entity = detail['entity']
                problem_entities[entity] = problem_entities.get(entity, 0) + 1
            
            self.warning(f"  → 【问题实体统计】（出现次数）：")
            for entity, count in sorted(problem_entities.items(), key=lambda x: x[1], reverse=True):
                expected = next((d['expected'] for d in inconsistency_details if d['entity'] == entity), "?")
                self.warning(f"    • {entity} (期望: {expected}): {count}次")
            
        else:
            if entities_verified > 0:
                auto_fix_info = f"（含 {entities_auto_fixed} 个自动修正）" if entities_auto_fixed > 0 else ""
                self.info(f"  ✓ 实体一致性检查通过：{entities_verified} 个实体翻译一致{auto_fix_info}")
                # 显示验证通过的实体
                if entity_check_log:
                    self.debug(f"  → 验证通过的实体详情：")
                    for log in entity_check_log[:10]:  # 显示前10行
                        if log["found"]:
                            self.debug(f"    【行{log['line']}】{', '.join(log['found'][:5])}")
            else:
                self.debug(f"  ✓ 本批次未检测到需要验证的实体")
        
        # 🔥 更新进度：检查完成
        if self._current_cache_project:
            self._update_stage_progress(self._current_cache_project, "entity_check", 1, 1)
        
        return corrected_texts
