"""
Griptape Tools
将自定义Agent的功能封装为Griptape Tools

架构说明：
- Tools接收workflow_state（共享状态字典），用于在Tools之间传递大对象（cache_project等）
- Tools内部调用现有的Agent执行业务逻辑
- Griptape只负责控制流程，不传递大数据对象
"""

import json
import msgspec
from typing import Dict, Any, List, Optional
from schema import Schema, Literal
from griptape.tools import BaseTool
from griptape.artifacts import TextArtifact, ErrorArtifact
from griptape.utils.decorators import activity

from Base.Base import Base
from ModuleFolders.TaskConfig.TaskConfig import TaskConfig
from ModuleFolders.Cache.CacheProject import CacheProject
from .PreprocessingAgent import PreprocessingAgent
from .TerminologyEntityAgent import TerminologyEntityAgent
from .TranslationRefinementAgent import TranslationRefinementAgent


class PreprocessingTool(BaseTool, Base):
    """预处理Tool"""
    
    def __init__(self, config: TaskConfig, workflow_state: Optional[Dict[str, Any]] = None):
        BaseTool.__init__(self)
        Base.__init__(self)
        self.config = config
        self.preprocessing_agent = PreprocessingAgent(config)
        # 共享的工作流状态（由WorkflowManager注入），用于在Tool之间传递大对象（cache_project等）
        self.workflow_state = workflow_state if workflow_state is not None else {}
    
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
                self.debug(f"[PreprocessingTool] 进入新阶段: {stage}, 总进度={total}")
            
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
        
        self.debug(f"[PreprocessingTool] 发送完整更新: stage={stage}, batch_info={batch_info}, line={update_data.get('line', 0)}/{update_data.get('total_line', 0)}, time={update_data.get('time', 0):.1f}s")
        self.emit(Base.EVENT.TASK_UPDATE, update_data)
    
    def to_activity_json_schema(self, activity, schema_id: str) -> dict:
        """重写以修复 $schema 字段问题"""
        schema = super().to_activity_json_schema(activity, schema_id)
        # 移除可能导致问题的 $schema 和 $id 字段（这些字段可能导致"relative URL without a base"错误）
        if "$schema" in schema:
            del schema["$schema"]
        if "$id" in schema:
            del schema["$id"]
        return schema
    
    @activity(
        config={
            "description": "对文本进行预处理：文本结构拆解和语域风格识别。直接调用即可，工具会自动获取所需数据。",
            "schema": Schema({}),
        },
    )
    def preprocess_text(self, params: dict) -> TextArtifact:
        """执行预处理"""
        try:
            self.info(f"[PreprocessingTool] 接收到调用请求，params={params}")
            self.info(f"[PreprocessingTool] workflow_state keys: {list(self.workflow_state.keys())}")
            
            cache_project: CacheProject = self.workflow_state.get("cache_project")
            if not cache_project:
                self.error("[PreprocessingTool] workflow_state中缺少cache_project")
                self.error(f"[PreprocessingTool] workflow_state内容: {self.workflow_state}")
                return ErrorArtifact("workflow_state中缺少cache_project（请检查WorkflowManager注入）")

            self.info("[PreprocessingTool] 开始执行预处理")
            
            # 🔥 发送UI阶段更新（包含统计数据）
            self._publish_stage_with_stats(cache_project, "preprocessing", "处理中")
            
            # 🔥 不再使用progress_callback，避免与新的阶段更新系统冲突
            # progress_callback会发送没有agent_stage的更新，可能导致UI显示错误
            
            result = self.preprocessing_agent.execute({"cache_project": cache_project})
            
            # ==========================================
            # DB Phase 1.5: 保存处理原子
            # ==========================================
            try:
                # 检查是否已注入 DB 信息
                if hasattr(cache_project, 'db_work_id') and hasattr(cache_project, 'db_doc_map'):
                    from ModuleFolders.Cache.DatabaseManager import DatabaseManager
                    db_manager = DatabaseManager()
                    
                    self.info("[DB] 开始同步处理原子...")
                    
                    # 建立 row_index -> atom_id 的全局映射
                    # 结构: { file_path: { row_index: atom_id } }
                    if not hasattr(cache_project, 'db_atom_map'):
                        cache_project.db_atom_map = {}
                    
                    # 遍历文件字典
                    for file_path, cache_file in cache_project.files.items():
                        items = cache_file.items
                        
                        doc_id = cache_project.db_doc_map.get(file_path)
                        if not doc_id:
                            continue
                        
                        # 构建原子数据列表（包含上下文信息和摘要）
                        atoms_data = []
                        for idx, item in enumerate(items):
                            # 构建完整上下文信息
                            context_info = {
                                "prev_source": items[idx-1].source_text if idx > 0 else None,
                                "prev_translated": None,  # 翻译完成后填充
                                "next_source": items[idx+1].source_text if idx < len(items)-1 else None,
                                "terminology": [],  # 术语识别后填充
                                "memory_refs": [],  # 翻译记忆检索后填充
                                "similar_atoms": []  # 相似历史翻译
                            }
                            
                            # 生成简单摘要（取前100字符）
                            summary = item.source_text[:100] + "..." if len(item.source_text) > 100 else item.source_text
                            
                            atoms_data.append({
                                "source_text": item.source_text,
                                "position": idx,
                                "summary": summary,
                                "context_info": context_info
                            })
                        
                        # 批量创建原子
                        atom_ids = db_manager.create_atoms_batch(doc_id, atoms_data)
                        
                        # 更新文档的原子总数
                        db_manager.update_document_atom_count(doc_id, len(atom_ids))
                        
                        # 建立映射
                        if len(atom_ids) == len(items):
                            file_atom_map = {}
                            for item, a_id in zip(items, atom_ids):
                                file_atom_map[item.row_index] = a_id
                                # 把 atom_id 塞回 item (运行时属性)
                                item.db_atom_id = a_id
                            
                            cache_project.db_atom_map[file_path] = file_atom_map
                            self.info(f"[DB] 文件 {file_path} 同步完成: {len(atom_ids)} 个原子")
                        else:
                            self.error(f"[DB] 原子数量不匹配! Items: {len(items)}, IDs: {len(atom_ids)}")
                    
                    # 持久化 atom_map 到 extra (支持断点续传)
                    if not hasattr(cache_project, 'extra') or not isinstance(cache_project.extra, dict):
                        cache_project.extra = {}
                    cache_project.extra['db_atom_map'] = cache_project.db_atom_map
                            
            except Exception as e:
                self.error(f"[DB] 原子同步失败: {e}")

            # 🔥 发送UI阶段更新：完成（包含统计数据）
            self._publish_stage_with_stats(cache_project, "preprocessing", "完成")

            # 更新共享状态（避免把大JSON再塞回LLM函数调用输出）
            if isinstance(result, dict):
                if result.get("cache_project"):
                    self.workflow_state["cache_project"] = result["cache_project"]
                if result.get("metadata") is not None:
                    self.workflow_state["metadata"] = result.get("metadata", {})

            # 返回小结果，供调试/链路可见性
            metadata = self.workflow_state.get("metadata", {})
            summary = {
                "success": bool(result.get("success")),
                "stage": "preprocess",
                "domain": metadata.get("domain"),
                "style": metadata.get("style"),
            }
            
            self.info(f"[PreprocessingTool] 预处理完成: {summary}")
            return TextArtifact(json.dumps(summary, ensure_ascii=False))
        except Exception as e:
            self.error(f"预处理工具执行失败: {e}", e)
            return ErrorArtifact(str(e))


class TerminologyTool(BaseTool, Base):
    """术语识别Tool"""
    
    def __init__(self, config: TaskConfig, workflow_state: Optional[Dict[str, Any]] = None):
        BaseTool.__init__(self)
        Base.__init__(self)
        self.config = config
        self.terminology_agent = TerminologyEntityAgent(config)
        self.workflow_state = workflow_state if workflow_state is not None else {}
    
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
                self.debug(f"[TerminologyTool] 进入新阶段: {stage}, 总进度={total}")
            
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
        
        self.debug(f"[TerminologyTool] 发送完整更新: stage={stage}, batch_info={batch_info}, line={update_data.get('line', 0)}/{update_data.get('total_line', 0)}, time={update_data.get('time', 0):.1f}s")
        self.emit(Base.EVENT.TASK_UPDATE, update_data)
    
    def to_activity_json_schema(self, activity, schema_id: str) -> dict:
        """重写以修复 $schema 字段问题"""
        schema = super().to_activity_json_schema(activity, schema_id)
        # 移除可能导致问题的 $schema 和 $id 字段（这些字段可能导致"relative URL without a base"错误）
        if "$schema" in schema:
            del schema["$schema"]
        if "$id" in schema:
            del schema["$id"]
        return schema
    
    @activity(
        config={
            "description": "识别术语和实体：NER、领域术语、文化负载词，并构建术语库。直接调用即可，工具会自动获取所需数据。",
            "schema": Schema({}),
        },
    )
    def identify_terminology(self, params: dict) -> TextArtifact:
        """执行术语识别"""
        try:
            self.info(f"[TerminologyTool] 接收到调用请求，params={params}")
            self.info(f"[TerminologyTool] workflow_state keys: {list(self.workflow_state.keys())}")
            
            cache_project: CacheProject = self.workflow_state.get("cache_project")
            metadata = self.workflow_state.get("metadata", {}) or {}
            if not cache_project:
                self.error("[TerminologyTool] workflow_state中缺少cache_project")
                self.error(f"[TerminologyTool] workflow_state内容: {self.workflow_state}")
                return ErrorArtifact("workflow_state中缺少cache_project（请检查WorkflowManager注入）")

            self.info("[TerminologyTool] 开始执行术语识别")
            
            # 🔥 发送UI阶段更新（包含统计数据）
            self._update_stage_progress(cache_project, "terminology", 0, 1)
            self._publish_stage_with_stats(cache_project, "terminology", "识别中")
            
            # 🔥 不再使用progress_callback，避免与新的阶段更新系统冲突
            
            result = self.terminology_agent.execute({
                "cache_project": cache_project,
                "metadata": metadata
            })
            
            # 🔥 发送UI阶段更新：完成（包含统计数据）
            self._update_stage_progress(cache_project, "terminology", 1, 1)
            self._publish_stage_with_stats(cache_project, "terminology", "完成")

            if isinstance(result, dict):
                if result.get("cache_project"):
                    self.workflow_state["cache_project"] = result["cache_project"]
                if result.get("terminology_database") is not None:
                    self.workflow_state["terminology_database"] = result.get("terminology_database", {})
                if result.get("memory_storage") is not None:
                    self.workflow_state["memory_storage"] = result.get("memory_storage", {})

            term_count = len(self.workflow_state.get("terminology_database", {}) or {})
            summary = {
                "success": bool(result.get("success")),
                "stage": "terminology",
                "terminology_count": term_count,
            }
            
            self.info(f"[TerminologyTool] 术语识别完成: {summary}")
            return TextArtifact(json.dumps(summary, ensure_ascii=False))
        except Exception as e:
            self.error(f"术语识别工具执行失败: {e}", e)
            return ErrorArtifact(str(e))


class TranslationTool(BaseTool, Base):
    """翻译Tool"""
    
    def __init__(self, config: TaskConfig, workflow_state: Optional[Dict[str, Any]] = None):
        BaseTool.__init__(self)
        Base.__init__(self)
        self.config = config
        self.translation_agent = TranslationRefinementAgent(config)
        self.workflow_state = workflow_state if workflow_state is not None else {}
    
    def to_activity_json_schema(self, activity, schema_id: str) -> dict:
        """重写以修复 $schema 字段问题"""
        schema = super().to_activity_json_schema(activity, schema_id)
        # 移除可能导致问题的 $schema 和 $id 字段（这些字段可能导致"relative URL without a base"错误）
        if "$schema" in schema:
            del schema["$schema"]
        if "$id" in schema:
            del schema["$id"]
        return schema
    
    @activity(
        config={
            "description": "执行翻译和优化：多步骤翻译、多版本融合、回译验证。直接调用即可，工具会自动获取所需数据。",
            "schema": Schema({}),
        },
    )
    def translate_and_refine(self, params: dict) -> TextArtifact:
        """执行翻译和优化"""
        try:
            self.info(f"[TranslationTool] 接收到调用请求，params={params}")
            self.info(f"[TranslationTool] workflow_state keys: {list(self.workflow_state.keys())}")
            
            # 检查是否已经翻译完成（防止重复调用）
            if self.workflow_state.get("did_translate"):
                translated_count = len(self.workflow_state.get("translation_results", []) or [])
                self.info(f"[TranslationTool] 翻译已完成（{translated_count}个单元），直接返回结果")
                summary = {
                    "success": True,
                    "stage": "translate",
                    "translated_count": translated_count,
                    "message": "翻译已完成，无需重复执行"
                }
                return TextArtifact(json.dumps(summary, ensure_ascii=False))
            
            cache_project: CacheProject = self.workflow_state.get("cache_project")
            if not cache_project:
                self.error("[TranslationTool] workflow_state中缺少cache_project")
                self.error(f"[TranslationTool] workflow_state内容: {self.workflow_state}")
                return ErrorArtifact("workflow_state中缺少cache_project（请检查WorkflowManager注入）")

            self.info("[TranslationTool] 开始执行翻译和优化")
            
            terminology_db = self.workflow_state.get("terminology_database", {}) or {}
            memory_storage = self.workflow_state.get("memory_storage", {}) or {}
            progress_callback = self.workflow_state.get("progress_callback")  # 获取进度回调
            planning_result = self.workflow_state.get("planning_result", {})  # 获取规划结果
            task_memory = self.workflow_state.get("task_memory", {})  # 获取任务元数据（chunk策略、实体数据库等）
            human_intervention_callback = self.workflow_state.get("human_intervention_callback")  # 🔥 获取人工介入回调

            result = self.translation_agent.execute({
                "cache_project": cache_project,
                "terminology_database": terminology_db,
                "memory_storage": memory_storage,
                "human_intervention_callback": human_intervention_callback,  # 🔥 传递人工介入回调
                "progress_callback": progress_callback,  # 传递进度回调
                "planning_result": planning_result,  # 🔥 传递规划结果（包含max_workers等配置）
                "task_memory": task_memory,  # 🔥 传递任务元数据（chunk策略、实体数据库等）
            })

            if isinstance(result, dict):
                if result.get("cache_project"):
                    self.workflow_state["cache_project"] = result["cache_project"]
                # 标记是否发生翻译（用于WorkflowManager判定成功）
                self.workflow_state["did_translate"] = bool(result.get("success"))
                self.workflow_state["translation_results"] = result.get("translation_results", [])

            translated_count = len(self.workflow_state.get("translation_results", []) or [])
            summary = {
                "success": bool(result.get("success")),
                "stage": "translate",
                "translated_count": translated_count,
            }
            
            self.info(f"[TranslationTool] 翻译完成: {summary}")
            return TextArtifact(json.dumps(summary, ensure_ascii=False))
        except Exception as e:
            self.error(f"翻译工具执行失败: {e}", e)
            return ErrorArtifact(str(e))

