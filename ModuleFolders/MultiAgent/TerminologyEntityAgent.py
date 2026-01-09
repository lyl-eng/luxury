"""
术语与实体Agent (Agent 1)
负责术语识别、知识库集成和全局一致性保障
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .BaseAgent import BaseAgent
from ModuleFolders.NERProcessor.NERProcessor import NERProcessor
from ModuleFolders.LLMRequester.LLMRequester import LLMRequester
from ModuleFolders.Cache.CacheProject import CacheProject
from ModuleFolders.ResponseExtractor.ResponseExtractor import ResponseExtractor


class TerminologyEntityAgent(BaseAgent):
    """
    Agent 1: 术语与实体Agent
    功能：
    1. 智能术语识别（NER、领域术语、文化负载词）
    2. 知识库集成（RAG）及memory
    3. 全局一致性控制
    """
    
    def __init__(self, config=None):
        super().__init__(
            name="TerminologyEntityAgent",
            description="术语识别与全局一致性保障Agent",
            config=config
        )
        
        self.ner_processor = NERProcessor()
        self.llm_requester = LLMRequester()
        self.response_extractor = ResponseExtractor()
        
        # 术语库存储（项目专属）
        self.terminology_db = {}  # {term: {translation, type, context, strategy}}
        
        # Memory存储
        self.memory_storage = {
            "translated_texts": [],
            "text_summaries": [],
            "reader_preferences": {},
            "translation_style_guide": {}
        }
        
        # 🔥 用于token统计
        self._current_cache_project = None
        
        # 🆕 语言到NER模型的映射
        self.language_model_map = {
            "japanese": "ja_core_news_md",
            "english": "en_core_web_sm",
            "chinese_simplified": "zh_core_web_sm",
            "chinese_traditional": "zh_core_web_sm",
            "korean": "ko_core_news_sm",
            "german": "de_core_news_sm",
            "french": "fr_core_news_sm",
            "spanish": "es_core_news_sm",
            "russian": "ru_core_news_sm"
        }
    
    def _update_token_stats(self, prompt_tokens: int, completion_tokens: int):
        """更新token统计并发送UI更新事件"""
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
        执行术语识别和一致性保障任务
        
        Args:
            input_data: 包含cache_project和metadata的字典
            
        Returns:
            包含术语库和更新后的cache_project的字典
        """
        self.log_agent_action("开始执行术语与实体识别")
        
        cache_project: CacheProject = input_data.get("cache_project")
        metadata = input_data.get("metadata", {})
        
        # 🔥 保存cache_project引用，用于token统计
        self._current_cache_project = cache_project
        
        if not cache_project:
            self.error("未找到cache_project数据")
            return {"success": False, "error": "缺少cache_project"}
        
        self.info("=" * 60)
        self.info("阶段2: 术语与实体识别")
        self.info("=" * 60)
        
        # 🆕 检查是否已存在术语库（支持复用）
        existing_terminology = cache_project.extra.get("terminology_database", {})
        existing_memory = cache_project.extra.get("memory_storage", {})
        
        if existing_terminology and len(existing_terminology) > 0:
            self.info(f"✅ 检测到已有术语库（{len(existing_terminology)} 个术语），直接复用")
            self.terminology_db = existing_terminology
            
            if existing_memory and len(existing_memory) > 0:
                self.memory_storage = existing_memory
                self.info(f"✅ 检测到已有Memory存储，直接复用")
            
            self.info("=" * 60 + "\n")
            self.log_agent_action("术语库复用", f"复用了 {len(self.terminology_db)} 个术语")
            
            return {
                "success": True,
                "cache_project": cache_project,
                "terminology_database": self.terminology_db,
                "memory_storage": self.memory_storage
            }
        
        # 如果没有已有术语库，执行正常的识别流程
        self.info("未检测到术语库，开始智能识别...")
        
        # 1. 智能术语识别
        self.info("→ 执行智能术语识别（NER、领域术语、文化负载词）...")
        terminology_results = self._identify_terminology(cache_project, metadata)
        self.info(f"✓ 识别到 {len(terminology_results)} 个潜在术语")
        
        # 2. 知识库集成（RAG）和查证
        self.info("→ 执行知识库集成与查证...")
        verified_terminology = self._verify_and_enrich_terminology(terminology_results)
        self.info(f"✓ 查证完成，确认 {len(verified_terminology)} 个术语")
        
        # 3. 构建项目专属术语库
        self.info("→ 构建项目专属术语库...")
        self._build_terminology_database(verified_terminology)
        self.info(f"✓ 术语库构建完成，共 {len(self.terminology_db)} 个术语")
        
        # 4. 更新Memory
        self.info("→ 更新Memory存储...")
        self._update_memory(cache_project, metadata)
        self.info(f"✓ Memory更新完成")
        self.info("=" * 60 + "\n")
        
        # 5. 将术语库保存到项目
        cache_project.extra["terminology_database"] = self.terminology_db
        cache_project.extra["memory_storage"] = self.memory_storage
        
        self.log_agent_action("术语识别完成", f"识别到 {len(self.terminology_db)} 个术语")
        
        return {
            "success": True,
            "cache_project": cache_project,
            "terminology_database": self.terminology_db,
            "memory_storage": self.memory_storage
        }
    
    def _identify_terminology(self, cache_project: CacheProject, metadata: Dict) -> List[Dict]:
        """
        智能术语识别
        识别三类关键语言单位：
        1. 命名实体（NER）
        2. 领域术语
        3. 文化负载词与习语
        """
        self.log_agent_action("执行智能术语识别")
        
        all_results = []
        
        # 收集所有文本数据
        items_data = []
        for file_path, cache_file in cache_project.files.items():
            for item in cache_file.items:
                if item.source_text and item.source_text.strip():
                    items_data.append({
                        "source_text": item.source_text,
                        "file_path": file_path
                    })
        
        # 🆕 1. 使用NER处理器识别命名实体（自动根据源语言选择模型）
        ner_model = self._select_ner_model()
        if ner_model:
            self.info(f"→ 使用NER模型识别命名实体: {ner_model}")
            entity_types = ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART"]
            ner_results = self.ner_processor.extract_terms(
                items_data=items_data,
                model_name=ner_model,
                entity_types=entity_types
            )
            
            for result in ner_results:
                result["category"] = "named_entity"
                result["priority"] = "high"  # 命名实体优先级高
            all_results.extend(ner_results)
            self.info(f"✓ NER识别完成，识别到 {len(ner_results)} 个命名实体")
        else:
            self.info("→ 未找到合适的NER模型，跳过NER识别")
        
        # 2. 使用LLM识别领域术语和文化负载词
        domain = metadata.get("domain", "general")
        llm_terminology = self._identify_terminology_with_llm(items_data, domain)
        all_results.extend(llm_terminology)
        
        return all_results
    
    def _select_ner_model(self) -> Optional[str]:
        """
        🆕 根据配置的源语言自动选择合适的NER模型
        
        Returns:
            模型名称，如果没有合适的模型则返回None
        """
        if not self.config:
            return None
        
        # 获取源语言配置
        source_language = getattr(self.config, 'source_language', 'auto')
        
        # 如果是自动检测，尝试从项目中检测主要语言
        if source_language == 'auto':
            self.debug("源语言为自动检测，暂时跳过NER识别（需要实际文本才能检测语言）")
            # TODO: 可以在这里调用语言检测逻辑
            return None
        
        # 根据语言映射选择模型
        model_name = self.language_model_map.get(source_language)
        if not model_name:
            self.debug(f"语言 '{source_language}' 没有对应的NER模型映射")
            return None
        
        # 检查模型文件是否存在
        model_path = os.path.join('.', 'Resource', 'Models', 'ner', model_name)
        if not os.path.exists(model_path):
            self.warning(f"NER模型不存在: {model_path}，跳过NER识别")
            self.info(f"💡 提示: 可以从 https://spacy.io/models 下载 {model_name} 并解压到 {model_path}")
            return None
        
        return model_name
    
    def _identify_terminology_with_llm(self, items_data: List[Dict], domain: str) -> List[Dict]:
        """
        🔥 使用LLM识别领域术语和文化负载词（并行处理）
        直接复用智能分块工具方法
        """
        self.log_agent_action("使用LLM识别领域术语和文化负载词")
        
        # 🔥 直接使用智能分块工具方法（与翻译agent完全相同的逻辑）
        chunks = self._smart_chunk_by_chars(items_data, max_chars=6000, get_text_func=lambda x: x["source_text"])
        
        if len(chunks) > 1:
            self.info(f"  文本较多（{len(items_data)}条），智能分块为 {len(chunks)} 批，并行识别...")
        
        # 🔥 【并行处理】使用线程池并行识别所有批次
        all_terms = []
        
        if len(chunks) == 1:
            # 只有1批，直接串行处理
            all_terms = self._identify_chunk_terms(chunks[0], 1, len(chunks), domain)
        else:
            # 多批并行处理
            with ThreadPoolExecutor(max_workers=min(len(chunks), 5)) as executor:
                # 提交所有任务
                future_to_chunk = {
                    executor.submit(self._identify_chunk_terms, chunk, idx, len(chunks), domain): idx
                    for idx, chunk in enumerate(chunks, 1)
                }
                
                # 收集结果
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    try:
                        chunk_terms = future.result()
                        all_terms.extend(chunk_terms)
                    except Exception as e:
                        self.error(f"第 {chunk_idx} 批术语识别失败: {e}", e)
        
        # 去重（基于术语名称）
        unique_terms = {}
        for term in all_terms:
            term_name = term.get("term", "").lower()
            if term_name and term_name not in unique_terms:
                unique_terms[term_name] = term
        
        final_terms = list(unique_terms.values())
        if len(chunks) > 1:
            self.info(f"✓ 并行识别完成，总计识别到 {len(final_terms)} 个独立术语（去重后）")
        
        return final_terms
    
    def _identify_chunk_terms(self, chunk: List[Dict], chunk_idx: int, total_chunks: int, domain: str) -> List[Dict]:
        """
        🆕 识别单个chunk的术语（用于并行处理）
        
        Args:
            chunk: 待识别的文本chunk
            chunk_idx: 当前chunk索引
            total_chunks: 总chunk数量
            domain: 领域
            
        Returns:
            识别到的术语列表
        """
        # 构建提示词（取每个item的前200字符）
        sample_texts = [item["source_text"][:200] for item in chunk]
        sample_text = "\n---\n".join(sample_texts)
        
        system_prompt = f"""你是一位专业的术语识别专家。请从以下文本中识别：
1. 领域术语：专业或领域特有的词汇和短语（如"{domain}"领域的专业术语）
2. 文化负载词：缺乏直接对等表达的词汇和习语

注意：
- 只识别真正需要固定翻译的术语（如专有名词、专业术语）
- 不要识别普通词汇
- 优先识别出现频率高的术语

请以JSON格式返回识别结果，格式如下：
{{
    "terms": [
        {{
            "term": "术语原文",
            "category": "domain_term" 或 "cultural_expression",
            "context": "出现上下文",
            "meaning": "语义解释",
            "translation_strategy": "直译/意译/语义补偿"
        }}
    ]
}}"""
        
        messages = [{
            "role": "user",
            "content": f"请识别以下文本中的领域术语和文化负载词：\n\n{sample_text}"
        }]
        
        # 调用LLM
        try:
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 解析JSON响应
                try:
                    json_start = response_content.find("{")
                    json_end = response_content.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_content[json_start:json_end]
                        result = json.loads(json_str)
                        
                        chunk_terms = []
                        for term_info in result.get("terms", []):
                            chunk_terms.append({
                                "term": term_info.get("term"),
                                "type": term_info.get("category", "unknown"),
                                "context": term_info.get("context", ""),
                                "meaning": term_info.get("meaning", ""),
                                "translation_strategy": term_info.get("translation_strategy", ""),
                                "category": term_info.get("category", "domain_term"),
                                "priority": "medium"
                            })
                        
                        if total_chunks > 1:
                            self.info(f"  ✓ 第 {chunk_idx}/{total_chunks} 批识别到 {len(chunk_terms)} 个术语")
                        
                        return chunk_terms
                        
                except json.JSONDecodeError:
                    self.warning(f"第 {chunk_idx} 批LLM返回的JSON格式不正确")
        except Exception as e:
            self.error(f"第 {chunk_idx} 批LLM术语识别失败: {e}", e)
        
        return []
    
    
    def _smart_chunk_by_chars(self, items: List, max_chars: int, get_text_func) -> List[List]:
        """
        🔥 【通用智能分块工具】- 按字符数智能分块
        可用于任何需要分块的场景（术语识别、翻译、查证等）
        
        这是从翻译agent提取的核心分块逻辑，保证所有agent使用完全相同的分块策略
        
        Args:
            items: 待分块的列表（可以是任何类型）
            max_chars: 单个chunk的最大字符数
            get_text_func: 从item中提取文本的函数（例如：lambda x: x["source_text"]）
            
        Returns:
            chunks: 分块后的列表
            
        Example:
            # 分块Dict列表
            chunks = _smart_chunk_by_chars(items_data, 6000, lambda x: x["source_text"])
            
            # 分块术语列表（术语较短，可以设置更大的batch）
            batches = _smart_chunk_by_chars(terms_list, 3000, lambda x: x.get("term", ""))
        """
        chunks = []
        current_chunk = []
        chunk_chars = 0
        
        for item in items:
            text = get_text_func(item)
            text_length = len(text)
            
            # 🔥 极端超长文本（超过max_chars）单独成chunk
            if text_length > max_chars:
                # 先提交当前chunk（如果有）
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    chunk_chars = 0
                
                # 极端超长文本单独成chunk
                chunks.append([item])
                continue
            
            # 🔥 智能打包：按总字符数限制
            # 如果加入当前item会超过max_chars，先提交当前chunk
            if current_chunk and (chunk_chars + text_length > max_chars):
                chunks.append(current_chunk)
                current_chunk = []
                chunk_chars = 0
            
            # 添加当前item到chunk
            current_chunk.append(item)
            chunk_chars += text_length
        
        # 处理最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _verify_and_enrich_terminology(self, terminology_results: List[Dict]) -> List[Dict]:
        """
        知识库集成（RAG）及查证
        利用外部资源对识别出的实体和术语进行查证
        🆕 批量处理优化：一次LLM调用查证所有新术语
        """
        self.log_agent_action("执行术语查证和知识库集成")
        
        verified_results = []
        new_terms_to_enrich = []  # 需要LLM查证的新术语
        new_terms_indices = []    # 记录新术语在结果列表中的位置
        
        # 第一遍：分离已有术语和新术语
        for idx, term_info in enumerate(terminology_results):
            term = term_info.get("term")
            
            # 1. 检查是否已存在于术语库
            if term in self.terminology_db:
                verified_info = self.terminology_db[term].copy()
                verified_info.update(term_info)
                verified_results.append(verified_info)
            else:
                # 2. 收集需要查证的新术语
                new_terms_to_enrich.append(term_info)
                new_terms_indices.append(len(verified_results))
                verified_results.append(None)  # 占位
        
        # 批量查证新术语（如果有）
        if new_terms_to_enrich:
            self.info(f"→ 批量查证 {len(new_terms_to_enrich)} 个新术语...")
            
            # 🔥 直接使用智能分块工具方法（与领域识别、翻译完全相同的逻辑）
            # 术语相对较短，可以设置较大的batch以减少LLM调用次数
            batches = self._smart_chunk_by_chars(new_terms_to_enrich, max_chars=3000, get_text_func=lambda x: x.get("term", ""))
            all_enriched = []
            
            for batch_num, batch in enumerate(batches, 1):
                if len(batches) > 1:
                    self.info(f"  处理第 {batch_num}/{len(batches)} 批（{len(batch)} 个术语）")
                
                enriched_batch = self._batch_enrich_terms_with_llm(batch)
                all_enriched.extend(enriched_batch)
            
            # 填充查证结果
            for idx, enriched_info in zip(new_terms_indices, all_enriched):
                verified_results[idx] = enriched_info
        
        return verified_results
    
    def _batch_enrich_terms_with_llm(self, terms_list: List[Dict]) -> List[Dict]:
        """
        🔥 批量使用LLM查证术语翻译
        采用与翻译agent相同的 <textarea> 格式 + ResponseExtractor 解析
        
        Args:
            terms_list: 待查证的术语列表
            
        Returns:
            查证后的术语列表
        """
        if not terms_list:
            return []
        
        # 🔥 构建批量翻译的prompt（使用 textarea 格式）
        terms_text = []
        for idx, term_info in enumerate(terms_list, 1):
            term = term_info.get("term")
            category = term_info.get("category", "unknown")
            context = term_info.get("context", "")[:50]  # 只取前50字符
            terms_text.append(f"{idx}. {term}")
        
        system_prompt = f"""你是一位专业的术语翻译专家。你的任务是为以下术语提供准确的中文翻译。

【翻译要求】
1. 根据术语的类型选择合适的翻译策略：
   - 专有名词（人名、地名）：音译为主
   - 生物/化学术语：使用标准学术译名
   - 普通术语：意译，符合中文习惯
2. 翻译必须准确、规范，符合专业领域的惯例
3. 不要添加任何解释或注释

【输出格式要求】
- 必须使用<textarea>标签包裹所有译文
- 逐行翻译，原文有{len(terms_list)}行，译文也必须有{len(terms_list)}行
- 每行格式：序号. 译文
- 序号必须从1到{len(terms_list)}连续，不要跳过
- 不要合并行，不要添加额外说明

格式示例：
<textarea>
1.第一个术语的译文
2.第二个术语的译文
3.第三个术语的译文
</textarea>"""
        
        # 构建用户消息
        user_content = "请为以下术语提供准确的中文翻译：\n\n<textarea>\n" + "\n".join(terms_text) + "\n</textarea>"
        
        messages = [{
            "role": "user",
            "content": user_content
        }]
        
        try:
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 🔥 使用 ResponseExtractor 解析（与翻译agent相同）
                source_text_dict = {str(i): term_info.get("term") for i, term_info in enumerate(terms_list)}
                translation_dict = self.response_extractor.extract_translation(source_text_dict, response_content)
                
                if translation_dict:
                    # 将翻译结果合并到术语信息中
                    success_count = 0
                    for idx, term_info in enumerate(terms_list):
                        translation = translation_dict.get(str(idx), "")
                        
                        # 清理序号前缀（如 "1." 或 "1. "）
                        translation = re.sub(r'^\d+\.\s*', '', translation).strip()
                        
                        # 清理Markdown标记
                        translation = re.sub(r'\*\*(.+?)\*\*', r'\1', translation)
                        translation = re.sub(r'\*(.+?)\*', r'\1', translation)
                        translation = translation.strip('*_').strip()
                        
                        if translation:
                            term_info["translation_suggestions"] = [translation]
                            term_info["llm_verification"] = "批量查证完成"
                            success_count += 1
                    
                    self.info(f"✓ 批量查证完成，成功处理 {success_count}/{len(terms_list)} 个术语")
                    return terms_list
                else:
                    self.warning("ResponseExtractor未能解析出翻译结果")
                    
        except Exception as e:
            self.error(f"批量术语查证失败: {e}", e)
        
        # 如果批量查证失败，返回原始信息
        return terms_list
    
    def _enrich_term_with_llm(self, term_info: Dict) -> Dict:
        """
        ⚠️ 已弃用：使用LLM丰富单个术语信息
        现在使用 _batch_enrich_terms_with_llm 进行批量处理
        保留此方法以防后续需要单独查证某些术语
        """
        term = term_info.get("term")
        category = term_info.get("category", "unknown")
        
        system_prompt = f"""你是一位专业的术语翻译专家。请为以下术语提供：
1. 准确的翻译建议
2. 使用场景说明
3. 翻译策略建议

术语：{term}
类别：{category}"""
        
        messages = [{
            "role": "user",
            "content": f"请为术语'{term}'提供翻译建议和说明。"
        }]
        
        try:
            platform_config = self.config.get_platform_configuration("translationReq") if self.config else {}
            skip, _, response_content, prompt_tokens, completion_tokens = self._llm_request_with_tracking(
                messages=messages,
                system_prompt=system_prompt,
                platform_config=platform_config
            )
            
            # 🔥 更新token统计
            self._update_token_stats(prompt_tokens, completion_tokens)
            
            if not skip and response_content:
                # 解析响应，提取翻译建议
                # 这里简化处理，实际可以更智能地解析
                term_info["llm_verification"] = response_content
                term_info["translation_suggestions"] = self._extract_translation_suggestions(response_content)
        except Exception as e:
            self.error(f"术语查证失败 {term}: {e}", e)
        
        return term_info
    
    def _extract_translation_suggestions(self, llm_response: str) -> List[str]:
        """
        从LLM响应中提取翻译建议
        ✅ 清理所有Markdown格式标记（**，__，*，_等）
        """
        import re
        suggestions = []
        # 简单的提取逻辑
        lines = llm_response.split("\n")
        for line in lines:
            if "翻译" in line or "译" in line:
                # 尝试提取可能的翻译
                parts = line.split("：") or line.split(":")
                if len(parts) > 1:
                    translation = parts[1].strip()
                    # ✅ 清理Markdown格式标记
                    # 移除粗体：**text** 或 __text__
                    translation = re.sub(r'\*\*(.+?)\*\*', r'\1', translation)
                    translation = re.sub(r'__(.+?)__', r'\1', translation)
                    # 移除斜体：*text* 或 _text_
                    translation = re.sub(r'\*(.+?)\*', r'\1', translation)
                    translation = re.sub(r'_(.+?)_', r'\1', translation)
                    # 移除行首行尾的多余空格和标点
                    translation = translation.strip('*_').strip()
                    if translation:
                        suggestions.append(translation)
        return suggestions[:3]  # 返回前3个建议
    
    def _build_terminology_database(self, verified_terminology: List[Dict]) -> None:
        """
        构建项目专属术语库（结构化资源）
        并将术语同步到 ElasticSearch (Phase 2)
        """
        self.log_agent_action("构建项目专属术语库并同步到DB")
        
        # 1. 内存更新
        for term_info in verified_terminology:
            term = term_info.get("term")
            if not term:
                continue
            
            # 构建术语库条目
            self.terminology_db[term] = {
                "term": term,
                "type": term_info.get("type", "unknown"),
                "category": term_info.get("category", "unknown"),
                "translation": term_info.get("translation_suggestions", [""])[0] if term_info.get("translation_suggestions") else "",
                "context": term_info.get("context", ""),
                "meaning": term_info.get("meaning", ""),
                "translation_strategy": term_info.get("translation_strategy", "直译"),
                "priority": term_info.get("priority", "medium"),
                "verified": True
            }

        # 2. 数据库同步 (ES) - 写入完整词汇信息
        try:
            from ModuleFolders.Cache.DatabaseManager import DatabaseManager
            db_manager = DatabaseManager()
            
            # 获取当前 work_id (默认 0)
            work_id = getattr(self._current_cache_project, 'db_work_id', 0)
            
            # 批量写入术语到 ES（包含完整词汇信息）
            for term, info in self.terminology_db.items():
                # 确定词汇类型
                term_type = info.get("type", "term")
                word_type_map = {
                    "named_entity": "entity",
                    "terminology": "term",
                    "cultural_expression": "idiom",
                    "unknown": "term"
                }
                word_type = word_type_map.get(term_type, "term")
                
                # 构建候选译法列表
                translations_list = []
                main_translation = info.get("translation", "")
                if main_translation:
                    translations_list.append({
                        "translation": main_translation,
                        "source": "LLM",
                        "confidence": info.get("confidence", 1.0),
                        "rank": 1,
                        "rationale": info.get("translation_strategy", "")
                    })
                
                # 收集相关原子ID（如果有的话）
                atom_ids = []
                if hasattr(self._current_cache_project, 'db_atom_map'):
                    # 遍历所有文件的 atom_map，找出包含这个术语的原子
                    for file_path, atom_map in self._current_cache_project.db_atom_map.items():
                        for row_idx, a_id in atom_map.items():
                            atom_ids.append(a_id)
                    # 限制数量避免过大
                    atom_ids = atom_ids[:10] if len(atom_ids) > 10 else atom_ids
                
                db_manager.upsert_term(
                    entry_key=term,
                    entry_val=main_translation,
                    work_id=work_id,
                    word_type=word_type,
                    domain=self.memory_storage.get("domain", "general"),
                    variants=[],  # TODO: 还没提取变体
                    example_sentences=[info.get("context", "")] if info.get("context") else [],
                    translations=translations_list,
                    atom_ids=atom_ids,
                    confidence=info.get("confidence", 1.0),
                    agent_notes=f"类型: {info.get('category', '')}, 含义: {info.get('meaning', '')}",
                    is_confirmed=info.get("verified", False)
                )
            
            self.info(f"[DB] 术语已同步到 ElasticSearch: {len(self.terminology_db)} 个条目 (Project ID: {work_id})")
            
        except Exception as e:
            self.error(f"[DB] 术语库同步失败: {e}")
    
    def _update_memory(self, cache_project: CacheProject, metadata: Dict) -> None:
        """
        更新Memory存储
        存储已翻译文本、摘要、读者倾向、翻译风格指南等
        """
        self.log_agent_action("更新Memory存储")
        
        # 存储元数据
        self.memory_storage["domain"] = metadata.get("domain", "general")
        self.memory_storage["style"] = metadata.get("style", "neutral")
        
        # 存储已翻译文本摘要（如果有）
        translated_texts = []
        for cache_file in cache_project.files.values():
            for item in cache_file.items:
                if hasattr(item, 'translated_text') and item.translated_text:
                    translated_texts.append({
                        "source": item.source_text[:100],  # 只存储前100字符
                        "translated": item.translated_text[:100]
                    })
        
        if translated_texts:
            self.memory_storage["translated_texts"] = translated_texts[-50:]  # 只保留最近50条
    
    def get_terminology_prompt(self) -> str:
        """
        生成术语表提示词，用于强制模型使用规范译法
        """
        if not self.terminology_db:
            return ""
        
        prompt = "\n\n【术语表】请严格按照以下术语表进行翻译，确保全文一致性：\n"
        for term, info in self.terminology_db.items():
            translation = info.get("translation", "")
            if translation:
                prompt += f"- {term} → {translation}\n"
        
        return prompt
    
    def get_memory_context(self) -> str:
        """
        获取Memory上下文，用于动态加载到prompt中
        """
        context_parts = []
        
        # 领域和风格信息
        domain = self.memory_storage.get("domain", "general")
        style = self.memory_storage.get("style", "neutral")
        if domain != "general":
            context_parts.append(f"文本领域：{domain}")
        if style != "neutral":
            context_parts.append(f"文本风格：{style}")
        
        # 翻译风格指南
        style_guide = self.memory_storage.get("translation_style_guide", {})
        if style_guide:
            context_parts.append(f"翻译风格指南：{json.dumps(style_guide, ensure_ascii=False)}")
        
        return "\n".join(context_parts) if context_parts else ""
