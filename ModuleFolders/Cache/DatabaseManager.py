"""
AiNiee 多智能体翻译系统 - 数据库管理器

提供完整的数据库操作接口，支持：
1. 项目和文档管理
2. 处理原子的 CRUD 操作
3. Agent 轨迹记录（事件溯源）
4. 上下文信息管理
5. 术语库检索（ElasticSearch）
"""

import psycopg2
from psycopg2.extras import execute_values, Json, RealDictCursor
from elasticsearch import Elasticsearch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import datetime
import hashlib


class DatabaseManager:
    """数据库管理器 - 单例模式"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config=None):
        if hasattr(self, 'initialized'):
            return
        
        # PostgreSQL 配置
        self.pg_config = config or {
            "host": "localhost",
            "port": 5432,
            "user": "admin",
            "password": "password",
            "dbname": "ainiee_db"
        }
        
        # ElasticSearch 配置
        self.es_url = "http://localhost:9200"
        try:
            self.es = Elasticsearch(self.es_url)
        except:
            self.es = None
            
        self.initialized = True

    def get_pg_connection(self):
        """获取 PostgreSQL 连接"""
        try:
            return psycopg2.connect(**self.pg_config)
        except Exception as e:
            print(f"[DB] PostgreSQL 连接失败: {e}")
            raise e

    # ==========================================
    # 项目管理 (Project Works)
    # ==========================================

    def create_project(self, name: str, src_lang: str, tgt_lang: str, 
                      workflow_config: Dict = None, topic_info: Dict = None,
                      translation_guide: Dict = None, prompt_templates: Dict = None) -> Optional[int]:
        """
        创建翻译项目
        
        Args:
            name: 项目名称
            src_lang: 源语言
            tgt_lang: 目标语言
            workflow_config: 工作流配置
            topic_info: 文本主题知识 {domain, topic, summary, style}
            translation_guide: 翻译指南 {style_requirements, fixed_translations, cultural_adaptation}
            prompt_templates: 提示词模板库
        
        Returns:
            work_id: 项目ID
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            sql = """
                INSERT INTO project_works 
                (work_name, src_lang, tgt_lang, workflow_config, topic_info, translation_guide, prompt_templates)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING work_id;
            """
            cur.execute(sql, (
                name, src_lang, tgt_lang, 
                Json(workflow_config or {}),
                Json(topic_info or {}),
                Json(translation_guide or {}),
                Json(prompt_templates or {})
            ))
            work_id = cur.fetchone()[0]
            conn.commit()
            print(f"[DB] 创建项目成功: work_id={work_id}, name={name}")
            return work_id
        except Exception as e:
            conn.rollback()
            print(f"[DB] 创建项目失败: {e}")
            return None
        finally:
            conn.close()

    def update_project_topic_info(self, work_id: int, topic_info: Dict):
        """
        更新项目的文本主题知识
        
        Args:
            work_id: 项目ID
            topic_info: {domain, topic, summary, style}
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE project_works SET topic_info = %s WHERE work_id = %s",
                (Json(topic_info), work_id)
            )
            conn.commit()
            print(f"[DB] 更新项目主题知识: work_id={work_id}")
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新项目主题知识失败: {e}")
        finally:
            conn.close()

    def update_project_translation_guide(self, work_id: int, translation_guide: Dict):
        """
        更新项目的翻译指南
        
        Args:
            work_id: 项目ID
            translation_guide: {style_requirements, fixed_translations, cultural_adaptation}
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE project_works SET translation_guide = %s WHERE work_id = %s",
                (Json(translation_guide), work_id)
            )
            conn.commit()
            print(f"[DB] 更新项目翻译指南: work_id={work_id}")
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新项目翻译指南失败: {e}")
        finally:
            conn.close()

    def get_project(self, work_id: int) -> Optional[Dict]:
        """获取项目信息"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT * FROM project_works WHERE work_id = %s", (work_id,))
            result = cur.fetchone()
            return dict(result) if result else None
        finally:
            conn.close()

    # ==========================================
    # 文档管理 (Source Docs)
    # ==========================================

    def create_document(self, work_id: int, file_path: str, 
                       file_name: str = None, doc_meta: Dict = None) -> Optional[int]:
        """
        创建源文档记录
        
        Returns:
            doc_id: 文档ID
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            sql = """
                INSERT INTO source_docs (work_id, file_path, file_name, doc_meta)
                VALUES (%s, %s, %s, %s)
                RETURNING doc_id;
            """
            cur.execute(sql, (work_id, file_path, file_name or file_path, Json(doc_meta or {})))
            doc_id = cur.fetchone()[0]
            conn.commit()
            print(f"[DB] 创建文档成功: doc_id={doc_id}, file={file_name or file_path}")
            return doc_id
        except Exception as e:
            conn.rollback()
            print(f"[DB] 创建文档失败: {e}")
            return None
        finally:
            conn.close()

    def update_document_atom_count(self, doc_id: int, total_atoms: int):
        """更新文档的原子总数"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE source_docs SET total_atoms = %s, status = 'processed' WHERE doc_id = %s",
                (total_atoms, doc_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新文档原子数失败: {e}")
        finally:
            conn.close()

    # ==========================================
    # 处理原子 (Processing Atoms) - 核心操作
    # ==========================================

    def create_atoms_batch(self, doc_id: int, atoms: List[Dict]) -> List[int]:
        """
        批量创建处理原子
        
        Args:
            doc_id: 文档ID
            atoms: 原子列表，每个原子包含:
                - source_text: 原文 (必须)
                - position: 位置索引
                - summary: 片段摘要 (可选)
                - context_info: 上下文信息 (可选)
                
        Returns:
            atom_ids: 创建的原子ID列表
        """
        if not atoms:
            return []
            
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            
            # 准备批量插入数据
            values = []
            for idx, atom in enumerate(atoms):
                source_text = atom.get('source_text', '')
                source_hash = hashlib.md5(source_text.encode()).hexdigest()
                position = atom.get('position', idx)
                summary = atom.get('summary', '')
                context_info = atom.get('context_info', {})
                
                values.append((
                    doc_id,
                    source_text,
                    source_hash,
                    position,
                    summary,
                    Json(context_info)
                ))
            
            # 批量插入
            sql = """
                INSERT INTO processing_atoms 
                (doc_id, source_text, source_hash, position, summary, context_info)
                VALUES %s
                RETURNING atom_id;
            """
            execute_values(cur, sql, values, template="(%s, %s, %s, %s, %s, %s)")
            
            # 获取返回的 atom_ids
            # 注意: execute_values 不能直接返回多个ID，需要重新查询
            cur.execute(
                "SELECT atom_id FROM processing_atoms WHERE doc_id = %s ORDER BY position",
                (doc_id,)
            )
            atom_ids = [row[0] for row in cur.fetchall()]
            
            conn.commit()
            print(f"[DB] 批量创建原子成功: doc_id={doc_id}, count={len(atom_ids)}")
            return atom_ids[-len(atoms):]  # 返回最后插入的N个ID
            
        except Exception as e:
            conn.rollback()
            print(f"[DB] 批量创建原子失败: {e}")
            return []
        finally:
            conn.close()

    def get_atom(self, atom_id: int) -> Optional[Dict]:
        """获取单个原子信息"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute("SELECT * FROM processing_atoms WHERE atom_id = %s", (atom_id,))
            result = cur.fetchone()
            return dict(result) if result else None
        finally:
            conn.close()

    def get_atoms_by_doc(self, doc_id: int) -> List[Dict]:
        """获取文档下的所有原子"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT * FROM processing_atoms WHERE doc_id = %s ORDER BY position",
                (doc_id,)
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def update_atom_translation(self, atom_id: int, translated_text: str, 
                                status_code: int = 1, quality_score: float = None):
        """
        更新原子的翻译结果
        
        Args:
            atom_id: 原子ID
            translated_text: 译文
            status_code: 状态码 (1=已初翻, 2=已润色, 3=已审核, 4=已完成)
            quality_score: 质量评分 (0-10)
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            sql = """
                UPDATE processing_atoms 
                SET translated_text = %s, 
                    status_code = %s,
                    quality_score = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE atom_id = %s
            """
            cur.execute(sql, (translated_text, status_code, quality_score, atom_id))
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新原子翻译失败: {e}")
        finally:
            conn.close()

    def update_atom_summary(self, atom_id: int, summary: str):
        """更新原子的片段摘要"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE processing_atoms SET summary = %s, updated_at = CURRENT_TIMESTAMP WHERE atom_id = %s",
                (summary, atom_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新原子摘要失败: {e}")
        finally:
            conn.close()

    def update_atom_examination(self, atom_id: int, examination: Dict):
        """
        更新原子的质量检查信息
        
        Args:
            atom_id: 原子ID
            examination: 质量检查信息 {
                "back_translation": "回译结果",
                "warning_level": "high/medium/low",
                "semantic_similarity": 0.95,
                "issues": ["问题1", "问题2"],
                "algorithm": "backtranslation"
            }
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE processing_atoms SET examination = %s, updated_at = CURRENT_TIMESTAMP WHERE atom_id = %s",
                (Json(examination), atom_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新原子质量检查信息失败: {e}")
        finally:
            conn.close()

    def update_atom_context(self, atom_id: int, context_info: Dict):
        """
        更新原子的上下文信息
        
        context_info 结构:
        {
            "prev_source": "前一原子原文",
            "prev_translated": "前一原子译文",
            "next_source": "后一原子原文",
            "terminology": [{"term": "xxx", "translation": "xxx"}],
            "memory_refs": [{"source": "xxx", "translation": "xxx", "score": 0.95}],
            "similar_atoms": [{"source": "xxx", "translated": "xxx"}]
        }
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE processing_atoms SET context_info = %s, updated_at = CURRENT_TIMESTAMP WHERE atom_id = %s",
                (Json(context_info), atom_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新原子上下文失败: {e}")
        finally:
            conn.close()

    def update_atom_vector(self, atom_id: int, vector: List[float]):
        """更新原子的语义向量"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            cur.execute(
                "UPDATE processing_atoms SET semantic_vec = %s WHERE atom_id = %s",
                (vector, atom_id)
            )
            conn.commit()
        except Exception as e:
            conn.rollback()
            print(f"[DB] 更新原子向量失败: {e}")
        finally:
            conn.close()

    def batch_update_atoms(self, updates: List[Dict]):
        """
        批量更新原子
        
        Args:
            updates: 更新列表，每个包含:
                - atom_id: 原子ID (必须)
                - translated_text: 译文 (可选)
                - status_code: 状态码 (可选)
                - quality_score: 质量评分 (可选)
                - context_info: 上下文信息 (可选)
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            for update in updates:
                atom_id = update.get('atom_id')
                if not atom_id:
                    continue
                    
                # 构建动态 UPDATE 语句
                set_clauses = []
                values = []
                
                if 'translated_text' in update:
                    set_clauses.append("translated_text = %s")
                    values.append(update['translated_text'])
                if 'status_code' in update:
                    set_clauses.append("status_code = %s")
                    values.append(update['status_code'])
                if 'quality_score' in update:
                    set_clauses.append("quality_score = %s")
                    values.append(update['quality_score'])
                if 'context_info' in update:
                    set_clauses.append("context_info = %s")
                    values.append(Json(update['context_info']))
                    
                if set_clauses:
                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    sql = f"UPDATE processing_atoms SET {', '.join(set_clauses)} WHERE atom_id = %s"
                    values.append(atom_id)
                    cur.execute(sql, values)
            
            conn.commit()
            print(f"[DB] 批量更新原子成功: count={len(updates)}")
        except Exception as e:
            conn.rollback()
            print(f"[DB] 批量更新原子失败: {e}")
        finally:
            conn.close()

    # ==========================================
    # Agent 轨迹 (Agent Traces) - 事件溯源
    # ==========================================

    def add_trace(self, atom_id: int, agent_role: str, action_type: str,
                 content: str = None, quality_report: Dict = None,
                 meta_data: Dict = None, input_tokens: int = 0, 
                 output_tokens: int = 0) -> int:
        """
        记录 Agent 操作轨迹
        
        Args:
            atom_id: 原子ID
            agent_role: 操作角色 (Translator, Reviewer, ConsistencyChecker, Human)
            action_type: 动作类型 (draft, refine, evaluate, final, human_edit)
            content: 产出内容 (译文/评分/意见)
            quality_report: 质量报告 {
                "score": 8.5,
                "back_translation": "回译结果",
                "issues": ["问题1", "问题2"],
                "status": "pass" | "needs_refinement"
            }
            meta_data: 其他元数据 (使用的策略、模型等)
            input_tokens: 输入token数
            output_tokens: 输出token数
            
        Returns:
            trace_id: 轨迹ID
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            
            # 判断是否是翻译类操作（需要设置 is_active）
            is_translation_action = action_type in ['draft', 'refine', 'final', 'human_edit']
            
            # 如果是翻译操作，先将该原子之前的记录设为非激活
            if is_translation_action:
                cur.execute(
                    "UPDATE agent_traces SET is_active = FALSE WHERE atom_id = %s AND is_active = TRUE",
                    (atom_id,)
                )
            
            # 插入新轨迹
            sql = """
                INSERT INTO agent_traces 
                (atom_id, agent_role, action_type, content, quality_report, 
                 meta_data, input_tokens, output_tokens, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING trace_id;
            """
            cur.execute(sql, (
                atom_id,
                agent_role,
                action_type,
                content,
                Json(quality_report or {}),
                Json(meta_data or {}),
                input_tokens,
                output_tokens,
                is_translation_action
            ))
            trace_id = cur.fetchone()[0]
            conn.commit()
            return trace_id
            
        except Exception as e:
            conn.rollback()
            print(f"[DB] 添加轨迹失败: {e}")
            return -1
        finally:
            conn.close()

    def add_trace_batch(self, traces: List[Dict]) -> List[int]:
        """
        批量添加 Agent 轨迹
        
        Args:
            traces: 轨迹列表，每个包含:
                - atom_id, agent_role, action_type, content, quality_report, meta_data
        """
        trace_ids = []
        for trace in traces:
            trace_id = self.add_trace(
                atom_id=trace.get('atom_id'),
                agent_role=trace.get('agent_role', 'Unknown'),
                action_type=trace.get('action_type', 'unknown'),
                content=trace.get('content'),
                quality_report=trace.get('quality_report'),
                meta_data=trace.get('meta_data'),
                input_tokens=trace.get('input_tokens', 0),
                output_tokens=trace.get('output_tokens', 0)
            )
            trace_ids.append(trace_id)
        return trace_ids

    def get_atom_traces(self, atom_id: int) -> List[Dict]:
        """获取原子的所有轨迹记录"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            cur.execute(
                "SELECT * FROM agent_traces WHERE atom_id = %s ORDER BY created_at",
                (atom_id,)
            )
            return [dict(row) for row in cur.fetchall()]
        finally:
            conn.close()

    def get_active_translation(self, atom_id: int) -> Optional[str]:
        """获取原子当前激活的译文"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT content FROM agent_traces WHERE atom_id = %s AND is_active = TRUE LIMIT 1",
                (atom_id,)
            )
            result = cur.fetchone()
            return result[0] if result else None
        finally:
            conn.close()

    # ==========================================
    # 知识库 (Knowledge Base) - RAG 支持
    # ==========================================

    def add_knowledge(self, work_id: int, content: str, kb_type: str,
                     vector: List[float] = None, meta_tags: Dict = None,
                     source_ref: str = None) -> Optional[int]:
        """
        添加知识条目
        
        Args:
            work_id: 项目ID
            content: 知识内容
            kb_type: 类型 ('tm'=翻译记忆, 'glossary'=术语, 'style_guide'=风格指南, 'external'=外部知识)
            vector: 语义向量 (768维)
            meta_tags: 标签元数据
            source_ref: 来源引用
            
        Returns:
            kb_id: 知识条目ID
        """
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            
            vec = None
            if vector:
                if isinstance(vector, np.ndarray):
                    vec = vector.tolist()
                else:
                    vec = vector
            
            sql = """
                INSERT INTO knowledge_base (work_id, content, kb_type, semantic_vec, meta_tags, source_ref)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING kb_id;
            """
            cur.execute(sql, (work_id, content, kb_type, vec, Json(meta_tags or {}), source_ref))
            kb_id = cur.fetchone()[0]
            conn.commit()
            return kb_id
        except Exception as e:
            conn.rollback()
            print(f"[DB] 添加知识条目失败: {e}")
            return None
        finally:
            conn.close()

    def search_knowledge(self, query_vec: List[float], work_id: int = None,
                        kb_type: str = None, limit: int = 5) -> List[Dict]:
        """
        RAG 检索：从知识库找相似内容
        
        Args:
            query_vec: 查询向量
            work_id: 限定项目ID
            kb_type: 限定类型 ('tm', 'glossary', 'style_guide', 'external')
            limit: 返回数量限制
            
        Returns:
            相似知识列表
        """
        if not query_vec:
            return []
            
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = [query_vec]
            
            if work_id:
                conditions.append("work_id = %s")
                params.append(work_id)
            if kb_type:
                conditions.append("kb_type = %s")
                params.append(kb_type)
            
            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params.append(limit)
            
            sql = f"""
                SELECT kb_id, content, kb_type, meta_tags, 
                       1 - (semantic_vec <=> %s::vector) as similarity
                FROM knowledge_base
                WHERE {where_clause} AND semantic_vec IS NOT NULL
                ORDER BY similarity DESC
                LIMIT %s;
            """
            cur.execute(sql, params)
            
            return [{
                "kb_id": row[0],
                "content": row[1],
                "kb_type": row[2],
                "meta_tags": row[3],
                "similarity": float(row[4])
            } for row in cur.fetchall()]
        except Exception as e:
            print(f"[DB] 知识库检索失败: {e}")
            return []
        finally:
            conn.close()

    def add_translation_memory(self, work_id: int, source: str, translation: str,
                              vector: List[float] = None, domain: str = None) -> Optional[int]:
        """
        添加翻译记忆 (TM)
        
        Args:
            work_id: 项目ID
            source: 原文
            translation: 译文
            vector: 原文语义向量
            domain: 领域标签
        """
        content = f"{source}\t{translation}"  # 用制表符分隔原文和译文
        meta_tags = {"source": source, "translation": translation}
        if domain:
            meta_tags["domain"] = domain
        return self.add_knowledge(work_id, content, "tm", vector, meta_tags)

    # ==========================================
    # 术语库 (ElasticSearch)
    # ==========================================

    def upsert_term(self, entry_key: str, entry_val: str, work_id: int,
                   word_type: str = "term", domain: str = "general", 
                   variants: List[str] = None, example_sentences: List[str] = None, 
                   translations: List[Dict] = None, atom_ids: List[int] = None,
                   confidence: float = 1.0, source_ref: str = None, 
                   agent_notes: str = None, is_confirmed: bool = False):
        """
        添加或更新术语/词汇
        
        Args:
            entry_key: 词汇原文
            entry_val: 最终译文
            work_id: 项目ID
            word_type: 词汇类型 (term/entity/keyword/acronym/idiom/concept)
            domain: 领域标签
            variants: 词汇变体列表
            example_sentences: 例句列表
            translations: 候选译法列表 [{translation, source, confidence, rank, rationale}]
            atom_ids: 相关翻译片段ID列表
            confidence: 最终置信度 (0-1)
            source_ref: 来源引用
            agent_notes: Agent备注
            is_confirmed: 是否人工确认
        """
        if not self.es:
            return
            
        try:
            doc_id = f"{work_id}_{entry_key}"
            doc = {
                "entry_key": entry_key,
                "entry_val": entry_val,
                "work_id": work_id,
                "word_type": word_type,
                "domain_tag": domain,
                "variants": [{"text": v, "lang": "auto"} for v in (variants or [])],
                "example_sentences": example_sentences or [],
                "translations": translations or [],
                "atom_ids": atom_ids or [],
                "confidence": confidence,
                "source_ref": source_ref,
                "agent_notes": agent_notes,
                "is_confirmed": is_confirmed,
                "updated_at": datetime.datetime.now().isoformat()
            }
            self.es.index(index="domain_lexicon", id=doc_id, body=doc)
            print(f"[DB] 术语更新成功: {entry_key} -> {entry_val}")
        except Exception as e:
            print(f"[DB] ES 术语更新失败: {e}")

    def confirm_term(self, work_id: int, entry_key: str, entry_val: str = None):
        """
        人工确认术语译文
        
        Args:
            work_id: 项目ID
            entry_key: 词汇原文
            entry_val: 确认的译文（如果要修改）
        """
        if not self.es:
            return
            
        try:
            doc_id = f"{work_id}_{entry_key}"
            update_body = {
                "doc": {
                    "is_confirmed": True,
                    "updated_at": datetime.datetime.now().isoformat()
                }
            }
            if entry_val:
                update_body["doc"]["entry_val"] = entry_val
            
            self.es.update(index="domain_lexicon", id=doc_id, body=update_body)
            print(f"[DB] 术语确认成功: {entry_key}")
        except Exception as e:
            print(f"[DB] ES 术语确认失败: {e}")

    def search_terms(self, query: str, work_id: int = None, 
                    domain: str = None, limit: int = 10) -> List[Dict]:
        """
        检索术语
        
        Args:
            query: 搜索关键词
            work_id: 限定项目ID
            domain: 限定领域
            limit: 返回数量限制
            
        Returns:
            术语列表
        """
        if not self.es:
            return []
            
        try:
            must_clauses = [{
                "multi_match": {
                    "query": query,
                    "fields": ["entry_key^3", "entry_val", "variants.text"]
                }
            }]
            
            filter_clauses = []
            if work_id:
                filter_clauses.append({"terms": {"work_id": [work_id, 0]}})
            if domain:
                filter_clauses.append({"term": {"domain_tag": domain}})
            
            body = {
                "query": {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses if filter_clauses else None
                    }
                },
                "size": limit
            }
            
            res = self.es.search(index="domain_lexicon", body=body)
            return [hit["_source"] for hit in res['hits']['hits']]
            
        except Exception as e:
            print(f"[DB] ES 术语检索失败: {e}")
            return []

    # ==========================================
    # 统计与查询
    # ==========================================

    def get_project_stats(self, work_id: int) -> Dict:
        """获取项目统计信息"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            
            # 文档数
            cur.execute("SELECT COUNT(*) FROM source_docs WHERE work_id = %s", (work_id,))
            doc_count = cur.fetchone()[0]
            
            # 原子数及状态分布
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN pa.status_code = 0 THEN 1 END) as pending,
                    COUNT(CASE WHEN pa.status_code = 1 THEN 1 END) as drafted,
                    COUNT(CASE WHEN pa.status_code >= 2 THEN 1 END) as refined,
                    COUNT(CASE WHEN pa.status_code >= 4 THEN 1 END) as completed,
                    AVG(pa.quality_score) as avg_quality
                FROM processing_atoms pa
                JOIN source_docs sd ON pa.doc_id = sd.doc_id
                WHERE sd.work_id = %s
            """, (work_id,))
            
            row = cur.fetchone()
            return {
                "doc_count": doc_count,
                "total_atoms": row[0] or 0,
                "pending": row[1] or 0,
                "drafted": row[2] or 0,
                "refined": row[3] or 0,
                "completed": row[4] or 0,
                "avg_quality": float(row[5]) if row[5] else None
            }
        finally:
            conn.close()

    def get_translation_progress(self, doc_id: int) -> Dict:
        """获取文档翻译进度"""
        conn = self.get_pg_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN status_code > 0 THEN 1 END) as translated,
                    COUNT(CASE WHEN status_code >= 4 THEN 1 END) as completed
                FROM processing_atoms
                WHERE doc_id = %s
            """, (doc_id,))
            
            row = cur.fetchone()
            total = row[0] or 1
            return {
                "total": total,
                "translated": row[1] or 0,
                "completed": row[2] or 0,
                "progress": round((row[1] or 0) / total * 100, 2)
            }
        finally:
            conn.close()
