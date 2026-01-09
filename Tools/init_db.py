"""
AiNiee 多智能体翻译系统 - 数据库初始化脚本

数据库架构说明：
- PostgreSQL + pgvector: 关系型数据库 + 向量检索
- ElasticSearch: 术语库全文检索

核心表：
1. project_works: 翻译项目
2. source_docs: 源文档
3. processing_atoms: 处理原子（核心表，包含上下文信息）
4. agent_traces: Agent操作轨迹
"""

import psycopg2
from elasticsearch import Elasticsearch
import time

# 数据库配置
PG_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "admin",
    "password": "password",
    "dbname": "ainiee_db"
}

def init_postgres():
    """初始化 PostgreSQL 数据库架构"""
    print("[INFO] 初始化 PostgreSQL 数据库架构...")
    retries = 5
    conn = None
    
    while retries > 0:
        try:
            conn = psycopg2.connect(**PG_CONFIG)
            print("[OK] 已连接到 PostgreSQL")
            break
        except Exception as e:
            print(f"[WARN] 连接失败，{2}秒后重试... ({5-retries}/5)")
            time.sleep(2)
            retries -= 1
            
    if not conn:
        print("[ERROR] 无法连接到 PostgreSQL")
        return

    try:
        cur = conn.cursor()
        
        # 1. 启用 pgvector 插件
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # ========================================
        # 清理旧表 (按依赖关系反向删除)
        # ========================================
        print("[INFO] 清理旧表...")
        cur.execute("DROP TABLE IF EXISTS agent_traces CASCADE;")
        cur.execute("DROP TABLE IF EXISTS processing_atoms CASCADE;")
        cur.execute("DROP TABLE IF EXISTS source_docs CASCADE;")
        cur.execute("DROP TABLE IF EXISTS project_works CASCADE;")
        # 移除不需要的旧表
        cur.execute("DROP TABLE IF EXISTS doc_structures CASCADE;")
        cur.execute("DROP TABLE IF EXISTS knowledge_base CASCADE;")
        cur.execute("DROP TABLE IF EXISTS projects CASCADE;")
        cur.execute("DROP TABLE IF EXISTS files CASCADE;")
        cur.execute("DROP TABLE IF EXISTS segments CASCADE;")
        
        # ========================================
        # 层级1: 项目与文档 (Project & Document)
        # ========================================
        
        # 1. project_works (项目表)
        print("[INFO] 创建表: project_works")
        cur.execute("""
            CREATE TABLE project_works (
                work_id SERIAL PRIMARY KEY,
                work_name VARCHAR(255) NOT NULL,
                src_lang VARCHAR(50),
                tgt_lang VARCHAR(50),
                
                -- 文本主题知识 (论文1.2/3.3: 领域、主题、摘要、风格)
                topic_info JSONB DEFAULT '{}',
                
                -- 翻译指南 (论文3.3: 定制化翻译需求)
                translation_guide JSONB DEFAULT '{}',
                
                -- 工作流配置
                workflow_config JSONB DEFAULT '{}',
                
                -- 提示词模板库 (论文3.3: 提示词管理与复用)
                prompt_templates JSONB DEFAULT '{}',
                
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            COMMENT ON TABLE project_works IS '翻译项目表';
            COMMENT ON COLUMN project_works.work_name IS '项目名称';
            COMMENT ON COLUMN project_works.topic_info IS '文本主题知识(领域domain、主题topic、摘要summary、风格style)';
            COMMENT ON COLUMN project_works.translation_guide IS '翻译指南(风格要求、特殊词汇译法、文化适配)';
            COMMENT ON COLUMN project_works.workflow_config IS '工作流配置(启用的Agent、参数等)';
            COMMENT ON COLUMN project_works.prompt_templates IS '提示词模板库(各步骤的提示词模板)';
        """)
        
        # 2. source_docs (源文档表)
        print("[INFO] 创建表: source_docs")
        cur.execute("""
            CREATE TABLE source_docs (
                doc_id SERIAL PRIMARY KEY,
                work_id INTEGER REFERENCES project_works(work_id) ON DELETE CASCADE,
                file_path TEXT NOT NULL,
                file_name VARCHAR(255),
                doc_meta JSONB DEFAULT '{}',
                total_atoms INTEGER DEFAULT 0,
                status VARCHAR(50) DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            COMMENT ON TABLE source_docs IS '源文档表';
            COMMENT ON COLUMN source_docs.doc_meta IS '文档元数据(格式、大小、MD5等)';
            COMMENT ON COLUMN source_docs.total_atoms IS '文档包含的原子总数';
        """)
        
        # ========================================
        # 层级2: 处理原子 (Processing Atoms) - 核心表
        # ========================================
        
        # 3. processing_atoms (处理原子表)
        print("[INFO] 创建表: processing_atoms")
        cur.execute("""
            CREATE TABLE processing_atoms (
                atom_id SERIAL PRIMARY KEY,
                doc_id INTEGER REFERENCES source_docs(doc_id) ON DELETE CASCADE,
                
                -- 原文信息
                source_text TEXT NOT NULL,
                source_hash VARCHAR(64),
                position INTEGER DEFAULT 0,
                
                -- 翻译结果
                translated_text TEXT,
                
                -- 片段摘要 (论文4.3.1: 为每个翻译片段生成摘要)
                summary TEXT,
                
                -- 上下文信息 (核心字段，包含翻译时的完整上下文)
                context_info JSONB DEFAULT '{}',
                
                -- 向量 (用于RAG检索)
                semantic_vec vector(768),
                
                -- 状态管理
                status_code INTEGER DEFAULT 0,
                quality_score REAL,
                
                -- 质量检查信息 (论文4.3.3: 回译、警告级别)
                examination JSONB DEFAULT '{}',
                
                -- 时间戳
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            COMMENT ON TABLE processing_atoms IS '处理原子表 - 最小可翻译单元';
            COMMENT ON COLUMN processing_atoms.source_text IS '原文文本';
            COMMENT ON COLUMN processing_atoms.translated_text IS '最终译文';
            COMMENT ON COLUMN processing_atoms.summary IS '片段摘要(双语)';
            COMMENT ON COLUMN processing_atoms.context_info IS '上下文信息快照(前后文、术语、翻译记忆等)';
            COMMENT ON COLUMN processing_atoms.examination IS '质量检查信息(回译、警告级别、语义相似度)';
            COMMENT ON COLUMN processing_atoms.status_code IS '状态码: 0=未翻译, 1=已初翻, 2=已润色, 3=已审核, 4=已完成';
            COMMENT ON COLUMN processing_atoms.quality_score IS '质量评分(0-10)';
        """)
        
        # ========================================
        # 层级3: 执行轨迹 (Agent Traces) - 事件日志
        # ========================================
        
        # 4. agent_traces (Agent轨迹表)
        print("[INFO] 创建表: agent_traces")
        cur.execute("""
            CREATE TABLE agent_traces (
                trace_id SERIAL PRIMARY KEY,
                atom_id INTEGER REFERENCES processing_atoms(atom_id) ON DELETE CASCADE,
                
                -- Agent信息
                agent_role VARCHAR(50) NOT NULL,
                action_type VARCHAR(50) NOT NULL,
                
                -- 产出内容
                content TEXT,
                
                -- 质量报告 (评分、回译、问题列表等)
                quality_report JSONB DEFAULT '{}',
                
                -- 元数据
                meta_data JSONB DEFAULT '{}',
                
                -- 成本追踪
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                
                -- 状态
                is_active BOOLEAN DEFAULT FALSE,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            COMMENT ON TABLE agent_traces IS 'Agent操作轨迹表 - 事件溯源日志';
            COMMENT ON COLUMN agent_traces.agent_role IS '操作角色: Translator, Reviewer, ConsistencyChecker, Human等';
            COMMENT ON COLUMN agent_traces.action_type IS '动作类型: draft, refine, evaluate, final, human_edit等';
            COMMENT ON COLUMN agent_traces.content IS '产出内容(译文、评分、修改意见等)';
            COMMENT ON COLUMN agent_traces.quality_report IS '质量报告(回译结果、评分详情、问题列表)';
            COMMENT ON COLUMN agent_traces.is_active IS '当前是否生效(同一atom只有一个trace为true)';
        """)

        # ========================================
        # 层级4: 知识库 (Knowledge Base) - RAG支持
        # ========================================
        
        # 5. knowledge_base (知识库表)
        print("[INFO] 创建表: knowledge_base")
        cur.execute("""
            CREATE TABLE knowledge_base (
                kb_id SERIAL PRIMARY KEY,
                work_id INTEGER REFERENCES project_works(work_id) ON DELETE CASCADE,
                
                -- 知识内容
                content TEXT NOT NULL,
                kb_type VARCHAR(50) NOT NULL,
                
                -- 向量 (用于RAG检索)
                semantic_vec vector(768),
                
                -- 元数据
                meta_tags JSONB DEFAULT '{}',
                source_ref TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            COMMENT ON TABLE knowledge_base IS '知识库表 - 存储翻译记忆、风格指南、外部知识等';
            COMMENT ON COLUMN knowledge_base.kb_type IS '知识类型: tm(翻译记忆), glossary(术语), style_guide(风格指南), external(外部知识)';
            COMMENT ON COLUMN knowledge_base.meta_tags IS '标签元数据(领域、来源、权重等)';
        """)

        # ========================================
        # 创建索引
        # ========================================
        print("[INFO] 创建索引...")
        
        # 业务索引
        cur.execute("CREATE INDEX idx_atoms_doc ON processing_atoms(doc_id);")
        cur.execute("CREATE INDEX idx_atoms_status ON processing_atoms(status_code);")
        cur.execute("CREATE INDEX idx_atoms_position ON processing_atoms(position);")
        cur.execute("CREATE INDEX idx_traces_atom ON agent_traces(atom_id);")
        cur.execute("CREATE INDEX idx_traces_active ON agent_traces(atom_id, is_active) WHERE is_active = TRUE;")
        cur.execute("CREATE INDEX idx_docs_work ON source_docs(work_id);")
        cur.execute("CREATE INDEX idx_kb_work ON knowledge_base(work_id);")
        cur.execute("CREATE INDEX idx_kb_type ON knowledge_base(kb_type);")
        
        # 向量索引 (仅在有数据后才有效)
        try:
            cur.execute("""
                CREATE INDEX atoms_vec_idx 
                ON processing_atoms USING ivfflat (semantic_vec vector_cosine_ops) WITH (lists = 100);
            """)
            cur.execute("""
                CREATE INDEX kb_vec_idx 
                ON knowledge_base USING ivfflat (semantic_vec vector_cosine_ops) WITH (lists = 100);
            """)
        except Exception as e:
            print(f"[INFO] 向量索引创建跳过 (表为空时正常): {e}")

        conn.commit()
        cur.close()
        conn.close()
        print("[OK] PostgreSQL 数据库架构初始化成功")
        
    except Exception as e:
        print(f"[ERROR] PostgreSQL 初始化失败: {e}")
        if conn:
            conn.rollback()
            conn.close()


def init_elasticsearch():
    """初始化 ElasticSearch 术语库索引"""
    print("[INFO] 初始化 ElasticSearch 术语库索引...")
    try:
        es = Elasticsearch("http://localhost:9200")
        
        if not es.ping():
            print("[ERROR] 无法连接到 ElasticSearch")
            return
        
        print("[OK] 已连接到 ElasticSearch")
        
        # domain_lexicon (术语库索引)
        # 论文4.4.3: 词汇表格式设计
        lexicon_mapping = {
            "mappings": {
                "properties": {
                    # 术语基本信息
                    "entry_key": {"type": "keyword"},  # 词汇原文
                    "entry_val": {"type": "text", "analyzer": "standard"},  # 最终译文
                    
                    # 项目关联 (复合主键: work_id + entry_key)
                    "work_id": {"type": "integer"},
                    
                    # 词汇类型 (论文4.4.3: 普通词汇、命名实体、术语、概念词、关键词、缩略词、谚语)
                    "word_type": {"type": "keyword"},
                    
                    # 相关翻译片段ID列表 (论文4.4.3: segment_ids)
                    "atom_ids": {"type": "integer"},
                    
                    # 术语详情
                    "domain_tag": {"type": "keyword"},  # 领域标签
                    "variants": {"type": "nested", "properties": {
                        "text": {"type": "text"},
                        "lang": {"type": "keyword"}
                    }},
                    "example_sentences": {"type": "text"},  # 词汇使用示例
                    
                    # 候选译法列表 (论文4.4.3: translations nested)
                    "translations": {"type": "nested", "properties": {
                        "translation": {"type": "text"},     # 译文
                        "source": {"type": "keyword"},       # 来源(LLM/词典/搜索引擎/机器翻译)
                        "confidence": {"type": "float"},     # 置信度
                        "rank": {"type": "integer"},         # 排名
                        "rationale": {"type": "text"}        # 排序理由
                    }},
                    
                    # 来源追踪
                    "source_ref": {"type": "keyword"},  # 对应 atom_id
                    "confidence": {"type": "float"},    # 最终置信度
                    
                    # Agent备注
                    "agent_notes": {"type": "text"},
                    
                    # 审校状态
                    "is_confirmed": {"type": "boolean"},  # 是否人工确认
                    
                    # 时间
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        # 删除旧索引并重建
        if es.indices.exists(index="domain_lexicon"):
            es.indices.delete(index="domain_lexicon")
            print("[INFO] 已删除旧索引 'domain_lexicon'")
            
        es.indices.create(index="domain_lexicon", body=lexicon_mapping)
        print("[OK] ElasticSearch 索引 'domain_lexicon' 创建成功")
        
    except Exception as e:
        print(f"[ERROR] ElasticSearch 初始化失败: {e}")


def show_schema():
    """显示当前数据库架构"""
    print("\n" + "="*60)
    print("AiNiee 多智能体翻译系统 - 数据库架构")
    print("="*60)
    print("""
┌─────────────────────────────────────────────────────────────┐
│                      project_works                          │
│  work_id, work_name, src_lang, tgt_lang                     │
│  topic_info      : 文本主题知识 (领域/主题/摘要/风格)       │
│  translation_guide: 翻译指南 (风格/固定译法/文化适配)       │
│  prompt_templates : 提示词模板库                            │
└────────────────────────┬────────────────────────────────────┘
                         │ 1:N
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌─────────────────┐ ┌─────────────┐ ┌─────────────────────────┐
│   source_docs   │ │knowledge_base│ │   domain_lexicon (ES)   │
│  (源文档)       │ │ (知识库/TM)  │ │  PK: work_id+entry_key  │
└────────┬────────┘ └─────────────┘ │  word_type  : 词汇类型  │
         │ 1:N                       │  translations: 候选译法 │
         ▼                           └─────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                  processing_atoms [核心表]                   │
│  source_text     : 原文                                     │
│  translated_text : 译文                                     │
│  summary         : 片段摘要                                 │
│  context_info    : 上下文信息 (前后文/术语/翻译记忆)        │
│  examination     : 质量检查信息 (回译/警告级别)             │
│  semantic_vec    : 语义向量 (768维)                         │
│  status_code     : 状态 (0→4)                               │
│  quality_score   : 质量评分                                 │
└────────────────────────┬────────────────────────────────────┘
                         │ 1:N
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    agent_traces [事件日志]                   │
│  agent_role   : Translator/Reviewer/ConsistencyChecker/Human│
│  action_type  : draft/refine/evaluate/final/human_edit      │
│  content      : 产出内容 (译文/评分/意见)                   │
│  quality_report: 质量报告 (回译/评分详情)                   │
│  is_active    : 当前生效版本                                │
└─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    init_postgres()
    init_elasticsearch()
    show_schema()
    print("\n[完成] 数据库初始化完成！")
