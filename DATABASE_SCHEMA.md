# AiNiee 多智能体翻译系统 - 数据库架构说明

本文档详细描述了 AiNiee 系统采用的"事件溯源 (Event Sourcing)"数据库架构。该架构将**项目管理**、**处理原子**与**执行轨迹**分离，实现了全流程的可追溯性和高可扩展性。

## 1. 核心关系型数据库 (PostgreSQL + PgVector)

### 1.1 项目与文档层 (Project & Document Level)

#### `project_works` (项目表)
存储翻译项目的元数据、主题知识和翻译指南。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `work_id` | SERIAL (PK) | 项目唯一标识 ID |
| `work_name` | VARCHAR(255) | 项目名称（如 "zh2en_20231001"） |
| `src_lang` | VARCHAR(50) | 源语言 |
| `tgt_lang` | VARCHAR(50) | 目标语言 |
| `topic_info` | JSONB | **文本主题知识**（领域、主题、摘要、风格） |
| `translation_guide` | JSONB | **翻译指南**（风格要求、特殊词汇译法、文化适配） |
| `workflow_config` | JSONB | 工作流配置快照（包含启用的 Agent、参数等） |
| `prompt_templates` | JSONB | **提示词模板库**（各步骤的提示词模板） |
| `status` | VARCHAR(50) | 项目状态 ('active', 'archived', 'deleted') |
| `created_at` | TIMESTAMP | 创建时间 |

#### `topic_info` 字段结构 (JSONB)

```json
{
  "domain": "文学/科技/法律/医学...",
  "topic": "主题关键词",
  "summary": "文本摘要",
  "style": "正式/非正式/文学化..."
}
```

#### `translation_guide` 字段结构 (JSONB)

```json
{
  "style_requirements": "翻译风格要求",
  "fixed_translations": {"term1": "固定译法1"},
  "cultural_adaptation": "文化适配说明",
  "target_audience": "目标受众"
}
```

#### `source_docs` (文档表)
存储项目下的源文件信息。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `doc_id` | SERIAL (PK) | 文档唯一标识 ID |
| `work_id` | INTEGER (FK) | 关联的项目 ID |
| `file_path` | TEXT | 文件路径 |
| `file_name` | VARCHAR(255) | 文件名 |
| `doc_meta` | JSONB | 文档元数据（格式、大小、MD5等） |
| `total_atoms` | INTEGER | 文档包含的原子总数 |
| `status` | VARCHAR(50) | 文档状态 ('pending', 'processed') |
| `created_at` | TIMESTAMP | 创建时间 |

### 1.2 处理原子层 (Processing Atoms) - **核心表**

#### `processing_atoms` (处理原子表)
存储最小的可翻译单元（通常是一句话或一个片段）。这是所有 Agent 操作的目标对象。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `atom_id` | SERIAL (PK) | 原子 ID |
| `doc_id` | INTEGER (FK) | 关联的文档 ID |
| `source_text` | TEXT | **原文文本** |
| `source_hash` | VARCHAR(64) | 原文哈希（用于去重/缓存） |
| `position` | INTEGER | 在文档中的位置索引 |
| `translated_text` | TEXT | **最终译文** |
| `summary` | TEXT | **片段摘要**（双语） |
| `context_info` | JSONB | **上下文信息快照**（核心字段） |
| `semantic_vec` | VECTOR(768) | 原文的语义向量（用于 RAG 检索） |
| `status_code` | INTEGER | 状态码 (0:未翻译, 1:已初翻, 2:已润色, 3:已审核, 4:已完成) |
| `quality_score` | REAL | 质量评分 (0-10) |
| `examination` | JSONB | **质量检查信息**（回译结果、警告级别、语义相似度） |
| `created_at` | TIMESTAMP | 创建时间 |
| `updated_at` | TIMESTAMP | 更新时间 |

#### `examination` 字段结构 (JSONB) - 质量检查信息

```json
{
  "back_translation": "回译结果",
  "warning_level": "high/medium/low",
  "semantic_similarity": 0.95,
  "issues": ["问题1", "问题2"],
  "algorithm": "backtranslation"
}
```

#### `context_info` 字段结构 (JSONB)

上下文信息包含翻译时的完整上下文，用于提高翻译一致性和连贯性：

```json
{
  "prev_source": "前一原子的原文",
  "prev_translated": "前一原子的译文",
  "next_source": "后一原子的原文",
  "terminology": [
    {"term": "术语原文", "translation": "术语译文", "domain": "领域"}
  ],
  "memory_refs": [
    {"source": "相似原文", "translation": "参考译文", "score": 0.95}
  ],
  "similar_atoms": [
    {"source": "历史原文", "translated": "历史译文"}
  ]
}
```

### 1.3 知识库层 (Knowledge Base) - **RAG 支持**

#### `knowledge_base` (知识库表)
存储翻译记忆 (TM)、风格指南、外部知识等，支持向量检索。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `kb_id` | SERIAL (PK) | 知识条目 ID |
| `work_id` | INTEGER (FK) | 关联的项目 ID |
| `content` | TEXT | **知识内容** |
| `kb_type` | VARCHAR(50) | 类型 ('tm'=翻译记忆, 'glossary'=术语, 'style_guide'=风格指南, 'external'=外部知识) |
| `semantic_vec` | VECTOR(768) | 语义向量（用于 RAG 检索） |
| `meta_tags` | JSONB | 标签元数据（领域、来源、权重等） |
| `source_ref` | TEXT | 来源引用 |
| `created_at` | TIMESTAMP | 创建时间 |

#### `meta_tags` 字段结构 (JSONB) - 翻译记忆示例

```json
{
  "source": "原文文本",
  "translation": "译文文本",
  "domain": "领域标签",
  "quality": 0.95
}
```

### 1.4 执行轨迹层 (Agent Traces) - **事件日志**

#### `agent_traces` (轨迹表)
记录所有 Agent 或人工对原子进行的操作。这是一张**只增不改**的日志表，通过 `is_active` 标记当前生效的版本。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `trace_id` | SERIAL (PK) | 轨迹 ID |
| `atom_id` | INTEGER (FK) | 关联的原子 ID |
| `agent_role` | VARCHAR(50) | 操作角色 ('Translator', 'Reviewer', 'ConsistencyChecker', 'Human', 'QualityAssessor') |
| `action_type` | VARCHAR(50) | 动作类型 ('draft', 'refine', 'evaluate', 'final', 'human_edit') |
| `content` | TEXT | **产出内容**（译文、评分、修改意见） |
| `quality_report` | JSONB | **质量报告**（包含评分、回译文、问题列表） |
| `meta_data` | JSONB | 其他元数据（使用的策略、模型、耗时等） |
| `input_tokens` | INTEGER | 输入 Token 数 |
| `output_tokens` | INTEGER | 输出 Token 数 |
| `is_active` | BOOLEAN | **当前是否生效**（同一 atom 只有一个 trace 为 true） |
| `created_at` | TIMESTAMP | 操作时间 |

#### `quality_report` 字段结构 (JSONB)

```json
{
  "score": 8.5,
  "back_translation": "回译结果",
  "issues": ["问题1", "问题2"],
  "status": "pass" | "needs_refinement"
}
```

---

## 2. 全文检索引擎 (ElasticSearch)

### 2.1 术语库索引 (`domain_lexicon`)

用于存储和检索项目术语/词汇表。**复合主键：`work_id + entry_key`**，确保同一术语在不同项目中可以有不同译法。

| 字段 (JSON) | 类型 | 说明 |
| :--- | :--- | :--- |
| `_id` | - | **文档ID = `{work_id}_{entry_key}`**（复合主键） |
| `entry_key` | keyword | 词汇原文 |
| `entry_val` | text | **最终译文** |
| `work_id` | integer | **项目 ID** (关联 `project_works.work_id`) |
| `word_type` | keyword | **词汇类型** (普通词汇/命名实体/术语/概念词/关键词/缩略词/谚语) |
| `atom_ids` | integer[] | **相关翻译片段 ID 列表** |
| `domain_tag` | keyword | 领域标签 |
| `example_sentences` | text | **词汇使用示例** |
| `translations` | nested | **候选译法列表** (含来源、置信度、排名、理由) |
| `variants` | nested | 词汇变体列表 |
| `source_ref` | keyword | 来源引用（对应 atom_id） |
| `confidence` | float | 最终置信度 |
| `agent_notes` | text | Agent 的备注或解释 |
| `is_confirmed` | boolean | **是否人工确认** |
| `updated_at` | date | 更新时间 |

#### `translations` 嵌套结构 - 候选译法

```json
{
  "translations": [
    {
      "translation": "候选译文",
      "source": "LLM/dictionary/search/mt",
      "confidence": 0.95,
      "rank": 1,
      "rationale": "排序理由"
    }
  ]
}
```

---

## 3. 数据流转示意

```
┌─────────────────────────────────────────────────────────────┐
│                      project_works                          │
│  work_id, work_name, src_lang, tgt_lang                     │
│  topic_info      : 文本主题知识 (领域/主题/摘要/风格)       │
│  translation_guide: 翻译指南 (风格要求/固定译法/文化适配)   │
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
         ▼                           │  is_confirmed: 人工确认 │
┌─────────────────────────────────┐ └─────────────────────────┘
│        processing_atoms ⭐核心表                             │
│  source_text     : 原文                                     │
│  translated_text : 译文                                     │
│  summary         : 片段摘要 ⭐新增                          │
│  context_info    : 上下文信息 (JSONB)                       │
│    ├─ prev_source     : 前一原子原文                        │
│    ├─ prev_translated : 前一原子译文                        │
│    ├─ next_source     : 后一原子原文                        │
│    ├─ terminology     : 相关术语列表                        │
│    ├─ memory_refs     : 翻译记忆引用                        │
│    └─ similar_atoms   : 相似历史翻译                        │
│  examination     : 质量检查信息 (回译/警告) ⭐新增          │
│  semantic_vec    : 语义向量 (768维)                         │
│  status_code     : 状态 (0→4)                               │
│  quality_score   : 质量评分                                 │
└────────────────────────┬────────────────────────────────────┘
                         │ 1:N
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    agent_traces ⭐事件日志                   │
│  agent_role   : Translator/Reviewer/ConsistencyChecker/Human│
│  action_type  : draft/refine/evaluate/final/human_edit      │
│  content      : 产出内容 (译文/评分/意见)                   │
│  quality_report: 质量报告 (回译/评分详情)                   │
│  is_active    : 当前生效版本                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 多智能体翻译流程数据写入

### 4.1 预处理阶段 (PreprocessingAgent)

| 操作 | 写入表 | 数据内容 |
| :--- | :--- | :--- |
| 创建项目 | `project_works` | work_name, src_lang, tgt_lang, workflow_config |
| 创建文档 | `source_docs` | work_id, file_path, file_name, doc_meta |
| 创建原子 | `processing_atoms` | doc_id, source_text, position, context_info (初始上下文) |

### 4.2 术语识别阶段 (TerminologyAgent)

| 操作 | 写入位置 | 数据内容 |
| :--- | :--- | :--- |
| 识别术语 | ElasticSearch `domain_lexicon` | entry_key, entry_val, domain_tag, example_sentences |
| 更新上下文 | `processing_atoms.context_info` | terminology 列表 |

### 4.3 翻译阶段 (TranslationAgent)

| 操作 | 写入表 | 数据内容 |
| :--- | :--- | :--- |
| 初翻 | `agent_traces` | agent_role="Translator", action_type="draft", content=译文 |
|  | `processing_atoms` | translated_text, status_code=1 |
| 质量评估 | `agent_traces` | agent_role="QualityAssessor", action_type="evaluate", quality_report |
| 润色 | `agent_traces` | agent_role="Translator", action_type="refine", content=润色后译文 |
|  | `processing_atoms` | translated_text, status_code=2 |
| 一致性检查 | `agent_traces` | agent_role="ConsistencyChecker", action_type="final", content=最终译文 |
|  | `processing_atoms` | translated_text, status_code=4, quality_score |

### 4.4 人工审校阶段

| 操作 | 写入表 | 数据内容 |
| :--- | :--- | :--- |
| 人工修改 | `agent_traces` | agent_role="Human", action_type="human_edit", content=修改后译文 |
|  | `processing_atoms` | translated_text, status_code=3 |

---

## 5. 架构优势

1.  **完整可追溯 (Traceability)**:
    *   通过 `agent_traces` 表，可以重现任何一句话的翻译历史：初翻 → 评分 → 回译验证 → 人工修改 → 一致性检查。

2.  **结构化上下文 (Context Awareness)**:
    *   `processing_atoms.context_info` 记录了翻译发生时的"现场"，包含前后文、术语、翻译记忆等，对于分析翻译质量至关重要。

3.  **向量增强 (Vector-Native)**:
    *   `processing_atoms.semantic_vec` 原生支持向量存储，为 RAG（检索增强生成）和语义一致性检查打下基础。

4.  **人机协作 (HITL Ready)**:
    *   `agent_traces` 明确区分了 `agent_role`，人工修改 (`Human`) 只是轨迹中的一种特殊 Event，系统可以自然地融合人工和 AI 的工作。

5.  **简洁高效**:
    *   相比论文中的多表设计（原始段落、翻译片段、匹配等），本架构将核心数据集中在 `processing_atoms` 表，上下文信息通过 JSONB 字段存储，减少了表关联查询，提高了效率。

---

## 6. 初始化数据库

```bash
# 进入项目目录
cd AiNiee-main

# 运行初始化脚本
python Tools/init_db.py
```

## 7. 查看数据

```bash
# 查看项目
docker exec ainiee_postgres psql -U admin -d ainiee_db -c "SELECT * FROM project_works;"

# 查看原子
docker exec ainiee_postgres psql -U admin -d ainiee_db -c "SELECT atom_id, LEFT(source_text, 50), status_code FROM processing_atoms LIMIT 20;"

# 查看轨迹
docker exec ainiee_postgres psql -U admin -d ainiee_db -c "SELECT trace_id, atom_id, agent_role, action_type, LEFT(content, 50) FROM agent_traces ORDER BY trace_id DESC LIMIT 20;"

# 查看术语库
curl http://localhost:9200/domain_lexicon/_search?pretty
```

