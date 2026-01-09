# Multi-Agent Translation System

A multi-agent translation system based on LLM (Large Language Models), implementing human-machine collaborative translation workflow.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Translation Flow              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │ Preprocessing│───▶│  Terminology │───▶│ Translation  │  │
│  │    Agent     │    │    Agent     │    │    Agent     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PostgreSQL + pgvector                   │   │
│  │  • project_works      • processing_atoms            │   │
│  │  • source_docs        • agent_traces                │   │
│  │  • knowledge_base     (Event Sourcing)              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              ElasticSearch                           │   │
│  │  • domain_lexicon (Terminology Index)               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### Multi-Agent Modules (`ModuleFolders/MultiAgent/`)

| File | Description |
|------|-------------|
| `WorkflowManager.py` | Agent workflow orchestration |
| `PreprocessingAgent.py` | Text preprocessing & domain recognition |
| `TerminologyEntityAgent.py` | Terminology extraction & entity recognition |
| `TranslationRefinementAgent.py` | Translation, refinement & quality assessment |
| `PlanningAgent.py` | Task planning & strategy selection |
| `HumanCollaborationNode.py` | Human-in-the-loop collaboration |
| `GriptapeTools.py` | Griptape framework tool wrappers |
| `GriptapeAgents.py` | Agent definitions |

### Database Layer (`ModuleFolders/Cache/`)

| File | Description |
|------|-------------|
| `DatabaseManager.py` | PostgreSQL & ElasticSearch operations |
| `CacheItem.py` | Translation unit data structure |
| `CacheProject.py` | Project-level cache management |
| `CacheFile.py` | File-level cache management |

### Tools (`Tools/`)

| File | Description |
|------|-------------|
| `init_db.py` | Database initialization script |

## Database Architecture

See [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for detailed database design.

**Key Tables:**
- `project_works` - Translation projects with topic info & guides
- `source_docs` - Source documents
- `processing_atoms` - Minimal translation units (core table)
- `agent_traces` - Agent operation logs (event sourcing)
- `knowledge_base` - Translation memory & external knowledge

**ElasticSearch Index:**
- `domain_lexicon` - Terminology with composite key (work_id + entry_key)

## Quick Start

### 1. Start Database Services

```bash
docker-compose up -d
```

### 2. Initialize Database Schema

```bash
python Tools/init_db.py
```

### 3. Verify Installation

```bash
# Check PostgreSQL
docker exec ainiee_postgres psql -U admin -d ainiee_db -c "\dt"

# Check ElasticSearch
curl http://localhost:9200/domain_lexicon/_mapping?pretty
```

## Reference

This project is inspired by the paper in `paper.txt`, implementing:
- Event Sourcing architecture for full traceability
- Vector-native storage with pgvector for RAG
- Human-in-the-loop (HITL) collaboration workflow
- Multi-agent coordination for translation quality

## License

MIT License

