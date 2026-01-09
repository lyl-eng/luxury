# Multi-Agent Translation System

A multi-agent translation system based on LLM (Large Language Models), implementing human-machine collaborative translation workflow.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Translation Flow                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ Preprocessing│───▶│  Terminology │───▶│ Translation  │          │
│  │    Agent     │    │    Agent     │    │    Agent     │          │
│  └──────────────┘    └──────────────┘    └──────────────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   Data Slot Layer                            │   │
│  │  ┌─────────────────────┐  ┌─────────────────────────────┐   │   │
│  │  │  PostgreSQL+pgvector│  │     ElasticSearch           │   │   │
│  │  │  • project_works    │  │  • domain_lexicon           │   │   │
│  │  │  • source_docs      │  │    (Terminology Index)      │   │   │
│  │  │  • processing_atoms │  └─────────────────────────────┘   │   │
│  │  │  • agent_traces     │                                    │   │
│  │  │  • knowledge_base   │                                    │   │
│  │  └─────────────────────┘                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
luxury/
├── AiNiee.py                          # Main entry point
├── requirements.txt                   # Python dependencies
├── docker-compose.yml                 # Database services
├── paper.txt                          # Reference paper
├── DATABASE_SCHEMA.md                 # Database design docs
│
├── Base/                              # Framework foundation
│   ├── Base.py                        # Base class with config
│   ├── EventManager.py                # Event pub/sub system
│   └── PluginManager.py               # Plugin management
│
├── Tools/
│   └── init_db.py                     # Database initialization
│
└── ModuleFolders/
    ├── MultiAgent/                    # 🤖 Multi-Agent Core
    │   ├── WorkflowManager.py         # Workflow orchestration
    │   ├── PreprocessingAgent.py      # Text preprocessing
    │   ├── TerminologyEntityAgent.py  # Terminology extraction
    │   ├── TranslationRefinementAgent.py  # Translation & QA
    │   ├── PlanningAgent.py           # Task planning
    │   ├── HumanCollaborationNode.py  # HITL collaboration
    │   ├── GriptapeTools.py           # Griptape tools
    │   └── GriptapeAgents.py          # Agent definitions
    │
    ├── Cache/                         # 💾 Data Slot Layer
    │   ├── DatabaseManager.py         # PostgreSQL & ES operations
    │   ├── CacheManager.py            # Cache orchestration
    │   ├── BaseCache.py               # Base cache class
    │   ├── CacheProject.py            # Project-level cache
    │   ├── CacheFile.py               # File-level cache
    │   └── CacheItem.py               # Translation unit
    │
    ├── LLMRequester/                  # 🌐 LLM API Layer
    │   ├── LLMRequester.py            # Unified interface
    │   ├── OpenaiRequester.py         # OpenAI/compatible
    │   ├── AnthropicRequester.py      # Claude
    │   ├── GoogleRequester.py         # Gemini
    │   ├── DashscopeRequester.py      # Qwen/Tongyi
    │   └── ...
    │
    ├── TaskExecutor/                  # ⚙️ Task Execution
    │   ├── MultiAgentTaskExecutor.py  # Multi-agent executor
    │   ├── TaskExecutor.py            # Standard executor
    │   ├── TranslatorTask.py          # Translation task
    │   └── PolisherTask.py            # Polish task
    │
    ├── FileReader/                    # 📖 File Input (23 formats)
    │   ├── FileReader.py              # Unified reader
    │   ├── DocxReader.py, EpubReader.py
    │   ├── BabeldocPdfReader.py
    │   ├── SrtReader.py, AssReader.py
    │   └── ...
    │
    ├── FileOutputer/                  # 📝 File Output (23 formats)
    │   ├── FileOutputer.py            # Unified writer
    │   └── ...
    │
    ├── PromptBuilder/                 # 📋 Prompt Construction
    ├── ResponseChecker/               # ✅ Response Validation
    ├── ResponseExtractor/             # 📤 Response Parsing
    ├── NERProcessor/                  # 🏷️ Named Entity Recognition
    ├── RequestLimiter/                # ⏱️ Rate Limiting
    ├── TextProcessor/                 # 📝 Text Processing
    ├── FileAccessor/                  # 📁 File Access Utils
    ├── FileConverter/                 # 🔄 Format Conversion
    ├── SimpleExecutor/                # ⚡ Simple Executor
    └── TaskConfig/                    # ⚙️ Task Configuration
```

## Core Components

### Multi-Agent Modules

| Module | Description |
|--------|-------------|
| `WorkflowManager` | Orchestrates agent workflow with Griptape |
| `PreprocessingAgent` | Text structure analysis, domain recognition |
| `TerminologyEntityAgent` | NER, terminology extraction & translation |
| `TranslationRefinementAgent` | Translation, quality assessment, refinement |
| `PlanningAgent` | Strategy selection, task planning |
| `HumanCollaborationNode` | Human-in-the-loop intervention |

### Data Slot Layer (Database)

| Component | Technology | Purpose |
|-----------|------------|---------|
| `DatabaseManager` | PostgreSQL + pgvector | Relational data, vector storage |
| `domain_lexicon` | ElasticSearch | Full-text terminology search |
| `init_db.py` | Docker | Database initialization |

### Supported File Formats

**Input/Output:** PDF, DOCX, EPUB, TXT, MD, SRT, ASS, VTT, LRC, PO, JSON (i18next), Ren'Py, MTool, TPP, VNT, Paratranz, and more.

## Database Schema

See [DATABASE_SCHEMA.md](DATABASE_SCHEMA.md) for detailed design.

**PostgreSQL Tables:**
- `project_works` - Projects with topic info, translation guides, prompt templates
- `source_docs` - Source documents metadata
- `processing_atoms` - Core translation units with context, summary, examination
- `agent_traces` - Event sourcing for all agent operations
- `knowledge_base` - Translation memory, external knowledge (RAG)

**ElasticSearch Index:**
- `domain_lexicon` - Terminology with composite key (work_id + entry_key), word types, candidate translations, human confirmation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Database Services

```bash
docker-compose up -d
```

### 3. Initialize Database Schema

```bash
python Tools/init_db.py
```

### 4. Verify Installation

```bash
# Check PostgreSQL tables
docker exec ainiee_postgres psql -U admin -d ainiee_db -c "\dt"

# Check ElasticSearch index
curl http://localhost:9200/domain_lexicon/_mapping?pretty
```

## Key Features

Based on the reference paper (`paper.txt`):

- **Event Sourcing Architecture** - Full traceability of all agent operations
- **Vector-Native Storage** - pgvector for semantic search and RAG
- **Human-in-the-Loop (HITL)** - Seamless human-AI collaboration
- **Multi-Agent Coordination** - Specialized agents for different translation tasks
- **Context-Aware Translation** - Terminology consistency, translation memory
- **Quality Assessment** - Back-translation, semantic similarity checks

## License

MIT License
