# RAG Portfolio Intelligence Engine

Turn any portfolio into a queryable AI assistant.

An end-to-end Retrieval-Augmented Generation (RAG) system that transforms raw portfolio data into a low-latency, context-aware assistant capable of answering structured questions about an engineer’s work, experience, and projects.

---

![Asad's AI Assistant](Asads%20AI%20Assistant.gif)

---

## Why This Exists

Traditional portfolios are static and require manual reading.

This system converts a portfolio into an interactive interface where users can ask questions such as:

* What IoT systems has this engineer built?
* What backend architectures has he designed?
* What differentiates his experience?

The system responds instantly using grounded, context-aware retrieval.

---

## Core Capabilities

* End-to-end automated RAG pipeline
* Sub-second response time using in-memory retrieval
* Hot data refresh without server restart
* Structured knowledge base generation (PDF)
* Modular, extensible architecture
* Production-ready FastAPI backend

---

## System Architecture

```text
Portfolio URLs
   ↓
ContentCrawler
   ↓
AIEngine (LLM synthesis)
   ↓
PDFGenerator
   ↓
DocumentChunker
   ↓
EmbedEngine
   ↓
SearchEngine (in-memory)
   ↓
ChatEngine
   ↓
User Query Response
```

---

## How It Works

1. Crawls portfolio sources
2. Converts raw data into structured documentation
3. Generates embeddings from semantic chunks
4. Stores vectors in memory
5. Retrieves relevant context and injects into LLM prompts

Result: fast, grounded, context-aware responses.

---

## Core Components

| Component       | Responsibility                                   |
| --------------- | ------------------------------------------------ |
| ContentCrawler  | Extracts and cleans portfolio data               |
| AIEngine        | Converts raw data into structured knowledge      |
| PDFGenerator    | Builds a portable knowledge base                 |
| DocumentChunker | Splits content into semantic sections            |
| EmbedEngine     | Generates embeddings                             |
| SearchEngine    | Performs fast vector similarity search           |
| ChatEngine      | Produces final responses using retrieved context |

---

## Performance Design

The system uses normalized vector similarity with NumPy for fast retrieval:

```python
similarities = np.dot(self.embeddings_matrix, query_vector)
```

* No external database dependency
* No iterative search loops
* Minimal latency

---

## Data Synchronization

The pipeline supports live updates:

* Crawl latest data
* Regenerate structured knowledge
* Recompute embeddings
* Update in-memory index

All without restarting the server.

---

## API

### Query

```http
GET /ask?query=Tell me about Asad's IoT experience
```

### Refresh Pipeline

```http
POST /pipeline/update
```

---

## Running the System

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run pipeline

```bash
python runner.py full
```

### Start server

```bash
uvicorn app:app --reload
```

---

## Use Cases

* AI-powered developer portfolio
* Personal knowledge assistant
* Internal documentation query system
* Resume-to-AI transformation

---

## Differentiation

Unlike most RAG demos, this system is:

* Fully automated end-to-end
* Designed for real-world usage
* Optimized for low-latency inference
* Built with a clear, practical use case

---

## Author

Asad Ahmed
Senior Full Stack AI & IoT Engineer

Expertise in distributed systems, AI pipelines, embedded systems, and scalable backend architectures.

---

## Roadmap

* UI dashboard
* Multi-user support
* Optional vector database integration
* Streaming responses

---

## Support

If this project is useful, consider starring the repository or sharing it within your network.
