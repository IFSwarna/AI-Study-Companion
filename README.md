# AI Study Companion (RAG + Tutor System)

An intelligent Retrieval-Augmented Generation (RAG) system designed as a **personal study companion**.  
Supports multi-document reasoning, memory, and multiple learning modes (strict, tutor, learn).

---

## Features

### Multi-Document RAG
- Upload multiple PDFs
- Semantic search using FAISS
- Context-aware retrieval across documents

### Chat Memory
- Short-term conversational memory
- Long-term memory with decay
- Context-aware follow-up answers

### Learning Modes
- Strict Mode → grounded, citation-only answers  
- Tutor Mode → simplified explanations + guidance  
- Learn Mode → step-by-step teaching with guiding questions  

### Real Citation System
- Extracts exact evidence from documents
- Displays sources with document names
- Prevents hallucination

### Confidence Scoring
- Evaluates answer vs evidence overlap
- Labels: High / Medium / Low confidence

### Performance Optimizations
- Cached embeddings
- Query embedding cache
- Precomputed knowledge base (.pkl)

### UI (Streamlit)
- Chat-style interface
- Typing animation
- Progress feedback (searching, reasoning, etc.)
- Source expansion panel

---

## Architecture

- User Input → Query Expansion → Retrieval (FAISS) → Evidence Extraction → LLM Reasoning → Answer

## Project Structure
AIENGINEER/
│
├── app.py # Streamlit UI
├── rag_core.py # Core RAG logic
├── rag_data/ # Stored embeddings (.pkl)
├── requirements.txt
└── README.md


## Installation

### 1. Clone repository
git clone https://github.com/YOUR_USERNAME/ai-study-companion.git
cd ai-study-companion

### 2. Install dependencies
pip install -r requirements.txt

### 3. Install & run Ollama
Make sure you have:
- :contentReference[oaicite:0]{index=0}
- Model: `mistral:7b`
- Embedding model: `nomic-embed-text`

## Run the App
streamlit run app.py


---

## How to Use

1. Upload one or multiple PDFs  
2. Click **Process Files**  
3. Ask questions in chat  
4. Choose mode:
   - strict
   - tutor
   - learn  
5. Expand sources to verify answers  

---

## Example Use Cases

- Physics study assistant
- Research paper reader
- Concept tutor
- Multi-document Q&A system

---

## Limitations

- Dependent on local LLM performance
- Large PDFs may increase processing time
- Memory system is heuristic-based (not perfect)

---

## Future Improvements

- Voice interaction
- Better chunk ranking
- Memory summarization
- Multi-modal support (images, diagrams)
- Deployment (cloud / API)

---

## Author

Built as part of an AI Engineering roadmap project.

---

## Why this project matters

This is not just a chatbot — it demonstrates:

- RAG system design
- Information retrieval (FAISS)
- Prompt engineering
- Memory systems
- UX for AI products
