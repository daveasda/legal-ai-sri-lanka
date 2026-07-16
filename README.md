# ⚖️ Legal AI Sri Lanka

An AI-powered legal assistant that lets users ask natural-language questions about Sri Lankan law and get grounded, source-backed answers — built on a Retrieval-Augmented Generation (RAG) pipeline.

## Overview

Legal information in Sri Lanka is scattered across dense statutes and PDFs that are hard for the public to search. This project indexes Sri Lankan legal documents into a local vector store and uses an LLM (via LangChain) to answer questions conversationally, grounded in the actual source text rather than the model's general knowledge.

## Features

- 💬 Conversational UI with persistent chat history
- 📚 Retrieval-Augmented Generation over Sri Lankan legal documents (PDF/TXT)
- 🔍 Local vector store — no legal data leaves the machine
- 🧠 Powered by LangChain for document ingestion, retrieval, and orchestration

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| LLM Orchestration | LangChain |
| Retrieval | RAG + local vector store |
| Interface | Web app (`app.py`) |

## Project Structure

```
legal-ai-sri-lanka/
├── app.py              # Main application entry point
├── docs/               # Sri Lankan legal documents (PDF/TXT)
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add documents**

Place your Sri Lankan legal documents (PDF/TXT) in the `docs/` folder.

**3. Run the app**
```bash
python app.py
```

Then open your browser at `http://localhost:7860`.

## Future Improvements

- Add citation highlighting so answers link back to the exact clause/section
- Expand the legal document corpus (case law, amendments)
- Add multi-language support (Sinhala/Tamil)
