# Fieldnotes: Local RAG Workshop

A classroom-ready **Chat with Documents** app. Upload PDF or DOCX files, inspect extracted text chunks, and ask questions answered by a local Ollama model. Retrieval and generation are separate and visible: the app first finds relevant chunks with local embeddings and cosine similarity, then sends only those excerpts and your question to the selected local model.

No API keys or paid services are used. Uploaded files and the index stay in the Streamlit session; the app does not save uploaded documents to disk.

## Requirements

- Windows, macOS, or Linux; Python 3.11 recommended (Python 3.12 is also supported by current dependencies).
- Ollama installed and running for answer generation. The app itself starts and explains missing Ollama/model states without crashing.
- A few GB of free disk space for Python dependencies and the embedding model. Ollama model size depends on the model variant.
- CPU is supported; no GPU is required.

## Setup

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that PowerShell window, then activate again. Alternatively run `.venv\Scripts\python.exe -m streamlit run app.py` without activating.

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run app.py
```

Streamlit opens the app at `http://localhost:8501`.

## Install Ollama and a model

Install Ollama from [ollama.com/download](https://ollama.com/download) and start its local service. Ollama normally runs in the background after installation. In a terminal, install one model (Qwen is preferred; Llama is optional fallback):

```powershell
ollama pull qwen2.5:3b
```

Or, if Qwen is not suitable for the machine:

```powershell
ollama pull llama3.2:3b
```

You only need one. Model downloads are explicit terminal commands: the application never pulls or downloads an Ollama model. On each run, it checks Ollama's installed model tags and selects the first Qwen variant if present; otherwise it selects an installed Llama variant. The settings sidebar displays the chosen tag and also permits an explicit choice among installed Qwen/Llama models.

The embedding model is separate from the generation model. It is `sentence-transformers/all-MiniLM-L6-v2`, runs locally on CPU, and is not an Ollama model. On first index, the app uses local cached files only by default. Enable **Allow first-time embedding model download** in the sidebar to explicitly download it from Hugging Face; after that it is cached locally. No API key is needed.

## Run the workshop

1. Start Ollama and confirm the app header reports **Connected** and shows a Qwen or Llama tag.
2. Upload a text-based syllabus PDF or a DOCX. Image-only PDFs are identified clearly; OCR is not bundled.
3. Adjust chunk size/overlap if desired, then select **Build document index**. Review document, character, page, and chunk counts.
4. Ask a question such as **“What are the eligibility requirements mentioned in this document?”**
5. Expand **Retrieved sources**. Compare the chunk IDs, source/page metadata, text, and cosine scores with the answer.
6. Ask something absent from the document to demonstrate the similarity threshold and grounded no-match response.
7. Change Top-K or the threshold and ask again to observe how retrieval changes.

Example questions:

- What are the eligibility requirements?
- What documents must applicants submit?
- When is the application deadline?
- What courses are required in the first semester?
- Does the document mention a scholarship amount? (Use when absent to demonstrate a no-match.)

## RAG in one picture

```text
PDF / DOCX
    ↓
Extract text (PDF pages retained) → Clean text → Split into overlapping chunks
    ↓
Local Sentence Transformers embeddings (one normalized vector per chunk)
    ↓
Question embedding → cosine similarity → threshold + Top-K retrieval
    ↓
Visible source excerpts + question → local Ollama Qwen (preferred) / Llama (fallback)
    ↓
Grounded answer + inspectable retrieved sources
```

An embedding represents text meaning as a vector. Cosine similarity compares vector direction:

`cosine_similarity(a, b) = (a · b) / (||a|| × ||b||)`

A score near 1 means the vectors point in a similar direction; a score near 0 means little directional similarity. The app normalizes vectors and computes this formula directly with NumPy. RAG does not give an LLM automatic access to a PDF: the application must retrieve useful document text and include it in the prompt first.

## Project map

- `app.py`: Streamlit workflow, session-state index, progress, settings, and chat.
- `core/document_loader.py`: PyMuPDF PDF extraction with page numbers and python-docx paragraph/table extraction.
- `core/text_cleaner.py`: whitespace/control-character cleanup that retains punctuation and paragraphs.
- `core/chunker.py`: configurable overlapping word chunks with source, page, and chunk ID metadata.
- `core/embeddings.py`: local CPU Sentence Transformers loader and normalized embeddings.
- `core/retriever.py`: NumPy cosine scores, threshold, Top-K ranking, and duplicate suppression.
- `core/model_selector.py`, `core/ollama_client.py`: runtime model-family selection and local Ollama HTTP calls.
- `core/rag_pipeline.py`: builds separated system/user messages, creates source-labelled context, and skips generation when retrieval has no qualifying result.
- `ui/`: restrained Streamlit styling and retrieved-source rendering.
- `tests/`: extraction, cleaning, chunking, embedding contract, similarity, retrieval, no-match, and model-selection tests.

## Security and boundaries

Uploaded excerpts are untrusted data. A system instruction tells the local model to ignore commands in document excerpts and answer from evidence only. This is a prompt-injection defense-in-depth measure, not a guarantee; do not use the workshop with sensitive documents or treat generated answers as authoritative. The app does not upload document contents to a hosted API. It sends the selected question and retrieved excerpts to the Ollama service on the local machine.

The initial index is in-memory and tied to the browser session. It is discarded when that Streamlit session ends. Files are limited to 50 MB each; text extraction is not OCR, and no vector database is required.

## Tests

```powershell
python -m pytest -q
```

PDF/DOCX tests generate documents in memory. Embedding and Ollama behavior use small deterministic test doubles, so tests need neither an installed Ollama model nor a Hugging Face download.

## Troubleshooting

- **Ollama not connected:** install Ollama, start the service, then refresh. The default local address is `http://localhost:11434`.
- **No supported model:** run `ollama pull qwen2.5:3b` or `ollama pull llama3.2:3b`, then refresh. Check installed tags with `ollama list`.
- **Embedding model unavailable offline:** connect once and enable its explicit first-time download option, or pre-cache `sentence-transformers/all-MiniLM-L6-v2` in the active Python environment.
- **PDF has no selectable text:** it may be scanned/image-only. Convert it to a text-based PDF or OCR it before upload; this app does not perform OCR.
- **DOCX/PDF extraction error:** check the file opens normally, is not encrypted/corrupt, and has the matching extension.
- **No relevant chunks:** lower the minimum similarity threshold or ask a more specific question using the document's vocabulary. The LLM is intentionally not called when no chunk clears the threshold.
- **Slow answers:** use a smaller Ollama model, reduce Top-K, and try shorter documents. Embedding and inference run locally.
- **Settings or uploads changed:** rebuild the index; the app disables chat until the index matches the current files and chunk settings.
