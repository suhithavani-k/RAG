from __future__ import annotations

import hashlib

import streamlit as st

from core.chunker import Chunk, create_chunks
from core.document_loader import DocumentLoadError, load_document
from core.embeddings import DEFAULT_EMBEDDING_MODEL, encode_texts, load_embedding_model
from core.ollama_client import OllamaClient, OllamaError
from core.model_selector import select_model
from core.rag_pipeline import answer_question
from ui.components import render_retrieved_sources
from ui.styles import apply_styles

st.set_page_config(page_title="Fieldnotes · Local RAG Workshop", page_icon="⌁", layout="wide")
apply_styles()


@st.cache_resource(show_spinner=False)
def cached_embedding_model(allow_download: bool):
    return load_embedding_model(allow_download=allow_download)


def initialize_state() -> None:
    st.session_state.setdefault("index", None)
    st.session_state.setdefault("history", [])


initialize_state()
client = OllamaClient()
try:
    model_names = client.list_models()
    ollama_error = None
    model_choice = select_model(model_names) if model_names else None
except OllamaError as error:
    model_names = []
    ollama_error = str(error)
    model_choice = None

with st.sidebar:
    st.markdown("<div class='eyebrow'>Workshop controls</div>", unsafe_allow_html=True)
    st.subheader("Index settings")
    chunk_size = st.slider("Chunk size · words", min_value=80, max_value=350, value=180, step=10)
    chunk_overlap = st.slider("Chunk overlap · words", min_value=0, max_value=min(100, chunk_size - 1), value=min(35, chunk_size - 1), step=5)
    st.caption("Changing chunk settings requires rebuilding the index.")
    st.divider()
    st.subheader("Retrieval settings")
    top_k = st.slider("Top-K", min_value=1, max_value=8, value=4)
    use_threshold = st.checkbox("Use minimum similarity", value=True)
    min_similarity = st.slider("Minimum cosine similarity", min_value=0.0, max_value=0.9, value=0.20, step=0.05, disabled=not use_threshold)
    temperature = st.slider("Answer temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    st.divider()
    st.subheader("Embedding model")
    st.code(DEFAULT_EMBEDDING_MODEL, language=None)
    allow_embedding_download = st.checkbox("Allow first-time embedding model download", value=False)
    st.caption("Runs locally on CPU. Model files are downloaded only after you enable this option.")
    st.divider()
    if model_choice and model_choice.ready:
        supported_models = [name for name in model_names if "qwen" in name.lower() or "llama" in name.lower()]
        selected_default = model_choice.name
        selected_index = supported_models.index(selected_default) if selected_default in supported_models else 0
        selected_model = st.selectbox("Ollama model", supported_models, index=selected_index)
        if selected_model != selected_default:
            st.caption(f"Auto-preferred: {selected_default}")
    else:
        selected_model = None
        st.caption("Install a Qwen model, or a Llama model as fallback, using the README setup steps.")
selected_family = select_model([selected_model]).family if selected_model else None

st.markdown("<div class='eyebrow'>Hands-on document intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-title'>Ask what your documents know.</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='hero-copy'>A local retrieval-augmented generation workshop. Inspect what was retrieved, "
    "then see how a local model turns those excerpts into a grounded answer.</div>",
    unsafe_allow_html=True,
)
model_status = f"<strong>{selected_model}</strong> · {selected_family}" if selected_model and selected_family else "<strong>No Qwen/Llama model</strong>"
ollama_status = "<strong>Connected</strong>" if ollama_error is None else "<strong>Not connected</strong>"
st.markdown(
    f"<div class='status-strip'><div class='status-pill'>Ollama · {ollama_status}</div>"
    f"<div class='status-pill'>Generation · {model_status}</div>"
    "<div class='status-pill'>Embeddings · Local CPU</div></div>",
    unsafe_allow_html=True,
)
if ollama_error:
    st.warning(f"{ollama_error} Ollama normally runs in the background; install it and start the local service.")
elif not selected_model:
    st.info("Ollama is running, but no Qwen or Llama model is installed. Install one with the commands in the README.")

upload_col, overview_col = st.columns([1.05, 0.95], gap="large")
with upload_col:
    st.markdown("<div class='section-label'>01 / Add source material</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Files are processed locally and are not sent to a hosted service.",
    )
    payloads: list[tuple[str, bytes]] = []
    if uploaded_files:
        for uploaded in uploaded_files:
            data = uploaded.getvalue()
            if len(data) > 50 * 1024 * 1024:
                st.error(f"{uploaded.name} is larger than the 50 MB per-file limit.")
                continue
            payloads.append((uploaded.name, data))
            st.caption(f"{uploaded.name} · {len(data) / 1024:.1f} KB")
    digest = hashlib.sha256()
    for filename, data in payloads:
        digest.update(filename.encode("utf-8"))
        digest.update(data)
    digest.update(f"{chunk_size}:{chunk_overlap}:{DEFAULT_EMBEDDING_MODEL}".encode("utf-8"))
    current_fingerprint = digest.hexdigest()
    index_is_current = bool(st.session_state.index and st.session_state.index["fingerprint"] == current_fingerprint)
    index_button = st.button("Build document index", type="primary", disabled=not payloads, use_container_width=True)

    if index_button:
        progress = st.progress(0, text="Preparing documents…")
        status = st.status("Building your local knowledge base", expanded=True)
        try:
            status.write("Extracting text and preserving source pages…")
            all_chunks: list[Chunk] = []
            characters = 0
            page_count = 0
            for filename, data in payloads:
                pages = load_document(filename, data)
                characters += sum(len(page.text) for page in pages)
                page_count += sum(page.page_number is not None for page in pages)
                all_chunks.extend(create_chunks(pages, filename, chunk_size, chunk_overlap))
            if not all_chunks:
                raise DocumentLoadError("No text could be indexed from the selected files.")
            progress.progress(0.35, text="Loading local embedding model…")
            status.write("Generating normalized embeddings on CPU…")
            encoder = cached_embedding_model(allow_embedding_download)
            vectors = encode_texts(encoder, [chunk.text for chunk in all_chunks])
            if vectors.shape[0] != len(all_chunks):
                raise ValueError("Embedding count did not match the number of document chunks.")
            st.session_state.index = {
                "fingerprint": current_fingerprint,
                "chunks": all_chunks,
                "vectors": vectors,
                "document_count": len(payloads),
                "characters": characters,
                "pages": page_count,
            }
            st.session_state.history = []
            progress.progress(1.0, text="Index ready")
            status.update(label="Index ready", state="complete", expanded=False)
            st.rerun()
        except DocumentLoadError as error:
            status.update(label="Document could not be indexed", state="error", expanded=True)
            st.error(str(error))
        except ModuleNotFoundError as error:
            status.update(label="Embedding dependencies are missing", state="error", expanded=True)
            package = error.name or "a required package"
            st.error(f"The embedding environment is missing `{package}`. Install the project dependencies, then restart Streamlit.")
            st.code(
                ".venv\\Scripts\\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
                ".venv\\Scripts\\python.exe -m pip install -r requirements.txt",
                language="powershell",
            )
        except Exception as error:
            status.update(label="Indexing stopped", state="error", expanded=True)
            st.error("The local embedding model could not be loaded or used. Check the troubleshooting section in the README.")
            st.caption(f"Technical detail: {type(error).__name__}")

with overview_col:
    st.markdown("<div class='section-label'>02 / Knowledge base</div>", unsafe_allow_html=True)
    index = st.session_state.index
    if index:
        st.markdown("**Indexed**" if index_is_current else "**Index needs rebuilding**")
        metrics = st.columns(3)
        metrics[0].metric("Documents", index["document_count"])
        metrics[1].metric("Chunks", len(index["chunks"]))
        metrics[2].metric("Characters", f"{index['characters']:,}")
        if index["pages"]:
            st.caption(f"Selectable-text PDF pages extracted: {index['pages']}")
        st.caption("Embedding state: normalized vectors ready · retrieval: cosine similarity · storage: session memory")
        if not index_is_current:
            st.info("The uploaded files or chunk settings changed. Rebuild the index before asking a question.")
    else:
        st.markdown("**Not indexed**")
        st.caption("Upload documents, then build an index to start a conversation.")

st.divider()
st.markdown("<div class='section-label'>03 / Ask your documents</div>", unsafe_allow_html=True)
if st.button("Clear conversation", disabled=not st.session_state.history):
    st.session_state.history = []
    st.rerun()

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant":
            render_retrieved_sources(entry.get("sources", []), "history")

question = st.chat_input("Ask a question about the indexed documents", disabled=not (index_is_current and selected_model))
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if ollama_error:
            answer = "Ollama is not connected. Start the local Ollama service and try again."
            sources = []
            st.warning(answer)
        else:
            try:
                encoder = cached_embedding_model(allow_embedding_download)
                query_vector = encode_texts(encoder, [question])[0]
                with st.spinner(f"Retrieving relevant excerpts, then asking {selected_model}…"):
                    answer, sources = answer_question(
                        question=question,
                        chunks=index["chunks"],
                        chunk_vectors=index["vectors"],
                        query_vector=query_vector,
                        client=client,
                        model=selected_model,
                        top_k=top_k,
                        min_similarity=min_similarity if use_threshold else None,
                        temperature=temperature,
                    )
                st.markdown(answer)
                render_retrieved_sources(sources, "latest")
            except OllamaError as error:
                answer = str(error)
                sources = []
                st.error(answer)
            except Exception as error:
                answer = "The question could not be processed. Check that the index and local embedding model are available."
                sources = []
                st.error(answer)
                st.caption(f"Technical detail: {type(error).__name__}")
        st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})

with st.expander("How this answer is built", expanded=False):
    st.markdown("**Document → Extract → Chunk → Embed → Compare → Retrieve → Context → LLM → Answer**")
    st.write("Embeddings represent meaning as vectors. Cosine similarity compares their directions. Retrieval selects the closest excerpts; only those excerpts and your question are passed to Ollama. The model does not automatically know the uploaded documents. Retrieved text is untrusted reference material, not an instruction to the model.")
