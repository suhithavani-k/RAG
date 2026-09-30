import streamlit as st

from core.retriever import RetrievedChunk


def render_retrieved_sources(results: list[RetrievedChunk], key_prefix: str) -> None:
    if not results:
        return
    with st.expander(f"Retrieved sources · {len(results)} chunks", expanded=False):
        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            page = f" · page {chunk.page_number}" if chunk.page_number is not None else ""
            st.markdown(
                f"**{rank}. {chunk.source}{page}** · `{chunk.chunk_id}` · similarity **{result.score:.3f}**"
            )
            st.caption(chunk.text)
            if rank < len(results):
                st.divider()
