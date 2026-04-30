import streamlit as st
from rag_core import *
import time


st.set_page_config(page_title="AI Study Companion", layout="wide")

def type_writer(text, speed=0.005):
    placeholder = st.empty()
    words = text.split(" ")
    displayed = ""
    for w in words:
        displayed += w + " "
        placeholder.markdown(displayed)
        time.sleep(speed)

def highlight_text(text, query):
    words = query.split()
    for w in words:
        if len(w) > 3:
            text = text.replace(w, f"**{w}**")
    return text

def extract_confidence(answer):
    if "CONFIDENCE:" in answer.upper():
        return answer.upper().split("CONFIDENCE:")[-1].strip().split("\n")[0]
    return "UNKNOWN"

def confidence_label(conf):
    conf = str(conf).upper()

    if "HIGH" in conf:
        return "High"
    elif "MEDIUM" in conf:
        return "Medium"
    elif "LOW" in conf:
        return "Low"
    else:
        return "Unknown"
    
def parse_mcq(answer):
    if "QUESTION:" not in answer:
        return None
    
    try:
        q_part = answer.split("QUESTION:")[1]
        parts = q_part.split("\n")
        
        question = parts[0].strip()
        options = [p.strip() for p in parts if p.strip().startswith(("A)", "B)", "C)"))]
        
        return question, options
    
    except:
        return None

debug_mode = st.sidebar.checkbox(" Debug Mode ")

if "initialized" not in st.session_state:
    chunks, metadata, index, chat_history, memory_store, memory_index = initialize_rag(
        folder_path=r"C:\Users\Insan\OneDrive\Documents\AIENGINEER",
        kb_name="physics"
    )

    st.session_state.chunks = chunks
    st.session_state.metadata = metadata
    st.session_state.index = index
    st.session_state.chat_history = chat_history
    st.session_state.memory_store = memory_store
    st.session_state.memory_index = memory_index

    st.session_state.messages = []
    st.session_state.initialized = True

st.sidebar.title("Knowledge Base")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Process Files"):
        with st.spinner("Processing documents..."):
            
            all_names = []

            for file in uploaded_files:
                kb_name = file.name.replace(".pdf", "")
                process_uploaded_file(file, kb_name)
                all_names.append(kb_name)

            chunks, metadata, index, chat_history, memory_store, memory_index = initialize_rag(
                folder_path=None,
                kb_name=all_names 
            )

            st.session_state.chunks = chunks
            st.session_state.metadata = metadata
            st.session_state.index = index
            st.session_state.chat_history = []
            st.session_state.memory_store = []
            st.session_state.memory_index = None
            st.session_state.messages = []

            st.success(f"{len(all_names)} documents loaded successfully!")

if st.sidebar.button(" Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.memory_store = []
    st.session_state.memory_index = None
    st.rerun()

st.title("AI Study Companion")
st.caption("Learn. Question. Understand.")

mode = st.selectbox("Mode", ["strict", "tutor", "learn"], index=1)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        if msg["role"] == "assistant" and "evidence" in msg:
            with st.expander("Sources"):
                for i, (idx, ev, src) in enumerate(msg["evidence"]):
                        st.markdown(f"Source [{i+1}]")
                        st.caption(f" {src}")
                        st.markdown(f" > {ev}")
                        st.divider()

query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        status = st.empty()
        progress = st.progress(0)
        
        status.markdown("Processing Question...")
        progress.progress(10)
        time.sleep(0.3)
        
        status.markdown("Searching Knowledge Base...")
        retrieved = retrieve(
            query,
            st.session_state.chunks,
            st.session_state.metadata,
            st.session_state.index
        )
        progress.progress(30)
        
        status.markdown("Extracting Relevant Evidence...")
        evidence = extract_evidence(
            query,
            retrieved
        )
        
        status.markdown(" Checking Past Conversation...")
        memory = retrieve_memory(
            query,
            st.session_state.memory_store,
            st.session_state.memory_index
        )
        progress.progress(70)
        
        status.markdown("Generating Answer...")
        answer, confidence = generate_answer(
            query,
            evidence,
            st.session_state.chat_history,
            memory,
            mode
        )


        conf_label = confidence
        
        st.session_state.memory_index = store_memory(
            query,
            answer,
            st.session_state.memory_store,
            st.session_state.memory_index,
            time_step=1
        )

        progress.progress(100)
        status.markdown("Done!")

    response = f"""
{answer}

---

**Confidence:** {conf_label}
"""

    type_writer(response)

    mcq = parse_mcq(answer)
    if mcq:
        question, options = mcq
        st.markdown(" Quick Check")
        st.write(question)
        
        cols = st.columns(len(options))

        for i, opt in enumerate(options):
            if cols[i].button(opt, key = f"opt_{i}_{len(st.session_state.messages)}"):
                st.success(f"You Selected: {opt}")

    with st.expander(" Sources"):
        for i, (idx, ev, src) in enumerate(evidence):
            highlighted = highlight_text(ev, query)
            
            st. markdown(f" Source [{i+1}]")
            st.caption(f" {src}")
            st.markdown(f"> {highlighted}")
            st.divider()
    
    if st.button("Regenerate Answer"):
        st.session_state.messages.pop()
        st.rerun()
        
    
    if debug_mode:
        st.write("Retrieved:", len(retrieved))
        st.write("Evidence:", len(evidence))
        st.write("Memory:", len(memory))
        
            
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "evidence": evidence
    })