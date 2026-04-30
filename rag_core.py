import ollama
import numpy as np
import pdfplumber
import faiss
import os
import pickle

query_cache = {}
MAX_MEMORY = 50

def process_uploaded_file(uploaded_file, kb_name):
    import tempfile
    
    os.makedirs("rag_data", exist_ok=True)
    file_name = f"rag_data/{kb_name}.pkl"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        text = load_file(temp_path)
        chunks, metadata = chunk_text([text], [uploaded_file.name])

        embeddings = embed_chunks(chunks)
        embedding_matrix = np.array(embeddings).astype("float32")

        with open(file_name, "wb") as f:
            pickle.dump((embedding_matrix, chunks, metadata), f)

        return file_name

    except Exception as e:
        print(f"[ERROR] Upload processing failed: {e}")
        return None
    
def load_file(file_path):
    text = ""
    
    with pdfplumber.open(file_path) as pdf:
        
        for page in pdf.pages:
            content = page.extract_text()
            
            if content:
                text += content + "\n"
                
    return text

def load_pdf(folder_path):
    all_texts = []
    doc_names = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(folder_path, file)
            text = load_file(full_path)
            
            all_texts.append(text)
            doc_names.append(file)
            
    return all_texts, doc_names

def chunk_text(texts, doc_names, chunk_size = 800, overlap = 150):
    chunks = []
    metadata = []
    
    for text, name in zip(texts,doc_names):
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if end < len(text):
                last_period = chunk.rfind(".")
                last_space = chunk.rfind(" ")
                
                if last_period > 0:
                    chunk = chunk[:last_period +1]
                elif last_space > 0: 
                    chunk = chunk[:last_space]
             
            chunk = chunk.strip()
            
            if len(chunk) >= 120:
                chunk = " ".join(chunk.split())
                chunks.append(chunk)
                metadata.append(name)
                    
            start += chunk_size - overlap
            
    return chunks, metadata

def clean_text(chunks):
    cleaned = []
    
    for chunk in chunks:
        chunk = chunk.strip()
        
        if len(chunk) < 120:
            continue
        
        chunk = " ".join(chunk.split())
        
        cleaned.append(chunk)
    return cleaned

def get_query_embedding(query):
    global query_cache
    
    if query in query_cache:
        return query_cache[query]
    
    try:
        emb = ollama.embed(
            model = "nomic-embed-text",
            input = query
        )["embeddings"][0]
        
        query_cache[query] = emb
        return emb
    
    except Exception as e:
        print(f"[ERROR] Query embedding failed: {e}")
        return None
    
def embed_chunks(chunks):
    try:
        response = ollama.embed(
            model = "nomic-embed-text",
            input = chunks
        )
        return response["embeddings"]
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return []
    
def expand_query(query):
    prompt = f"""
    You are an AI that Improve the search queries.
    
    Generate 3 alternative versions of the following question.
    Keep the meaning the same, but rephrase the words.
    
    Question: {query}
    
    Output only the 3 rephrased questions as a list.
    """
    try:
        response = ollama.chat(
            model = "mistral:7b",
            messages =[{"role": "user","content": prompt}]
        )

        text = response["message"]["content"]

        queries = [q.strip("-").strip() for q in text.split("\n") if q.strip()]

        return list(set([query] + queries))
    
    except Exception as e:
        print(f"[ERROR] Query Expansion failed: {e}")
        return [query]
    
def retrieve(query, chunks, metadata, index, k=5):
    all_queries = expand_query(query)
    results = set()
    
    try:
        for q in all_queries:
            query_embedding = get_query_embedding(q)
            if query_embedding is None:
                continue

            vec = np.array([query_embedding]).astype("float32")

            _, indices = index.search(vec, k)

            for i in indices[0]:
                results.add((chunks[i], metadata[i]))

        return list(results)[:k]
    
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return []
    
def extract_evidence(query, retrieved_items):
    evidence_list = []
    
    for i, (chunk, source) in enumerate(retrieved_items):
        prompt = f"""
        Extract ONLY the most relevant sentence(s) from the text below that help answer the question.
        
        Rules:
        - Copy exact pbrases (do NOT rewrite)
        - Keep it short (1 - 3 sentences MAX)
        - if nothing is relevant, return "NONE"
        
        Text:
        {chunk}
        
        Question:
        {query}
        
        Relevant evidence:
        """
        
        response = ollama.chat(
            model= "mistral:7b",
            messages = [{"role":"user","content":prompt}]
        )
        
        ev = response["message"]["content"].strip()
        
        if ev != "NONE":
            evidence_list.append((i+1, ev, source))
            
    return evidence_list

def is_good_memory(answer):
    if "I DON'T KNOW" in answer:
        return False
    if len(answer.strip()) < 50:
        return False
    return True

def store_memory(query, answer, memory_store, memory_index, time_step):
    if not is_good_memory(answer):
        return memory_index
    
    text = f"User: {query}\nAssistant: {answer}"
    
    emb = ollama.embed(
        model="nomic-embed-text",
        input=text
    )["embeddings"][0]

    memory_store.append({
        "text": text,
        "embedding": emb,
        "score": 1.0,
        "time": time_step
    })

    vec = np.array([emb]).astype("float32")
    
    if memory_index is None:
        memory_index = faiss.IndexFlatL2(vec.shape[1])
        
    memory_index.add(vec)
    
    if len(memory_store) > MAX_MEMORY:
        memory_store.pop(0)

    return memory_index

def apply_memory_decay(memory_store, current_time, decay_rate=0.05):
    for mem in memory_store:
        if not isinstance(mem, dict):
            continue 
        
        age = current_time - mem["time"]
        mem["score"] *= np.exp(-decay_rate * age)
        
def retrieve_memory(query, memory_store, memory_index, k=3):
    if memory_index is None or len(memory_store) == 0:
        return []
    
    try:
        emb = get_query_embedding(query)
        if emb is None:
            return []

        vec = np.array([emb]).astype("float32")

        _, indices = memory_index.search(vec, k * 2)

        results = []
        
        for i in indices[0]:
            if i < len(memory_store):
                mem = memory_store[i]
                
                if not isinstance(mem, dict):
                    continue
                
                if mem["score"] > 0.2:
                    results.append(mem)
        
        results = sorted(results, key=lambda x: x["score"], reverse= True)
        
        return [m["text"] for m in results[:k]]
    
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return []
    
def generate_answer(query, evidence_list, chat_history, memory_chunks, mode = "strict"):
    evidence_text = ""
    
    for idx, ev, source in evidence_list:
        evidence_text += f"[{idx}] ({source}) {ev}\n\n"
    
    memory_text = "\n\n".join(memory_chunks)
    
    strict_prompt = f"""
    You are a strict AI assistant.

    You MUST follow these rules:
    - Answer ONLY using the provided evidence
    - Do NOT use outside knowledge
    - If the answer is directly stated, explain it clearly
    - If the answer is NOT directly stated:
        - Try to infer using ONLY the given evidence
        - Explain your reasoning step-by-step
    - If it is still not possible to answer, reply EXACTLY with: "I DON'T KNOW"
    - Cite the source number like [1], [2] when using information
    
    Evidence:{
        evidence_text
        }
    Memory:{
        memory_text
        }
    Answer:
    """
    
    tutor_prompt = f"""
    You are a AI specialized in tutoring.
    
    You MUST follow these rules:
    - Guide the student
    - Make the subject intruiging
    - Use outside knowledge only when the subject correlate
    - Use simple words
    - Use evidence when needed
    - If answer doesn't correlate:
        - Try to explain the reason you didn't reach the answer
    - If you cannot answer the student, reply EXACTLY with: "I DON'T KNOW"
    
    Evidence:{
        evidence_text
        }
    Memory:{
        memory_text
        }
    Answer:
    """
    
    guided_prompt = f"""
    You are the best interactive tutor.
    You guide your student without giving them the exact answer.
    
    You MUST follow these rules:
    - Teach the student with a step-by-step guide
    - Breakdown concepts into small parts that is easy to understand
    - After explaining, give the student ONE easy guiding question that will help them gain a deeper understanding
    - Help the student to think, not GIVE them ANSWER
    - Use ONLY Provided evidence
    - If answer doesn't correlate:
        - Try to explain the reason you didn't reach the answer
    - If you cannot answer the student, reply EXACTLY with: "I DON'T KNOW"
    - If the user give the correct answer to the guiding question:
        - Always try to give the student compliment whenever they answer your guiding questions correctly.
        - Move on to explain the subject, DO NOT keep on asking the same question.
    -If the user give the incorrect answer to the guiding question:
        - Give them consolidation on their misunderstanding, and give them a hint or clue to reach the desired answer.
        - Then ask another guiding question based on the given hints.
        
    Evidence:{
        evidence_text
        }
    Memory:{
        memory_text
        }
    Answer:
    """
    
    if mode == "tutor":
        system_prompt = tutor_prompt
    elif mode == "learn":
        system_prompt = guided_prompt
    else:
        system_prompt = strict_prompt
    
    messages =[{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role":"user","content":query})
    
    try:
        response = ollama.chat(
            model = "mistral:7b",
            messages = messages
        )
        
        answer = response["message"]["content"]
    
        score = verify_answer(answer, evidence_list)
        confidence = get_confidence_label(score)

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": answer})
    
    except Exception as e:
        print(f"[ERROR] LLM Failed: {e}")
        return []
        
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})
        
    if len(chat_history) > 10:
        chat_history.pop(0)
        chat_history.pop(0)
            
    return answer, confidence

def verify_answer(answer, evidence_list):
    evidence_text = " ".join([ev for _, ev, _ in evidence_list]).lower()
    answer_text = answer.lower()
    
    overlap = 0
    for word in answer_text.split():
        if word in evidence_text:
            overlap += 1
            
    ratio = overlap / max(len(answer_text.split()), 1)
    
    if ratio > 0.5:
        return "HIGH"
    elif ratio > 0.25:
        return "MEDIUM"  
    else: 
        return "LOW"
    
def get_confidence_label(score):
    if score == "HIGH":
        return "Extremely Confident"
    elif score == "MEDIUM": 
        return "Moderately Confident"
    else:
        return "Low Confidence"
    
def initialize_rag(folder_path=None, kb_name="default"):

    if isinstance(kb_name, list):
        all_chunks = []
        all_metadata = []
        all_embeddings = []

        for name in kb_name:
            try:
                file_path = f"rag_data/{name}.pkl"

                if not os.path.exists(file_path):
                    print(f"[WARNING] Missing file: {file_path}")
                    continue

                with open(file_path, "rb") as f:
                    embedding_matrix, chunks, metadata = pickle.load(f)

                    all_chunks.extend(chunks)
                    all_metadata.extend(metadata)
                    all_embeddings.append(embedding_matrix)

            except Exception as e:
                print(f"[ERROR] Failed loading {name}: {e}")
    
        if len(all_embeddings) == 0:
            raise ValueError("No embeddings were loaded. Check file names or processing step.")

        if len(all_embeddings) == 1:
            embedding_matrix = all_embeddings[0]
        else:
            embedding_matrix = np.vstack(all_embeddings)

    else:
        file_path = f"rag_data/{kb_name}.pkl"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embedding file not found: {file_path}")

        with open(file_path, "rb") as f:
            embedding_matrix, all_chunks, all_metadata = pickle.load(f)

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    return all_chunks, all_metadata, index, [], [], None
    
def run_rag(query, mode, state):
    state["time_step"] += 1
    
    apply_memory_decay(state["memory_store"], state["time_step"])
    
    retrieved_items = retrieve(query, state["chunks"], state["metadata"], state["index"])
    
    evidence_list = extract_evidence(query, retrieved_items)
    
    memory_chunks = retrieve_memory(query, state["memory_store"], state["memory_index"])
    
    answer = generate_answer(
        query,
        evidence_list,
        state["chat_history"],
        memory_chunks,
        mode
    )
    
    state["memory_index"] = store_memory(
        query,
        answer,
        state["memory_store"],
        state["memory_index"],
        state["time_step"]
    )
    
    return answer, evidence_list