#  RAG-based Question Answering System on Swiggy Annual Report (2023–24)

##  Objective

The objective of this project is to build a Retrieval-Augmented Generation (RAG) based AI system that answers user questions strictly based on the Swiggy Annual Report 2023–24.

The system retrieves relevant context from the document and generates accurate, context-grounded responses without hallucination.

---

##  Dataset

**Document Used:** Swiggy Annual Report 2023–2024  
**Format:** PDF  
**Source:** Official Mail  

Annual Report Reference: Swiggy Annual Report 2023–24

(Note: As required in the assignment, the source document is publicly available.)

---

## System Architecture

The application follows the RAG (Retrieval-Augmented Generation) architecture:

1. **Document Loading**
   - Load Swiggy Annual Report PDF

2. **Text Chunking**
   - Split document into meaningful chunks
   - Preserve metadata

3. **Embedding Generation**
   - Generate vector embeddings using an embedding model

4. **Vector Database**
   - Store embeddings in FAISS / ChromaDB
   - Perform semantic similarity search

5. **Retrieval**
   - Retrieve top relevant chunks based on user query

6. **LLM Response Generation**
   - Pass retrieved context to LLM
   - Generate answer strictly from retrieved content

7. **User Interface**
   - Built using Streamlit
   - Displays:
     - Final Answer
     - Supporting Context (optional)

---

##  Tech Stack

- Python
- Streamlit
- LangChain
- FAISS / Chroma
- OpenAI / LLM API
- PyPDF

---

##  Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/swiggy-rag-ai-dashboard.git
cd swiggy-rag-ai-dashboard
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application
```bash
streamlit run app.py
```

---

##  Example Questions

- What was Swiggy’s net loss in FY24?
- How many cities does Swiggy operate in?
- Who is the CFO?
- What is the revenue for FY24?
- Explain Quick Commerce performance.

---

##  Important Design Constraint

The system is designed to:
- Answer strictly based on the Swiggy Annual Report
- Avoid hallucinations
- Use retrieved document context only

---

##  Key Features

- Semantic Search  
- Context-aware responses  
- Clean UI  
- Scalable RAG architecture  
- Supports natural language queries  

---

##  Functional Requirements Covered

- PDF Processing
- Chunking & Preprocessing
- Embedding Generation
- Vector Database Storage
- Semantic Retrieval
- Context-grounded LLM Response
- Interactive UI

(All requirements implemented as per assignment guidelines.)

---

## Future Improvements

- Add page number citations
- Add similarity score display
- Deploy on Streamlit Cloud
- Add downloadable response export

---


