
###**Day 2: Data Engineering & RAG Preparation**

**Goal:** Understand how to prepare raw text so a model can actually read and understand it. "Garbage in, Garbage out" applies doubly to Generative AI.

**1. The Workflow: S3 to Vector Store**
The standard AWS pattern for RAG (Retrieval Augmented Generation) is:
`S3 (Source Docs)` -> `Bedrock Knowledge Base (Ingestion)` -> `Chunking` -> `Embedding Model` -> `Vector Store`.

**2. Data Sources & Metadata**

* **S3:** The primary storage. You drop your PDFs, TXTs, or MD files here.
* **Metadata Files (`.metadata.json`):** This is critical for the exam. If you have a file `contract.pdf`, you can upload `contract.pdf.metadata.json` alongside it.
* *Why?* It allows you to add custom tags (e.g., `"department": "legal"`). Later, when querying, you can **filter** specifically for "legal" documents before the model even generates an answer.



**3. Chunking Strategies (High Priority Exam Topic)**
Models have a limit on how much text they can process at once. We must break large documents into "chunks."

* **Fixed-Size Chunking:**
* *What:* "Cut every 200 words."
* *Pros:* Fast, cheap, easy.
* *Cons:* It might cut a sentence in half, destroying the meaning.


* **Semantic Chunking:**
* *What:* Uses an embedding model to see where the *topic* changes (e.g., between paragraphs or sections) and cuts there.
* *Pros:* Keeps related information together. Much higher quality retrieval.
* *Cons:* More computationally expensive (slower ingestion).


* **Hierarchical Chunking:**
* *What:* Creates "Parent" chunks (large summaries) and "Child" chunks (detailed text). Good for answering "high-level" questions that also need specific details.


* **Overlap:**
* You almost always want **10-20% overlap** between chunks. This ensures that if a sentence is cut, the context is preserved in the next chunk.



---

###**Day 2 Scenario Quiz**You are building a RAG application for a medical research firm. The source documents are complex research papers where the **meaning** of a paragraph relies heavily on the sentences immediately preceding it. If you cut a paragraph in the middle, the medical context could be lost, leading to dangerous hallucinations.

**Question:**

1. Which **chunking strategy** should you choose to maximize safety and context retention?
2. You need to ensure the chatbot *only* searches through documents tagged as "Cardiology" when a user asks about heart conditions. How do you implement this filtering **before** the vector search?