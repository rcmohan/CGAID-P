# Daily Plan

## Plan for 12/15

### BEFORE 12/26 — REQUIRED SETUP (CRITICAL)

Complete **before vacation**:

#### 1. Download / Prepare Offline Materials

* Exam Guide PDF
* Your cheat sheets (print or local)
* Architecture diagrams (PNG/PDF)
* Personal notes per domain

#### 2. Local Hands-On Prep

* Clone or scaffold:

  * A local RAG demo (even pseudocode)
  * Step Functions JSON definitions
  * Prompt templates + versions
* Download SDK docs / saved examples if needed

#### 3. Offline Practice Assets

* Printed:

  * Service comparison tables
  * Threat → Control → AWS Service matrix
* Flashcards (paper or local app)

---

## WEEK 1: FOUNDATIONS & RAG (12/15–12/21)

### **Day 1: Bedrock Fundamentals**

**Focus:** Amazon Bedrock fundamentals

**Objective:** Understand Model Selection and Inference Parameters.

**1. Model Selection (The "Who"):**

* **Titan:** AWS Native. Good for cost-effective, general tasks. *Titan Image Generator* for images. *Titan Embeddings* for RAG.
* **Claude (Anthropic):** The "Brain." High reasoning, large context window (200k tokens). Use for complex chain-of-thought, code generation, and summarizing massive documents.
* **Llama (Meta):** Open/General purpose. Good for chat and text generation where you might want fine-tuning flexibility later.
* **Jurassic (AI21):** Good for specialized text generation/instruction following.

**2. Inference Parameters (The "How"):**

* **Temperature (0.0 - 1.0):**
  * Low (0): Deterministic. Math, code, data extraction.
  * High (1): Creative. Storytelling, brainstorming.
* **Top P (Nucleus Sampling):** Cuts off the "long tail" of unlikely words.
  * Example: If you set Top P to 0.9, the model only considers the top words that make up 90% of the probability. It ignores the weird, rare words.
* **Top K:**
  * Simpler. Just picks from the top K words (e.g., Top 10).

**3. Throughput (The "Cost"):**
* **On-Demand:** Standard. Pay per token. Great for spiky or low volume.
* **Provisioned:** Reserved capacity (Provisioned Throughput Units - PTUs).
  * *Crucial for Exam:* You **MUST** use Provisioned Throughput if you want to use a **Fine-tuned model** or if you need **guaranteed performance/bandwidth**.

---

* Bedrock APIs, model families, embeddings vs text models
* On-demand vs provisioned throughput
* Cross-region inference, fallback strategies

**Milestone:**
You can answer: *“Why Bedrock instead of SageMaker here?”*

---

**Day 1 Checkpoint Question:**
You are designing a chatbot for a bank that answers customer questions about interest rates. The answers must be strictly factual based on the retrieved documents.
1.  Which **Temperature** range would you use?
2.  If the bank decides to **fine-tune** a Titan model on their specific banking terminology, which **Throughput Mode** is legally required?


## Plan for **Day 2: Data Engineering for GenAI**.

Before we dive in, here is the quick answer to yesterday’s question:

* **Temperature:** 0 (For strict factual accuracy).
* **Throughput:** Provisioned Throughput (This is **mandatory** for fine-tuned models).

---

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

### **Day 2 Scenario Quiz**

You are building a RAG application for a medical research firm. The source documents are complex research papers where the **meaning** of a paragraph relies heavily on the sentences immediately preceding it. If you cut a paragraph in the middle, the medical context could be lost, leading to dangerous hallucinations.

**Question:**

1. Which **chunking strategy** should you choose to maximize safety and context retention?
2. You need to ensure the chatbot *only* searches through documents tagged as "Cardiology" when a user asks about heart conditions. How do you implement this filtering **before** the vector search?


## Plan for **Day 3: Vector Stores & RAG**.

First, let's check your understanding from yesterday's scenario:

* **Chunking Strategy:** **Semantic Chunking** was the right choice. Since the meaning depends on context, you want to cut the text only where the topic naturally changes, not at an arbitrary word count.
* **Filtering:** You would use **Metadata Filters**. By attaching a `.metadata.json` file (e.g., `{"department": "cardiology"}`) to your source documents in S3, you can tell the vector store to "only search vectors where department = cardiology."

---

### **Day 3: Vector Stores & RAG****Goal:** 

Understand where the "brains" of your application live. Once you've chunked your data, you need to store it in a way that the AI can search by *meaning*, not just keywords.

**1. The "Embedding" Concept**

* **What is it?** Converting text into a long list of numbers (a vector).
* **Why?** Computers can't compare "King" and "Queen" to see if they are related. But if "King" = `[0.9, 0.1]` and "Queen" = `[0.85, 0.2]`, the math shows they are close neighbors.
* **The Model:** **Amazon Titan Embeddings** is the standard AWS answer for creating these vectors. It supports up to 8k tokens input.

**2. Amazon Bedrock Knowledge Bases (The "Easy Button")**
For the exam, this is often the preferred solution because it is **fully managed**.

* **How it works:** You point it to your S3 bucket. It automatically handles the fetching, chunking, embedding (using Titan), and storing in a vector database.
* **The Sync:** Crucial Exam Detail! When you upload a new file to S3, the Knowledge Base **does not** see it automatically. You must call the `StartIngestionJob` API (or click "Sync" in the console) to update the index.

**3. Choosing a Vector Store**
Bedrock Knowledge Bases manages the vector store for you, but you need to choose *which* store to use under the hood:

* **OpenSearch Serverless:** The default, easiest, and most scalable option.
* **Aurora PostgreSQL (with pgvector):** Best if you *already* have an Aurora database and want to keep your vector data next to your relational business data.
* **Pinecone / Redis Enterprise:** Supported third-party options if your company already uses them.

**4. RAG Implementation Pattern**
The exam will ask you to order these steps:

1. **User asks a question.**
2. **Embed the question:** Convert user text to a vector using the *same* model (Titan Embeddings) used for the docs.
3. **Semantic Search:** Find the chunks in the database that are mathematically closest to the question vector.
4. **Augment:** Take those chunks and paste them into the prompt (context window).
5. **Generate:** Send the prompt + chunks to the LLM (Claude/Titan) to get the answer.

---

###**Day 3 Scenario Quiz**You are architecting a solution for a law firm. They have thousands of legal case files in PDF format stored in S3. They want a chatbot to answer questions about past cases.
**Constraints:**

1. They want to minimize operational overhead (Serverless preferred).
2. They strictly require that if a PDF is deleted from S3, it must be removed from the search results immediately upon the next sync.
3. They want to filter search results by "Case Year" before the model generates an answer.

**Question:**
Which combination of **Bedrock feature** and **Vector Store** is the most appropriate, and how would you implement the "Case Year" requirement?