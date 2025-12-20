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

###**Day 2 Scenario Quiz**You are building a RAG application for a medical research firm. The source documents are complex research papers where the **meaning** of a paragraph relies heavily on the sentences immediately preceding it. If you cut a paragraph in the middle, the medical context could be lost, leading to dangerous hallucinations.

**Question:**

1. Which **chunking strategy** should you choose to maximize safety and context retention?
2. You need to ensure the chatbot *only* searches through documents tagged as "Cardiology" when a user asks about heart conditions. How do you implement this filtering **before** the vector search?