
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