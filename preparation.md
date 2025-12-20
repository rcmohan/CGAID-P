# Preparing for **AWS Certified Generative AI Developer – Professional (AIP-C01)**

A **day-by-day, exam-aligned preparation plan** for **AWS Certified Generative AI Developer – Professional (AIP-C01)** covering **12/15/25 → 1/9/26**, with:

* **1/8 and 1/9 reserved strictly for review**
* **12/26–1/1 as offline-only study**
* Clear **weekly goals and milestones**
* A **prep checklist** to set up everything needed *before* the offline week

The plan is tightly mapped to the **official AIP-C01 domains and weightings** (Domains 1–5) as defined in the AWS Exam Guide .

## Plan from Gemini

### **Strategic Overview**

* **Phase 1 (Dec 15 - Dec 24):** The "Heavy Lift." We cover Domains 1, 2, and 3 (approx. 77% of the exam).
* **Phase 2 (Dec 25):** The "Offline Pack." We prepare your "go-bag" of study materials.
* **Phase 3 (Dec 26 - Jan 1):** The "Offline Maintenance." Light review to keep neural pathways active without burnout.
* **Phase 4 (Jan 2 - Jan 7):** Operations, Optimization, and Practice.
* **Phase 5 (Jan 8 - Jan 9):** Final Review & Exam.

I have updated our hidden learning plan to match this daily breakdown.

### **Daily Plan & Goals**

#### **Week 1: The Foundation (Domains 1 & 2)**

* **Goal:** Master how to pick a model, how to talk to it (Prompting), and how to give it memory (RAG).
* **Dec 15 (Today):** **Bedrock Fundamentals.**
  * *Topics:* Model families (Titan, Claude, Llama, Cohere), Inference parameters, Throughput modes.
* **Dec 16 (Tue):** **Data Engineering.**
  * *Topics:* Data prep for RAG, Chunking strategies (Fixed vs. Semantic), S3 data sources.
* **Dec 17 (Wed):** **Vector Stores & RAG.**
  * *Topics:* Embeddings (Titan Embeddings), Vector DBs (OpenSearch, Aurora, Pinecone), Knowledge Bases.
* **Dec 18 (Thu):** **Prompt Engineering.**
  * *Topics:* Zero/Few-shot, Chain-of-Thought, Bedrock Prompt Management, Template versions.
* **Dec 19 (Fri):** **Agents.**
  * *Topics:* Bedrock Agents, Action Groups, OpenAPI Schemas, Lambda integration.
* **Dec 20 (Sat):** **Deployment & Integration.**
  * *Topics:* Streaming APIs, VPC Endpoints, PrivateLink, Cross-Region Inference.
* **Dec 21 (Sun):** **Week 1 Review.**
  * *Activity:* I will give you a tough scenario-based quiz covering all the above.

#### **Week 2: Safety & The "Go-Bag" (Domain 3)**

* **Goal:** Secure the application and prepare for offline study.
* **Dec 22 (Mon):** **Safety (Guardrails).**
* **Dec 23 (Tue):** **Security (IAM & Encryption).**
* **Dec 24 (Wed):** **Governance (Logging & Compliance).**
* **Dec 25 (Thu):** **Offline Prep Day.**
  * *Task:* We will create a "Cheat Sheet" together for you to save as a PDF/Note. I will provide code snippets (Boto3) and architecture patterns for you to save for offline review.

#### **Week 3: Vacation (Maintenance Mode)**

* **Dec 26 - Jan 1:** **Offline Review.**
  * *Goal:* Spend 15-20 mins/day reviewing the cheat sheets and code snippets we prepared. No heavy lifting.

#### **Week 4: Operations & Polish (Domains 4 & 5)**

* **Goal:** Optimization, Cost, and Troubleshooting.
* **Jan 2 (Fri):** **Cost Optimization.**
* **Jan 3 (Sat):** **Performance & Latency.**
* **Jan 4 (Sun):** **Monitoring (CloudWatch/X-Ray).**
* **Jan 5 (Mon):** **Evaluation (LLM-as-a-Judge).**
* **Jan 6 (Tue):** **Troubleshooting Scenarios.**
* **Jan 7 (Wed):** **Full "Mock" Session** (We simulate exam conditions).

#### **Final Days**

* **Jan 8 (Thu):** Review Notes.
* **Jan 9 (Fri):** **Exam Day!**

---

## Plan from ChatGPT

## Plan from ChatGPT 5.2

**Timeline**

* **Week 1 (12/15–12/21):** Domain 1 – Foundations, RAG, Vector Stores (31%)
* **Week 2 (12/22–12/25):** Domain 2 – Agents, Integration, Deployment (26%)
* **Week 3 (12/26–1/1):** Offline consolidation + hands-on rehearsal
* **Week 4 (1/2–1/7):** Domains 3–5 – Security, Optimization, Testing (43%)
* **Final Review:** 1/8–1/9

---

## WEEK 1: FOUNDATIONS & RAG (12/15–12/21)

#### Weekly Goal

You should be able to **design and justify**:

* A complete Bedrock-based GenAI architecture
* RAG pipelines with correct vector store choices
* FM selection, routing, and resilience strategies

---

### 12/15 (Sun)

**Focus:** Exam framing + architecture mindset

* Read exam guide end-to-end (fast skim)
* Lock domain weightings and question types
* Write down:

  * What is *in scope*
  * What is explicitly *out of scope*

**Milestone:**
You can explain *what the exam is testing vs what it is not*.

---

### 12/16 (Mon)

**Focus:** Amazon Bedrock fundamentals

* Bedrock APIs, model families, embeddings vs text models
* On-demand vs provisioned throughput
* Cross-region inference, fallback strategies

**Milestone:**
You can answer: *“Why Bedrock instead of SageMaker here?”*

---

### 12/17 (Tue)

**Focus:** RAG architectures

* Bedrock Knowledge Bases vs custom RAG
* Chunking strategies (fixed, semantic, hierarchical)
* Metadata enrichment patterns

**Milestone:**
You can draw **2 RAG architectures** and defend each.

---

### 12/18 (Wed)

**Focus:** Vector stores & embeddings

* OpenSearch vs Aurora pgvector vs DynamoDB vectors
* Hybrid search, reranking
* Embedding selection tradeoffs (dimension, cost, latency)

**Milestone:**
You can choose the *right* vector store under exam pressure.

---

### 12/19 (Thu)

**Focus:** Prompt engineering & governance (Domain 1.6)

* Prompt templates, versioning
* Prompt Flows
* Prompt regression testing

**Milestone:**
You understand **prompt governance as a system**, not text tweaking.

---

### 12/20 (Fri)

**Focus:** Domain 1 consolidation

* Review all Domain 1 tasks
* Write a **1-page Domain 1 cheat sheet**

---

### 12/21 (Sat)

**Light review**

* Re-draw one full GenAI system from memory
* No new content

---

## WEEK 2: AGENTS & INTEGRATION (12/22–12/25)

#### Weekly Goal

You should confidently handle **agentic workflows**, **tool calling**, and **enterprise integration questions**.

---

### 12/22 (Sun)

**Focus:** Agentic AI fundamentals

* Bedrock Agents
* Strands Agents, Agent Squad
* MCP concepts (client/server, stateless vs ECS-based)

**Milestone:**
You can explain *when NOT to use agents*.

---

### 12/23 (Mon)

**Focus:** Orchestration patterns

* Step Functions (ReAct, retries, circuit breakers)
* Human-in-the-loop patterns
* Tool failure handling

**Milestone:**
You can order multi-step workflows correctly (ordering questions).

---

### 12/24 (Tue)

**Focus:** FM deployment & API integration

* Lambda vs Bedrock vs SageMaker endpoints
* Streaming responses
* Rate limiting, retries, backoff

---

### 12/25 (Wed)

**Light day**

* Domain 2 recap
* Update cheat sheets

---

## WEEK 3: OFFLINE CONSOLIDATION (12/26–1/1)

### Constraints

* **No internet**
* **Offline notes + offline hands-on only**

---

### 12/26–1/1 (Daily, ~30–45 min/day)

Each day:

* Review **one domain**
* Re-draw architectures from memory
* Walk through “what would AWS expect?” scenarios

Suggested focus:

* 12/26: Domain 1
* 12/27: Domain 2
* 12/28: Domain 3
* 12/29: Domain 4
* 12/30: Domain 5
* 12/31–1/1: Full mental walkthroughs

**Milestone:**
You can mentally simulate answering exam questions without notes.

---

## WEEK 4: SECURITY, OPTIMIZATION, TESTING (1/2–1/7)

#### Weekly Goal

You should *dominate* governance, cost, observability, and evaluation questions.

---

### 1/2 (Thu)

**Focus:** AI Safety & Guardrails (Domain 3)

* Bedrock Guardrails
* Prompt injection defense
* Output verification

---

### 1/3 (Fri)

**Focus:** Data security & governance

* IAM, VPC endpoints
* PII detection
* Audit logging

---

### 1/4 (Sat)

**Focus:** Cost & performance optimization (Domain 4)

* Token efficiency
* Caching
* Throughput planning

---

### 1/5 (Sun)

**Focus:** Monitoring & observability

* Bedrock invocation logs
* CloudWatch metrics
* X-Ray tracing

---

### 1/6 (Mon)

**Focus:** Testing & evaluation (Domain 5)

* LLM-as-a-Judge
* RAG evaluation
* Regression testing

---

### 1/7 (Tue)

**Full mock mental exam**

* End-to-end review
* Identify weak spots
* Final cheat sheet edits

---

## FINAL REVIEW

### 1/8 (Wed)

* Review **notes only**
* Service comparisons
* Guardrails vs Comprehend vs Macie
* No new studying

### 1/9 (Thu – Exam Day)

* Light skim
* Rest
* Exam

---

## FINAL PASS CRITERIA (SELF-CHECK)

Before exam day, you should be able to:

* Instantly pick the **AWS-native** answer
* Eliminate **out-of-scope services**
* Explain **why a solution is governed, scalable, and secure**