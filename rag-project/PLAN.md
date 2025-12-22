Below is a **concrete project evolution plan** that turns your *working Bedrock KB + Lambda RAG* into a **Week-1–complete, exam-aligned hands-on system**.
Nothing theoretical—each bullet maps to **code, config, or architecture you will actually implement**.

---

# Project Evolution: From “Working RAG” → “Domain-1 Complete System”

You already have:

* Bedrock KB (Titan embeddings)
* S3 Vectors
* Lambda query path
* Nova Lite generation model

We’ll **extend**, not rewrite.

---

## Exam framing + architecture mindset

**Hands-on goal:** Prove you understand *what the exam is actually testing* by encoding it into architecture choices.

### What you will do

Create an **ARCHITECTURE.md** in the repo with two sections:

#### In scope (explicitly covered by your project)

* Bedrock-managed RAG (KB + RetrieveAndGenerate)
* FM selection & routing
* Vector store tradeoffs
* IAM + governance boundaries
* Cost & latency awareness

#### Out of scope (explicitly not built)

* Fine-tuning
* Training pipelines
* Custom embedding training
* Model internals

**Why this matters**
The exam tests **design judgment**, not coding trivia. Writing this doc forces you to separate *what AWS wants you to know* from what it doesn’t.

---

## Bedrock fundamentals

**Hands-on goal:** Model selection, routing, and resilience

### Changes to implement

#### 1. Multi-model generation routing

Update Lambda to support **model routing**:

```json
{
  "question": "...",
  "modelTier": "cheap | balanced | high-quality"
}
```

Routing logic:

* `cheap` → Nova Micro
* `balanced` → Nova Lite (default)
* `high-quality` → Claude / Llama (if enabled)

**What you learn**

* Generation model is a *runtime choice*
* Embeddings stay fixed
* This answers: *“Why Bedrock instead of SageMaker?”*

---

#### 2. Fallback strategy (resilience)

Wrap `retrieve_and_generate` in:

* Primary model call
* Automatic fallback on:

  * `ThrottlingException`
  * `ModelTimeoutException`

**Exam concept covered**

* Cross-model resilience
* Graceful degradation
* Availability over perfection

---

## RAG architectures

**Hands-on goal:** Defend **two RAG architectures with code**

### Architecture A — Bedrock KB (what you have)

* Managed ingestion
* Managed retrieval
* Managed grounding

### Architecture B — DIY RAG (add this)

Add a **second Lambda** that:

1. Calls `Retrieve` only
2. Manually assembles the prompt
3. Calls `InvokeModel`

Use the *same vector store*.

**What you compare**

| Aspect       | KB RAG | DIY RAG |
| ------------ | ------ | ------- |
| Control      | Medium | High    |
| Complexity   | Low    | High    |
| Auditability | High   | Medium  |
| Exam default | ✅      | ❌       |

You now *prove* why KB exists.

---

### Chunking strategy experiment

Create **two KBs** using the same docs:

* KB-A: small chunks + overlap
* KB-B: larger chunks

Run identical queries, compare:

* Recall
* Hallucination rate
* Cost

---

## Vector stores & embeddings

**Hands-on goal:** Make vector store choice *visible*, not theoretical

### Extend project with a **vector-store switch**

Add a config flag:

```json
"vectorBackend": "s3vectors | opensearch"
```

Implement **one** additional backend:

* OpenSearch Serverless (vector + keyword)

Run the same queries:

* With filters
* Without filters
* With metadata constraints

**What you’ll experience**

* Why S3 Vectors is default
* When OpenSearch is justified
* Why DynamoDB vectors rarely win in exams

---

### Embedding tradeoff experiment

Create **two ingestion runs**:

* Titan 1024-dim
* Titan 512-dim

Measure:

* Retrieval quality
* Latency
* Cost

You’ll *feel* the tradeoff instead of memorizing it.

---

## Prompt engineering & governance

**Hands-on goal:** Treat prompts as **versioned infrastructure**

### Changes to implement

#### 1. Prompt templates in S3

Move prompts out of code:

```
s3://rag-prompts/
  ├── v1/system.txt
  ├── v2/system.txt
```

Lambda loads prompt version by config:

```json
"promptVersion": "v2"
```

---

#### 2. Prompt regression testing

Create a small test harness:

* Fixed questions
* Expected answer patterns
* Run nightly (or manually)

Store results in DynamoDB or S3.

**What this teaches**

* Prompt governance ≠ clever wording
* It’s CI/CD for reasoning

---

## Domain 1 consolidation

**Hands-on goal:** Produce a **1-page cheat sheet backed by experience**

Your cheat sheet will include:

* When to use KB vs DIY RAG
* Vector store decision table
* Embedding vs generation model rules
* Governance patterns
* Failure modes & fallbacks

Because you built everything, this won’t be memorization.

---

# Final project structure (what you’ll end with)

```
/rag-project
 ├── ingestion/
 │    ├── kb-s3vectors/
 │    ├── kb-opensearch/
 │
 ├── query/
 │    ├── lambda-kb-rag/
 │    ├── lambda-diy-rag/
 │
 ├── prompts/
 │    ├── v1/
 │    ├── v2/
 │
 ├── experiments/
 │    ├── chunking/
 │    ├── embeddings/
 │    ├── vectorstores/
 │
 ├── ARCHITECTURE.md
 ├── DOMAIN1-CHEATSHEET.md
```

---

