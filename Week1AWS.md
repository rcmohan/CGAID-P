## Project: “AskMyDocs” — Bedrock RAG Q&A Web App (serverless, minimal moving parts)

A simple production-style project that covers **Bedrock APIs, embeddings vs text models, RAG/Knowledge Bases, guardrails, logging, and basic evaluation**.

---

## What it does

1. You upload PDFs/text files (policies, design docs, runbooks) to S3.
2. A Bedrock **Knowledge Base** indexes them (chunking + embeddings + vector store managed by Bedrock).
3. A web UI lets you ask questions.
4. The backend retrieves relevant chunks and uses a **text model** to answer, with **citations** and **guardrails**.

---

## AWS services (simple stack)

* **Amazon S3**: document storage
* **Amazon Bedrock**

  * **Knowledge Bases** (managed RAG)
  * **Embedding model** (e.g., Titan Embeddings)
  * **Text model** (e.g., Claude or Titan Text)
  * **Guardrails**
  * **Model invocation logging** (to CloudWatch/S3)
* **AWS Lambda**: backend API
* **Amazon API Gateway**: HTTPS endpoint
* **AWS Amplify (or S3 static hosting + CloudFront)**: simple frontend
* **Amazon CloudWatch**: logs/metrics

This maps tightly to the exam themes: RAG, prompt governance, safety controls, monitoring, and API integration.

---

## Architecture (high level)

1. **Ingestion**

   * Upload docs → S3 (prefix `/docs/`)
   * Knowledge Base syncs from S3 on schedule or on-demand

2. **Query path**

   * UI → API Gateway → Lambda
   * Lambda calls Bedrock Knowledge Base “retrieve-and-generate” (or retrieve + invoke)
   * Guardrails applied to input/output
   * Response includes:

     * answer
     * citations (document + chunk refs)
     * token usage (for cost visibility)

---

## Core features (keep it small but “exam-complete”)

### Must-have (1–2 days)

* Document upload to S3
* Knowledge Base configured on that S3 path
* “Ask” endpoint that returns grounded answer + citations
* Streaming optional (nice-to-have)

### Nice-to-have (adds 1 day)

* **Conversation memory**: store chat history in DynamoDB (last N turns)
* **Prompt versioning**: store prompt template versions in SSM Parameter Store or S3
* **Simple evaluation**: a “golden Q&A” JSON file + nightly Lambda job that runs Qs and records results (CloudWatch logs)

---

## Bedrock concepts you’ll implement

* **Embeddings vs text models**

  * Embeddings used by Knowledge Base to build vectors
  * Text model used to generate final answer
* **Bedrock APIs**

  * Invoke via KB retrieve/generate (simplest) or:

    * Retrieve → InvokeModel / InvokeModelWithResponseStream
* **Guardrails**

  * Block toxic content
  * Redact PII
  * Enforce “answer only from provided documents; if not found, say you don’t know”
* **Observability**

  * Invocation logs + CloudWatch metrics for latency/tokens/errors

---

## Step-by-step implementation plan

1. **Create S3 bucket**: `askmydocs-<env>`
2. **Enable Bedrock models** in your region (text + embeddings)
3. **Create a Bedrock Knowledge Base**

   * Data source: S3 bucket `/docs/`
   * Embedding model: Titan Embeddings
   * Managed vector store (keep default)
4. **Create a Guardrail**

   * Add topic filters + PII redaction
   * Add grounding instruction: “If not in sources, say you don’t know.”
5. **Lambda + API Gateway**

   * `POST /ask`: `{ question, sessionId }`
   * Calls KB retrieve-and-generate with guardrail
   * Returns `{ answer, citations, usage }`
6. **Frontend**

   * Minimal React page (Amplify hosting)
   * Upload docs (pre-signed S3 upload) + ask questions
7. **Logging**

   * CloudWatch Logs for Lambda
   * Enable Bedrock invocation logging to CloudWatch/S3

---

## Deployment options (pick one)

* **Fastest:** AWS Console for KB/Guardrails + SAM/CDK for Lambda/API + Amplify for UI
* **More “professional”:** CDK for everything possible (KB/guardrails may still require some console steps depending on coverage in your region/tooling)

---

## Acceptance tests (so you know it’s “done”)

* Upload a doc containing a specific fact → ask → answer includes citation
* Ask something not in docs → model replies “I don’t know based on provided sources”
* Enter an email/phone number in prompt → output redacts it (guardrail)
* CloudWatch shows token usage + latency per request

---

If you want, I can produce:

* A concrete **CDK (TypeScript or Python) skeleton** for S3 + Lambda + API Gateway + IAM
* A minimal **Lambda handler** showing the Knowledge Base retrieve-and-generate call pattern
* A tiny **React UI** (Amplify-ready) with doc upload + chat UI + citations rendering


Below is a **clean, exam-ready summary** of exactly what you did in IAM to make Amazon Bedrock Knowledge Bases work. This is the distilled “mental model” you should keep.

---

## Goal

Create an **Amazon Bedrock Knowledge Base with S3 Vectors** using least-privilege IAM, without using the root user.

---

## 1. You stopped using the root user (required)

* Root user **cannot** create Bedrock Knowledge Bases
* You created a **non-root IAM user**
* This user is the one interacting with the Bedrock console

**Why it matters:**
Root is blocked by design for GenAI governance. This is intentional and exam-relevant.

---

## 2. IAM user permissions (who *requests* the KB)

You granted the IAM user **control-plane permissions** to manage Bedrock and delegate execution.

### IAM user needed:

* **AmazonBedrockFullAccess**
  → to list models, create Knowledge Bases, manage configs
* **S3 access to the document bucket**
  → to select and validate data sources
* **S3 Vectors permissions**
  → to allow creation of vector buckets
* **`iam:PassRole`**
  → to allow Bedrock to assume an execution role

### Key insight

`iam:PassRole` must target the **execution role ARN**, not an S3 or S3 Vectors resource.

---

## 3. You learned S3 Vectors is a separate IAM namespace

* Vector storage is **not regular S3**
* It uses the **`s3vectors:*`** permission namespace
* Bedrock cannot create vector buckets unless explicitly allowed

**Why this caused errors earlier:**
Older docs say “Bedrock-managed vector store,” but the new backend is **Amazon S3 Vectors**, which requires explicit IAM permissions.

---

## 4. You created the correct Bedrock execution role (most important step)

This role is **assumed by the Bedrock service**, not by you.

### The execution role required:

#### A. Correct trust policy (this was the blocker)

```json
{
  "Effect": "Allow",
  "Principal": { "Service": "bedrock.amazonaws.com" },
  "Action": "sts:AssumeRole"
}
```

**Not**:

* `s3.amazonaws.com`
* `lambda.amazonaws.com`
* Any other service

Without this, Bedrock could not finish KB creation—even though vector buckets were created.

---

#### B. Permissions on the execution role

You attached:

* `s3vectors:*`
  → to create and manage vector buckets
* `s3:GetObject` / `s3:ListBucket`
  → to read documents from the source bucket

**No Bedrock permissions were needed on the role.**

---

## 5. You separated responsibilities correctly

| Component       | Responsibility                |
| --------------- | ----------------------------- |
| IAM User        | Create and configure KB       |
| Execution Role  | Read data + manage vectors    |
| Bedrock Service | Orchestrates KB lifecycle     |
| Trust Policy    | Allows Bedrock to assume role |

This separation is **core AWS security design**.

---

## 6. Why vector buckets existed but KB did not

* Vector bucket creation happens **before** role assumption
* Bedrock failed at the **assume-role step**
* AWS rolled back the Knowledge Base but left the vector buckets
* Fixing the trust policy resolved this immediately

---

## 7. Final working state (mental checklist)

### IAM User

* AmazonBedrockFullAccess
* S3 access to docs bucket
* s3vectors permissions
* iam:PassRole → execution role

### Execution Role

* Trusts `bedrock.amazonaws.com`
* Has `s3vectors:*`
* Has S3 read access

### Result

* Knowledge Base created successfully
* Vector store managed automatically
* Documents can be synced
* RAG queries work

---

## Why this is exam-relevant

You exercised **all** of these tested concepts:

* Root user restrictions
* Least privilege
* Service-assumed roles
* Trust vs permission policies
* New service IAM namespaces (S3 Vectors)
* Delegation via `iam:PassRole`

This is exactly the level of understanding the **AWS Certified Generative AI Developer – Professional** exam expects.

---

If you want next:

* A one-page **IAM diagram**
* A **CDK version** of this setup
* Cleanup of orphaned vector buckets
* Or help wiring the KB into Lambda / API Gateway


Yes — **with one important nuance**.

Below is the exact behavior of your current Bedrock Knowledge Base + S3 Vectors setup.

---

## What happens in your setup (authoritative behavior)

### ✅ 1. Initial creation

When the Knowledge Base is created and you run **Sync** the first time:

* Bedrock:

  1. Reads documents from your S3 data bucket
  2. Chunks the documents
  3. Generates embeddings using **Titan Text Embeddings**
  4. Stores embeddings in the **S3 Vector bucket**
* The Knowledge Base becomes queryable immediately after sync completes

So yes — embeddings are created and stored automatically.

---

## ⚠️ 2. What happens when you add *new* documents

### ❌ Not fully automatic by default

If you upload new files to the S3 bucket:

* **Embeddings are NOT generated immediately**
* **Nothing happens until a sync is triggered**

This is by design.

---

## How new documents get embedded (two supported modes)

### Option A — Manual sync (default)

You must explicitly run:

* **Knowledge Base → Data sources → Sync**

Bedrock will:

* Detect new/changed files
* Embed only the delta
* Update the S3 Vector store

This is the **default, safest, lowest-cost mode**.

---

### Option B — Automatic sync (optional)

You can enable **automatic sync** for the data source.

When enabled:

* Bedrock periodically scans the S3 bucket
* New or modified objects are automatically embedded
* Vector store is updated without manual action

Tradeoff:

* Higher background cost
* Less control over when embeddings are created

---

## What Bedrock does *not* do

❌ No S3 event trigger by default
❌ No real-time embedding on upload
❌ No Lambda invocation unless you add one

Bedrock uses **batch-style sync**, not event-driven ingestion.

---

## Exactly what is stored where

| Component        | Stored data                        |
| ---------------- | ---------------------------------- |
| S3 data bucket   | Original documents                 |
| S3 Vector bucket | Vector embeddings + metadata       |
| Bedrock KB       | Pointers + config (not embeddings) |

Bedrock does **not** store embeddings internally — S3 Vectors is the source of truth.

---

## What happens during queries

When you query the Knowledge Base:

1. Query text is embedded (on demand)
2. Similar vectors are retrieved from S3 Vectors
3. Retrieved chunks are passed to the text model
4. The model generates a grounded answer

Embeddings are **not regenerated** during query time.

---

## Cost implications (important)

You pay for embeddings:

* **Only during sync**
* **Only for new/changed content**

You do **not** pay repeatedly for the same documents.

This is why manual sync is recommended for learning and exams.

---

## Recommended configuration for you

Given your goals (learning + certification):

* ✅ Manual sync
* ✅ Small document batches
* ✅ Sync only when content changes

This gives:

* Predictable cost
* Clear mental model
* Exam-aligned behavior

---

## One-sentence summary (exam-ready)

> In Amazon Bedrock Knowledge Bases with S3 Vectors, embeddings are generated during data source sync operations and stored in the vector store; new documents are embedded only when a manual or automatic sync is triggered.

---

If you want next:

* How to enable **automatic sync safely**
* How to detect **which documents were embedded**
* How to force **re-embedding**
* How to wire S3 events → Lambda → KB sync (advanced, optional)
Below is a **precise, system-level explanation** of what lives inside an **Amazon Bedrock Knowledge Base (KB)** and how its **pointers + configuration** control retrieval and generation.

---

## What is *actually* stored in a Bedrock Knowledge Base?

A Bedrock KB **does not store embeddings or documents**.
It stores **metadata and orchestration instructions**.

Think of the KB as a **control plane object**.

### The KB contains:

### 1. **Data source pointers**

Logical references to:

* S3 document bucket (original files)
* S3 Vector bucket (embeddings + chunk metadata)

These are **ARN-level references**, not data copies.

---

### 2. **Embedding configuration**

Defines:

* Which embedding model to use (e.g., Titan Text Embeddings v2)
* Expected embedding dimensionality
* Text-only vs multimodal constraints

This ensures:

* Query embeddings are generated **compatibly** with stored vectors
* Retrieval space is consistent

---

### 3. **Chunking configuration**

Defines how documents were split:

* Chunk size
* Overlap
* Parsing strategy

This matters because:

* Retrieval returns **chunks**, not documents
* Chunk boundaries affect semantic recall

---

### 4. **Vector store configuration**

Defines:

* Which vector store backend to query (S3 Vectors, OpenSearch, etc.)
* Index identifiers
* Similarity metric (e.g., cosine similarity)

This is how the KB knows **where and how to search**.

---

### 5. **Retrieval configuration**

Defines:

* `topK` (how many chunks to retrieve)
* Filters (if any)
* Hybrid vs vector-only retrieval (backend-dependent)

This controls **precision vs recall**.

---

### 6. **Generation configuration**

Defines:

* Which text generation model to use
* How retrieved chunks are injected into the prompt
* Prompt templates and grounding behavior

This is what turns retrieval into **RAG**.

---

## What is *not* in the KB

❌ No embeddings
❌ No document content
❌ No vectors
❌ No indexes

All of that lives in **S3** and **S3 Vectors**.

---

## How pointers + config are used during a query (step-by-step)

Let’s walk through a text query:

> *“What is our refund policy for digital purchases?”*

### Step 1: Query arrives at the Knowledge Base

The KB is the **entry point**.

---

### Step 2: Query embedding is generated

The KB:

* Reads its **embedding config**
* Calls the **same embedding model** used during ingestion
* Produces a query vector with the correct dimension

This avoids vector-space mismatch.

---

### Step 3: Vector store lookup

Using its **vector store pointer**, the KB:

* Sends the query vector to the correct S3 Vector bucket
* Applies similarity search using the configured metric
* Retrieves the top-K most similar **chunk vectors**

At this point:

* The KB does **not** know document semantics
* It only orchestrates where to search and how

---

### Step 4: Chunk metadata resolution

Each returned vector includes metadata:

* Source document reference
* Chunk text
* Offsets / identifiers

The KB uses this metadata to:

* Assemble the retrieved context
* Preserve document boundaries if needed
* Track citations

---

### Step 5: Prompt construction

The KB applies its **generation config**:

* Injects retrieved chunks into a system prompt
* Applies grounding rules (e.g., “answer only from sources”)
* Sends the prompt to the selected text model

---

### Step 6: Answer generation

The model generates an answer:

* Conditioned on retrieved chunks
* With optional citations
* Without needing to know anything about S3 or vectors

---

## Why the KB abstraction exists (important insight)

Without a KB, **you** would have to:

* Track which embedding model was used
* Ensure dimension compatibility
* Query the correct vector index
* Manage chunk metadata
* Assemble prompts manually

The KB centralizes that logic.

---

## Mental model (use this)

> **The Knowledge Base is a routing table + policy engine for RAG.**
> It doesn’t store knowledge; it knows *where* the knowledge lives and *how* to use it.

---

## One-paragraph exam-ready explanation

> An Amazon Bedrock Knowledge Base stores configuration and pointers to data sources and vector stores, including embedding model selection, chunking strategy, retrieval parameters, and generation settings. At query time, the Knowledge Base uses this configuration to embed the query, retrieve relevant vectors from the configured vector store, resolve chunk metadata, and construct a grounded prompt for the text generation model. The Knowledge Base itself does not store embeddings or documents.

---

If you want next:

* A diagram comparing **KB vs DIY RAG**
* How this differs from **Agents + KB**
* How **filters and metadata** affect retrieval
* How **re-embedding** works when configs change


---
Below is a **clear, system-level explanation** of all four topics, written to align with how AWS Bedrock actually works (and how the exam frames it).

---

# 1. Knowledge Base (KB) vs DIY RAG

## Architecture comparison

### DIY RAG (you build everything)

```
User Query
   │
   ▼
Embed Query (you pick model)
   │
   ▼
Vector DB (you manage index)
   │
   ▼
Top-K Chunks
   │
   ▼
Prompt Assembly (manual)
   │
   ▼
LLM Invoke
```

### Bedrock Knowledge Base (managed RAG)

```
User Query
   │
   ▼
Bedrock Knowledge Base
   ├─ Uses stored embedding config
   ├─ Routes to correct vector store
   ├─ Applies retrieval config
   ├─ Resolves chunk metadata
   └─ Builds grounded prompt
        │
        ▼
     LLM Invoke
```

---

## What KB removes vs DIY

| Concern                     | DIY RAG     | Bedrock KB   |
| --------------------------- | ----------- | ------------ |
| Embedding model consistency | You manage  | Guaranteed   |
| Vector DB lifecycle         | You manage  | Managed      |
| Chunk metadata tracking     | Manual      | Automatic    |
| Prompt grounding            | Manual      | Built-in     |
| Re-embedding logic          | Custom code | Built-in     |
| Security / IAM              | Complex     | Standardized |

**Mental model:**
DIY RAG = *pipeline code*
KB = *RAG control plane*

---

# 2. KB alone vs Agents + KB

This is one of the most important distinctions.

---

## KB alone (retrieval-only RAG)

```
User → KB → Retrieve → Generate → Answer
```

Characteristics:

* Single query
* Stateless
* No tools
* No memory
* Deterministic retrieval + generation

Best for:

* Q&A over documents
* Search
* Policy lookups
* Documentation bots

---

## Agents + KB (reasoning + actions)

```
User
  │
  ▼
Agent
  ├─ Decides: "Do I need knowledge?"
  │      └─ Query KB (optional)
  ├─ Decides: "Do I need a tool?"
  │      └─ Invoke Lambda / API
  ├─ Maintains session state
  └─ Iterates reasoning steps
        │
        ▼
     Final Answer
```

Key differences:

| Aspect       | KB only     | Agent + KB        |
| ------------ | ----------- | ----------------- |
| Reasoning    | Single step | Multi-step        |
| Tool use     | No          | Yes               |
| Memory       | No          | Yes               |
| Control flow | Fixed       | Dynamic           |
| KB role      | Core engine | One of many tools |

**Important:**
A KB does **not** decide *when* to retrieve.
An Agent does.

---

# 3. How filters and metadata affect retrieval

## What metadata is

Every embedded chunk stores metadata such as:

* source document name
* S3 object key
* page number
* section headers
* custom tags (if provided)

Example:

```json
{
  "source": "refund-policy.pdf",
  "page": 4,
  "department": "finance",
  "effective_date": "2024-01-01"
}
```

---

## How filters work

Filters are applied **before similarity scoring**.

### Without filters

```
Query → All vectors → Similarity ranking → Top-K
```

### With filters

```
Query
  │
  ▼
Filter (metadata constraints)
  │
  ▼
Subset of vectors
  │
  ▼
Similarity ranking
```

Example filter:

* `department = finance`
* `effective_date >= 2024-01-01`

This:

* Reduces noise
* Improves precision
* Prevents irrelevant chunks from ever being scored

**Exam insight:**
Filters constrain *candidate space*, not similarity math.

---

# 4. How re-embedding works when configs change

This is critical.

---

## When re-embedding is required

Re-embedding is triggered when you change:

| Change               | Re-embed needed? |
| -------------------- | ---------------- |
| Embedding model      | ✅ Yes            |
| Embedding dimensions | ✅ Yes            |
| Chunk size / overlap | ✅ Yes            |
| Document content     | ✅ Yes            |
| Retrieval `topK`     | ❌ No             |
| Generation model     | ❌ No             |

Why:

* Embeddings must live in the **same vector space**
* Chunk boundaries define semantic units

---

## What happens during re-embedding

When you trigger a re-sync:

1. KB detects config mismatch
2. Existing vectors are marked stale
3. Documents are re-chunked
4. New embeddings are generated
5. Vector store is replaced/updated
6. Retrieval uses new index

Old vectors are **not reused**.

---

## Why Bedrock stores config centrally

The KB stores:

* embedding model ID
* vector dimension
* chunking strategy

This ensures:

* Query embeddings always match stored vectors
* No silent corruption
* Deterministic retrieval behavior

DIY systems often get this wrong.

---

# Final mental models (memorize these)

### KB

> A configuration-driven RAG orchestrator that knows *where* embeddings live and *how* to use them.

### Agent

> A reasoning engine that decides *when* to retrieve, *what* to retrieve, and *what else* to do.

### Filters

> Pre-retrieval constraints that reduce the candidate vector space.

### Re-embedding

> Mandatory whenever vector space semantics change.

---

If you want next, I can:

* Draw an **Agent + KB + Tools** execution trace
* Show **CloudTrail events** for KB sync vs query
* Compare **Bedrock KB vs OpenSearch hybrid search**
* Map these concepts directly to **exam question patterns**


---


Below is the **exact, mechanics-level answer** to how **Amazon Bedrock Knowledge Bases (KB)** decide **Top-K retrieval**, and what *can* and *cannot* change it.

---

## Short answer (exam-ready)

> **Top-K is not inferred from the prompt.**
> It is a **retrieval configuration parameter** set on the Knowledge Base or supplied explicitly in the API call. The prompt can influence *how retrieved chunks are used*, but not *how many are retrieved*.

---

## Where Top-K is actually decided

There are **three possible places** Top-K can come from, in this order of precedence:

### 1️⃣ API request (highest priority)

If you call the KB programmatically (SDK / API) and specify `numberOfResults` (or equivalent), **that value is used**.

Example (conceptual):

```json
{
  "retrieveAndGenerateConfiguration": {
    "knowledgeBaseConfiguration": {
      "retrievalConfiguration": {
        "vectorSearchConfiguration": {
          "numberOfResults": 5
        }
      }
    }
  }
}
```

This **overrides** the KB default.

---

### 2️⃣ Knowledge Base default retrieval configuration

If the API request does **not** specify Top-K:

* KB uses its **configured default Top-K**
* This is typically set during KB creation or left at the AWS default

This is what the **console “Test Knowledge Base”** uses.

---

### 3️⃣ Vector store capability (lowest priority)

The vector backend (S3 Vectors, OpenSearch, etc.) enforces:

* Maximum allowable K
* Performance constraints

But it **does not decide K**—it only executes it.

---

## What the prompt can and cannot do

### ❌ The prompt cannot:

* Change Top-K
* Request “more results” dynamically
* Control vector similarity thresholds
* Affect embedding search parameters

The prompt is **downstream of retrieval**.

---

### ✅ The prompt *can*:

* Ask the model to:

  * summarize fewer points
  * cite selectively
  * ignore weak evidence
* Influence how retrieved chunks are synthesized

But by then, retrieval has already happened.

---

## Internal retrieval flow (step-by-step)

```
User Query
   │
   ▼
KB reads retrieval config
   │
   ├─ topK = 3 (example)
   │
   ▼
Embed query
   │
   ▼
Vector similarity search
   │
   ▼
Return top-K chunks (exactly K)
   │
   ▼
Prompt construction
   │
   ▼
Text model generation
```

Top-K is resolved **before** prompt construction.

---

## What happens if the “needed results” change?

### Scenario 1: You want fewer results

You must:

* Lower `topK` in:

  * the API call, or
  * the KB retrieval configuration

Prompt alone is insufficient.

---

### Scenario 2: You want more results for complex questions

You must:

* Increase `topK` in:

  * the API call (recommended), or
  * the KB default

Common pattern:

* Simple Q → `topK = 3`
* Complex analysis → `topK = 8–10`

This decision belongs to **application logic**, not the model.

---

## Advanced: What about reranking?

Some backends support **reranking**, but note:

* Reranking:

  * reorders retrieved chunks
  * does **not** change K
* Flow:

  ```
  top-K retrieval → rerank → top-K (same count)
  ```

If a reranker drops chunks internally, that is **post-retrieval logic**, not KB retrieval logic.

---

## Why Bedrock does it this way (important insight)

AWS deliberately separates:

* **Retrieval control** → deterministic, auditable
* **Generation behavior** → probabilistic, prompt-driven

This ensures:

* Predictable cost
* Reproducible retrieval
* Governance and safety
* No “hidden” retrieval changes via prompt injection

This design is **explicitly aligned with Responsible AI** principles and exam expectations.

---

## Exam-ready rules to remember

1. **Top-K is a retrieval parameter, not a prompt feature**
2. Prompt ≠ retrieval control
3. API-level config overrides KB defaults
4. Agents (not KBs) decide *when* to change Top-K dynamically
5. Changing Top-K does **not** require re-embedding

---

## One-paragraph exam answer

> In Amazon Bedrock Knowledge Bases, the number of retrieved results (Top-K) is determined by the retrieval configuration defined on the Knowledge Base or explicitly provided in the retrieve-and-generate API call. The prompt does not dynamically influence Top-K selection; it only affects how the retrieved chunks are synthesized during generation. This separation ensures deterministic retrieval behavior, predictable cost, and governance.

---

If you want next, I can:

* Show how **Agents dynamically vary Top-K**
* Explain **similarity score thresholds vs Top-K**
* Compare **Top-K vs Max Tokens tradeoffs**
* Map this to **common exam trick questions**
