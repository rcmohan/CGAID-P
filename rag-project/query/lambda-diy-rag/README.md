# DIY RAG with **Amazon Bedrock**.

You are correct. Thank you for calling it out.

There is **no `generate()` method** on the **`bedrock-agent-runtime`** boto3 client. My earlier wording conflated **Bedrock Agent Runtime** with **Bedrock Model Runtime**. Below is the **accurate, API-level correction**, aligned with the SDK and the exam.

All of this is in the context of **Amazon Bedrock**.

---

# Correct API Surfaces (Very Important)

AWS Bedrock has **two different runtime planes**:

| Runtime plane     | Boto3 client            | Purpose                  |
| ----------------- | ----------------------- | ------------------------ |
| **Model runtime** | `bedrock-runtime`       | Invoke foundation models |
| **Agent runtime** | `bedrock-agent-runtime` | Knowledge Bases + Agents |

They are intentionally separated.

---

## 1. `bedrock-agent-runtime.retrieve()`

### ✔ Exists

### ✔ Retrieval-only

**What it does**

* Queries a **Bedrock Knowledge Base**
* Returns **chunks + metadata**
* **Does not invoke a foundation model**

**Use cases**

* Custom RAG
* Agent reasoning loops
* LLM-as-a-judge
* Full prompt control

### Payload

```python
response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId="kb-123456",
    retrievalQuery={
        "text": "What is the refund policy?"
    },
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 5
        }
    }
)
```

### Output (simplified)

```json
{
  "retrievalResults": [
    {
      "content": { "text": "Refunds are allowed within 30 days..." },
      "location": {
        "s3Location": { "uri": "s3://docs/policy.pdf" }
      },
      "score": 0.82,
      "metadata": { "source": "policy.pdf" }
    }
  ]
}
```

---

## 2. **There is NO `bedrock-agent-runtime.generate()`**

### ❌ Does NOT exist

**Why**

* Agent Runtime **never directly invokes FMs**
* Model invocation is handled by:

  * `bedrock-runtime.invoke_model()`, or
  * Managed flows like `retrieve_and_generate()`, or
  * Agent orchestration (`invoke_agent`)

This separation is **architectural**, not accidental.

---

## 3. `bedrock-runtime.invoke_model()`

### ✔ Correct replacement for “generate”

This is the **actual FM invocation API**.

### What it does

* Calls a foundation model
* You provide **all prompt/context**
* No retrieval unless *you* do it

### Payload (example: Claude)

```python
response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-3-haiku",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Answer using this policy: Refunds allowed within 30 days." }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2
    })
)
```

### Output (simplified)

```json
{
  "content": [
    {
      "type": "text",
      "text": "Returns after 45 days are not allowed."
    }
  ]
}
```

---

## 4. `bedrock-agent-runtime.retrieve_and_generate()`

### ✔ Exists

### ✔ Managed RAG

**What it does**

* Retrieval from Knowledge Base
* Prompt construction
* Model invocation
* Citations
* Guarded, opinionated flow

You **do not** call the model directly.

### Payload

```python
response = bedrock_agent_runtime.retrieve_and_generate(
    input={ "text": "Can I return an item after 45 days?" },
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "kb-123456",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        }
    }
)
```

### Output (simplified)

```json
{
  "output": {
    "text": "Returns are only allowed within 30 days."
  },
  "citations": [
    {
      "retrievedReferences": [
        {
          "location": {
            "s3Location": { "uri": "s3://docs/policy.pdf" }
          }
        }
      ]
    }
  ]
}
```

---

# Correct Mental Model (Exam-Critical)

### ❌ Incorrect

> `bedrock-agent-runtime.generate()`

### ✅ Correct

* **Retrieve only:** `bedrock-agent-runtime.retrieve()`
* **Generate only:** `bedrock-runtime.invoke_model()`
* **Managed RAG:** `bedrock-agent-runtime.retrieve_and_generate()`
* **Agents:** `bedrock-agent-runtime.invoke_agent()`

---

## Exam One-Liner

> **Agent Runtime never “generates” directly; only the Model Runtime invokes FMs.**

---


Below is the **correct, production-accurate way** to pass **retrieved RAG context** into `bedrock_runtime.invoke_model()` when you are doing **custom RAG** (i.e., *not* using `retrieve_and_generate()`).

All examples are in the context of **Amazon Bedrock** and align with how the exam expects you to reason.

---

# Core Principle (Exam-Critical)

> **`invoke_model()` has no concept of “RAG.”**
> You must **explicitly inject retrieved context into the prompt payload** in a model-specific format.

There is **no reserved field** like `context`, `documents`, or `retrieved_chunks`.

---

# High-Level Flow (Correct)

1. Retrieve documents
   → `bedrock-agent-runtime.retrieve()`
2. Normalize / rank / trim chunks
3. Inject them into the **prompt body**
4. Call
   → `bedrock-runtime.invoke_model()`

---

# Step 1: Retrieved Chunks (Example Shape)

From `retrieve()` you typically get:

```json
[
  {
    "content": { "text": "Refunds are allowed within 30 days with receipt." },
    "metadata": { "source": "policy.pdf" },
    "score": 0.82
  },
  {
    "content": { "text": "Digital goods are non-refundable." },
    "metadata": { "source": "digital_policy.pdf" },
    "score": 0.76
  }
]
```

You must **decide**:

* How many chunks
* Order
* Formatting
* Citations or not

That decision is **on you**.

---

# Step 2: Canonical RAG Prompt Structure (Recommended)

This structure is **exam-safe and production-safe**:

```
SYSTEM / INSTRUCTIONS
CONTEXT (retrieved documents)
QUESTION
CONSTRAINTS
```

---

# Step 3: Inject Context — Claude / Messages API Example

```python
context_blocks = "\n\n".join(
    f"[Source: {c['metadata']['source']}]\n{c['content']['text']}"
    for c in retrieved_chunks
)

prompt = f"""
You are a customer support assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

<context>
{context_blocks}
</context>

Question:
Can I return an item after 45 days?
"""

response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-3-haiku",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": prompt }
                ]
            }
        ],
        "max_tokens": 300,
        "temperature": 0.1
    })
)
```

### Why this works

* Context is **explicitly bounded**
* Hallucination risk reduced
* Easy to audit and log
* Matches AWS best-practice patterns

---

# Step 4: Alternative — Strict JSON Context (High Control)

Used when you want **schema enforcement or downstream parsing**.

```python
context_payload = [
    {
        "source": c["metadata"]["source"],
        "text": c["content"]["text"]
    }
    for c in retrieved_chunks
]

prompt = {
    "instructions": "Answer using only the provided context.",
    "context": context_payload,
    "question": "Can I return an item after 45 days?"
}

response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-3-haiku",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": json.dumps(prompt) }
                ]
            }
        ],
        "max_tokens": 300
    })
)
```

---

# Step 5: Guardrails You Should Add (Exam-Relevant)

When doing custom RAG, AWS expects you to mention:

* **Token trimming**
* **Chunk limits**
* **Instruction hierarchy**
* **Explicit grounding rules**

Example addition:

```
If the context does not explicitly answer the question, respond with "Insufficient information."
Do not use prior knowledge.
```

---

# Common Mistakes (Seen on Exams)

❌ Passing retrieved text as a separate field
❌ Assuming `invoke_model()` understands citations
❌ Sending too many chunks (token blow-up)
❌ Not bounding context (hallucinations)

---

# Comparison With `retrieve_and_generate()`

| Custom RAG (`invoke_model`) | Managed RAG         |
| --------------------------- | ------------------- |
| You format context          | AWS formats context |
| You control prompts         | Limited control     |
| No automatic citations      | Citations included  |
| Required for agents         | Not agent-friendly  |

---

# One-Line Exam Answer

> **To pass RAG context to `invoke_model()`, you must manually embed retrieved text into the prompt body in a model-specific format; Bedrock does not provide a dedicated context field.**

If you want next, I can:

* Show **agent-style RAG injection**
* Compare **Claude vs Titan prompt formats**
* Walk through **token budgeting math**
* Provide **exam trick questions** on RAG injection

---
---


## 1. `bedrock_agent_runtime.retrieve()`

### What it does

* **Retrieval only**
* Queries a **Bedrock Knowledge Base**
* Returns **raw chunks + metadata**
* **Does NOT call an FM**

### When to use

* You want **full control** over:

  * Prompt construction
  * Post-processing
  * Multi-step orchestration
* You are building **custom RAG**, agent flows, or evaluations

### Exam mindset

> “Separation of concerns”
> Retrieval is decoupled from generation.

### Typical payload

```python
response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId="kb-123456",
    retrievalQuery={
        "text": "What is the refund policy?"
    },
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 5
        }
    }
)
```

### Output shape (simplified)

```json
{
  "retrievalResults": [
    {
      "content": { "text": "Refunds are allowed within 30 days..." },
      "location": { "s3Location": { "uri": "s3://docs/policy.pdf" } },
      "score": 0.82,
      "metadata": {
        "source": "policy.pdf",
        "lastUpdated": "2024-01-10"
      }
    }
  ]
}
```

---

## 2. `bedrock_agent_runtime.generate()`

### What it does

* **Generation only**
* Calls a **foundation model**
* No retrieval, no grounding unless **you provide context**

### When to use

* You already have:

  * Retrieved documents
  * Tool outputs
  * Structured context
* You want **full prompt control**

### Exam mindset

> “FM invocation without orchestration”

### Typical payload

```python
response = bedrock_agent_runtime.generate(
    modelId="anthropic.claude-3-haiku",
    inputText="""
    Using the following context, answer the question.

    Context:
    Refunds are allowed within 30 days with receipt.

    Question:
    Can I return an item after 45 days?
    """,
    generationConfiguration={
        "temperature": 0.2,
        "maxTokens": 300
    }
)
```

### Output shape (simplified)

```json
{
  "outputText": "Based on the policy, returns after 45 days are not allowed..."
}
```

---

## 3. `bedrockagent.retrieve_and_generate()`

### What it does

* **One-call managed RAG**
* Performs:

  1. Retrieval from Knowledge Base
  2. Prompt construction
  3. FM invocation
* AWS manages chunking, grounding, and citations

### When to use

* If you want:

  * **Fastest production RAG**
  * Minimal glue code
  * Built-in grounding
* If you accept **less customization**

### Exam mindset

> “Managed, governed RAG”

### Typical payload

```python
response = bedrockagent.retrieve_and_generate(
    input={
        "text": "Can I return an item after 45 days?"
    },
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "kb-123456",
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        }
    }
)
```

### Output shape (simplified)

```json
{
  "output": {
    "text": "Returns are only allowed within 30 days..."
  },
  "citations": [
    {
      "retrievedReferences": [
        {
          "location": {
            "s3Location": { "uri": "s3://docs/policy.pdf" }
          }
        }
      ]
    }
  ]
}
```

---

## Side-by-Side Comparison (Exam Gold)

| API                       | Retrieval | Generation | Control    | Citations | Typical Use          |
| ------------------------- | --------- | ---------- | ---------- | --------- | -------------------- |
| `retrieve()`              | ✅         | ❌          | High       | ❌         | Custom RAG, agents   |
| `generate()`              | ❌         | ✅          | High       | ❌         | Prompt-only FM calls |
| `retrieve_and_generate()` | ✅         | ✅          | Low–Medium | ✅         | Managed RAG          |

---

## How to choose (Exam Logic)

**Choose `retrieve_and_generate()` when:**

* Simple RAG
* Governance & grounding matter
* Speed to production > flexibility

**Choose `retrieve()` + `generate()` when:**

* Agents
* Tool calling
* Multi-step reasoning
* Custom prompts or evaluations
* LLM-as-Judge flows

---

## One-Line Memory Hook

> **`retrieve()` gets facts, `generate()` thinks, `retrieve_and_generate()` does both—opinionated and managed.**

