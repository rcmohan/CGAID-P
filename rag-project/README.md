## Repository root: `/rag-project`

### `ARCHITECTURE.md`

* Your two RAG diagrams (KB vs DIY)
* “In scope / out of scope”
* Model routing + fallback design
* Vector store decision rationale

### `DOMAIN1-CHEATSHEET.md`

* 1-page exam summary built from your experiments

---

# 1) Ingestion code

## `ingestion/kb-s3vectors/`

**Purpose:** Everything required to create and run the Knowledge Base using **S3 Vectors**.

Put these here:

* `kb_config.md`

  * KB name, KB ID, region, embedding model, chunking settings
* `sync_instructions.md`

  * How to upload docs to S3
  * How to run manual sync
  * How to enable auto-sync (optional)
* `iam/`

  * `execution-role-trust.json` (the Bedrock assume-role trust policy)
  * `execution-role-permissions.json` (`s3vectors:*` + S3 read bucket policy)
  * `user-passrole.json` (`iam:PassRole` policy for your console user)
* Optional automation scripts:

  * `sync_kb.py` (script to trigger sync via API if you later automate it)

**No application Lambda code belongs here.** This is “data plane setup”.

---

## `ingestion/kb-opensearch/`

**Purpose:** Alternative KB using **OpenSearch Serverless** (for hybrid search/reranking experiments).

Put these here:

* `kb_config.md` (OpenSearch collection info, index config)
* `opensearch_setup.md` (collection creation, network/access policy)
* `iam/` policies specific to OpenSearch access
* `rerank_notes.md` (what reranking config you used, results)

---

# 2) Query code

## `query/lambda-kb-rag/`

**Purpose:** The main Lambda that calls:

* `bedrock-agent-runtime.retrieve_and_generate` against your KB

This is where your current file goes.

Put these here:

* `app.py` (your Lambda handler; basically your `rag1.py` cleaned up)
* `requirements.txt` (boto3/botocore versions if packaging)
* `event_examples/`

  * `console_test.json` (Lambda console test payload)
  * `apigw_body.json` (API Gateway `body` example)
* `config/`

  * `models.json` (mapping tiers → model ARNs, e.g., Nova Micro/Lite/Claude)
  * `defaults.json` (default topK, default prompt version)
* `lib/`

  * `event_parser.py` (robust parsing for body/input/direct JSON)
  * `bedrock_kb_client.py` (wrap retrieve_and_generate)
  * `response_formatter.py` (normalize citations into your output format)

Env vars used here:

* `KNOWLEDGE_BASE_ID`
* `BEDROCK_MODEL_ARN` (or per-request model routing)
* `AWS_REGION`

This folder becomes your “production” API.

---

## `query/lambda-diy-rag/`

**Purpose:** The DIY RAG Lambda for comparison.

Put these here:

* `app.py`

  * calls KB `Retrieve` (not generate)
  * builds the prompt manually
  * calls `bedrock-runtime.invoke_model` to generate
* `lib/`

  * `retrieval.py` (KB retrieve call)
  * `prompt_builder.py` (inject chunks into prompt template)
  * `invoke_model.py` (Bedrock runtime call)
  * `citations.py` (citation formatting)

This folder is your “manual control baseline”.

---

# 3) Prompt governance artifacts

## `prompts/v1/`

* `system.txt` (system prompt template)
* `rag_instructions.txt` (grounding rules, citation requirements)
* `answer_format.txt` (output formatting rules)

## `prompts/v2/`

* Same files, updated
* add `CHANGELOG.md` to explain what changed and why

**No Lambda code here.** These are treated as versioned assets.

---

# 4) Experiments

## `experiments/chunking/`

**Purpose:** Compare chunking settings and outcomes.

Put these here:

* `kb-a-small-chunks.md`
* `kb-b-large-chunks.md`
* `questions.json` (the same set of test questions)
* `results/`

  * `kb-a-results.json`
  * `kb-b-results.json`
* `analysis.md` (what improved/worsened and why)

---

## `experiments/embeddings/`

**Purpose:** Compare 1024 vs 512 dims (or model variants if you try them).

Put these here:

* `embedding_runs.md` (what embedding model + dims used)
* `questions.json`
* `results/`
* `analysis.md`

---

## `experiments/vectorstores/`

**Purpose:** Compare S3 Vectors vs OpenSearch (and optional hybrid search).

Put these here:

* `setup.md`
* `questions.json`
* `results/`
* `analysis.md`

---

# 5) Optional infrastructure-as-code

## `infra/` (optional but recommended)

If you later CDK/SAM this:

* `cdk/` or `sam/`

  * Lambda + API Gateway definitions
  * IAM role definitions
  * Env var wiring

---

# Where your current file goes

Your current Lambda file (`rag1.py`) belongs in:

✅ `query/lambda-kb-rag/app.py` 

Then you gradually refactor helper functions into `query/lambda-kb-rag/lib/`.
