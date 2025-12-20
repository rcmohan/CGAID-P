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
