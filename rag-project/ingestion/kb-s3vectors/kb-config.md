# Knowledge Base Configuration (S3 Vectors)

## Overview
This Knowledge Base is implemented using **Amazon Bedrock Knowledge Bases**
with **Amazon S3 Vectors** as the vector store and **Titan Text Embeddings v2**
as the embedding model.

This KB is used for Retrieval-Augmented Generation (RAG) via the
`RetrieveAndGenerate` API.

---

## Knowledge Base Details

- **Knowledge Base Name**: `knowledge-base-quick-start-pbary`
- **Knowledge Base ID**: `YORHGZPMCH`
- **Region**: `us-east-1`
- **Status**: Available
- **Created**: December 17, 2025 (UTC-05:00)

---

## Data Source

- **Source Type**: Amazon S3
- **Source Bucket**:  
  `s3://askmydocs-dev/`
- **Sync Status**: Available
- **Sync Mode**: Manual (via console “Sync”)

Documents uploaded to this bucket are ingested into the Knowledge Base
during a sync operation.

---

## Embeddings Configuration

- **Embedding Model**: Titan Text Embeddings v2
- **Vector Type**: Float embeddings
- **Vector Dimensions**: 1024

> NOTE: The embedding model is fixed at ingestion time. All future queries
embed user input using the same model to ensure vector compatibility.

---

## Vector Store

- **Vector Store Type**: Amazon S3 Vectors
- **Vector Bucket**:  
  `bedrock-knowledge-base-05xjk2`
- **Vector Index**:  
  `bedrock-knowledge-base-default-index`
- **Vector Index ARN**:  
  `arn:aws:s3vectors:us-east-1:055022918897:bucket/bedrock-knowledge-base-05xjk2/index/bedrock-knowledge-base-default-index`

S3 Vectors stores all embeddings generated during ingestion and supports
similarity search during retrieval.

---

## Retrieval & Generation

- **RAG Type**: Vector store–based retrieval
- **Generation Model (Query Time)**: Configurable  
  (e.g., `amazon.nova-lite-v1:0`)
- **Embedding Model (Query Time)**: Titan Text Embeddings v2 (same as ingestion)

---

## Notes

- Changing the **generation model** does **not** require re-embedding.
- Changing the **embedding model** requires a full re-sync.
- Vector store creation and management are handled automatically by Bedrock.
