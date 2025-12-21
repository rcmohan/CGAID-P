# Knowledge Base Sync Instructions

## Uploading Documents

1. Navigate to Amazon S3
2. Open bucket: `askmydocs-dev`
3. Upload supported documents (PDF, TXT, DOCX, etc.)
4. Maintain logical folder structure if needed (optional)

---

## Triggering Ingestion

1. Open Amazon Bedrock Console
2. Go to **Knowledge Bases**
3. Select `knowledge-base-quick-start-pbary`
4. Under **Data sources**, click **Sync**

The sync process:
- Parses documents
- Chunks text
- Generates embeddings
- Stores vectors in S3 Vectors

---

## Sync Behavior

- **Incremental**: Only new or changed objects are reprocessed
- **Idempotent**: Re-syncing unchanged files does not duplicate embeddings
- **Manual**: Sync is triggered explicitly (no automatic S3 event trigger)

---

## When Re-Sync Is Required

Re-sync is required if:
- New documents are added
- Existing documents are modified
- Chunking configuration changes
- Embedding model changes

Re-sync is **not required** if:
- Generation model changes
- Prompt templates change
