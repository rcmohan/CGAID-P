# Lambda DIY RAG

This project implements a **custom RAG (Retrieval-Augmented Generation) pipeline** using AWS Lambda and Amazon Bedrock. Unlike the managed `retrieve_and_generate` API, this "DIY" approach gives full control over the retrieval and generation steps, allowing for custom prompting and multi-model support.

## Architecture

1.  **Retrieve**: Uses `bedrock-agent-runtime.retrieve()` to query the Knowledge Base and fetch relevant text chunks.
2.  **Generate**: Uses `bedrock-runtime.invoke_model()` to send the prompt (context + question) to a Foundation Model (FM) for the final answer.

## Key Features

*   **Multi-Model Support**: The `lambda_handler` dynamically constructs the correct API payload based on the selected model family:
    *   **Anthropic Claude 3**: Uses the Messages API format, adding the required `anthropic_version` header and top-level inference parameters.
    *   **Amazon Nova Pro**: Uses the Nova-specific body structure, placing parameters inside `inferenceConfig` and using a simplified `content` structure (no `type` field).
*   **Context Injection**: Manually formats retrieved chunks into a structured prompt context.
*   **Streaming Response Parsing**: Correctly reads and parses the `StreamingBody` returned by `invoke_model` to extract the final text.

## Implementation Details

### Model-Specific Payload Logic
Different Bedrock models require different JSON body structures. This implementation handles them robustly:

```python
# Claude (Anthropic)
if "claude" in model_id:
    body_content = {
        "anthropic_version": "bedrock-2023-05-31", # Required
        "messages": [...],
        "max_tokens": 100,
        "temperature": 0.5
    }

# Nova (Amazon)
elif "nova" in model_id:
    body_content = {
        "messages": [
            {
                "role": "user",
                "content": [
                     { "text": ... } # No "type": "text" allowed
                ]
            }
        ],
        "inferenceConfig": { # specific to Nova
            "max_new_tokens": 100, 
            "temperature": 0.5
        }
    }
```

## Recent Fixes & Improvements

During development, the following issues were resolved to ensure production readiness:

1.  **API Parsing**: Fixed `KeyError` issues when parsing various input event formats (API Gateway vs. direct Lambda invocation).
2.  **Robust Imports**: Resolved `ImportModuleError` by removing imports for undefined exceptions (`ThrottlingException`) and sticking to standard `ClientError` handling.
3.  **Metadata Safety**: Added safeguards when accessing `source` metadata to prevent crashes on missing fields.
4.  **Claude Compatibility**: Added the mandatory `anthropic_version` field to requests targeting Claude 3 models.
5.  **Nova Compatibility**:
    *   Moved `max_tokens` to `inferenceConfig` to avoid "extraneous key" validation errors.
    *   Removed `type: text` from message content to satisfy Nova's stricter schema.
6.  **Response Handling**: Implemented `json.loads(response['body'].read())` to correctly parse the streaming output from `invoke_model`.

