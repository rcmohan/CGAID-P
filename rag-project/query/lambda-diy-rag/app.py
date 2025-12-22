import os
import json
import logging
import time
import boto3
from botocore.exceptions import ClientError 

## Initialize the Boto3 client
"""
JSON Input Format
{
  "question": "string",
  "sessionId": "string (optional)",
  "topK": 5,
  "filters": { "department": "finance" }
}

JSON Output Format
{
  "answer": "string",
  "citations": [
    { "source": "s3://bucket/key", "snippet": "..." }
  ],
  "usage": { "inputTokens": 0, "outputTokens": 0 }
}

"""

"""

PS1:    If you get a timeout error, like this following, change the Timeout to 30 seconds in the Lambda function:

Status: Failed
Response:
{
  "errorType": "Sandbox.Timedout",
  "errorMessage": "Error: Task timed out after 3.00 seconds"
}



"""

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Make sure to change 

def lambda_handler(event, context):
    # Attempt to parse input from event directly (flat JSON)
    # The user's test event uses "input" for the question
    question = event.get('input') or event.get('question')
    sessionId = event.get('sessionId')
    topK = event.get('topK', 5)
    filters = event.get('filters')
    model_tier = event.get('modelTier', 'balanced')

    # Handle case where input might be in 'body' (API Gateway)
    if not question and 'body' in event:
        body = event['body']
        if isinstance(body, str):
            body = json.loads(body)
        question = body.get('input') or body.get('question')
        sessionId = body.get('sessionId')
        topK = body.get('topK', 5)
        filters = body.get('filters')

    model_id = decide_model_tier(question, model_tier)

    response = call_bedrock(question, sessionId, topK, filters, model_id)
    parsed_response = parse_response(response)
    return parsed_response

def decide_model_tier(question, model_tier):
    if model_tier == "cheap":
        model_id = os.environ["MODEL_NOVA_MICRO"]
    elif model_tier == "performance":
        model_id = os.environ["MODEL_NOVA_LITE"]
    elif model_tier == "deep":
        model_id = os.environ["MODEL_CLAUDE_HAIKU"]
    else:
        model_id = os.environ["MODEL_NOVA_LITE"]  # default
    return model_id

def call_bedrock(question, sessionId, topK, filters, model_id):

    logger.info(f"Calling Bedrock with question: {question}, sessionId: {sessionId}, topK: {topK}, filters: {filters}, modelTier: {model_id}")

    knowledge_base_id = os.environ["KNOWLEDGE_BASE_ID"]
    # Configure vector search
    vector_search_config = {
        "numberOfResults": topK
    }
    if filters:
        vector_search_config["filter"] = filters

    # Call the RetrieveAndGenerate API
    max_tokens = 100
    temperature = 0.7
    try_backup = True # TODO: implement backup
    logger.info(f"Calling RetrieveAndGenerate with knowledge_base_id: {knowledge_base_id}, model_id: {model_id}, question: {question}, sessionId: {sessionId}, topK: {topK}, filters: {filters}, temperature: {temperature}, max_tokens: {max_tokens}, try_backup: {try_backup}")
    response = retrieve_and_generate(knowledge_base_id, model_id, question, sessionId, topK, filters, temperature, max_tokens)
    logger.info(f"RetrieveAndGenerate response: {response}")
    return response
    
def retrieve_and_generate(knowledge_base_id, model_id, question, sessionId, topK, filters, temperature, max_tokens):

    bedrock_agent_runtime = boto3.client(
        "bedrock-agent-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

    embeddings = bedrock_agent_runtime.retrieve(knowledgeBaseId=knowledge_base_id, retrievalQuery={"text": question}, retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": topK}})
    bedrock_runtime = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

    try:
        retrieved_chunks = embeddings["retrievalResults"]
        context_payload = [
            {
                "source": c["metadata"].get("source", "unknown") if c.get("metadata") else "unknown",
                "text": c["content"]["text"]
            }
            for c in retrieved_chunks
        ]

        prompt = {
            "instructions": "Answer the question using only the provided context.",
            "context": context_payload,
            "question": question
        }

        if "claude" in model_id:
            body_content = {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            { "type": "text", "text": json.dumps(prompt) }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
        elif "nova" in model_id:
            body_content = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            { "text": json.dumps(prompt) }
                        ]
                    }
                ],
                "inferenceConfig": {
                    "max_new_tokens": max_tokens,   
                    "temperature": temperature
                }
            }
        else:
            # Fallback for other models (e.g. Titan, default to basic)
             body_content = {
                "inputText": json.dumps(prompt),
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature
                }
            }
        logger.info(f"Calling InvokeModel with body_content: {body_content}")
        response = bedrock_runtime.invoke_model(modelId=model_id, contentType="application/json", accept="application/json", body=json.dumps(body_content))
        logger.info(f"InvokeModel response: {response}")
        
        response_body = json.loads(response.get('body').read())
        output_text = ""
        
        # Parse output based on response structure
        if "content" in response_body:
             # Standard Messages API (Claude)
             output_text = response_body["content"][0]["text"]
        elif "output" in response_body:
             # Nova often returns output -> message -> content
             message = response_body["output"].get("message", {})
             content = message.get("content", [])
             if content:
                 output_text = content[0].get("text", "")
        
        # Return in a format that parse_response expects (or just return the parsed dict directly)
        return {
            "output": { "text": output_text },
            "citations": [], # Citations not supported in DIY simplified mode
            "sessionId": sessionId
        }
    except ClientError as e:
        logger.error(f"RetrieveAndGenerate failed: {e}")
        raise


def parse_response(response):
    # Extract the answer from the output
    answer = response.get('output', {}).get('text', '')
    
    # Extract citations if available
    citations = response.get('citations', [])
    
    # Extract sessionId to maintain context
    session_id = response.get('sessionId', '')
    
    return {
        "answer": answer,
        "citations": citations,
        "sessionId": session_id
    }
    