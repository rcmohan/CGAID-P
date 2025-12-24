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

logger = logging.getLogger()
logger.setLevel(logging.INFO)

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
    # Use bedrock-agent-runtime for Knowledge Base RAG operations
    bedrock_agent_runtime = boto3.client(
        "bedrock-agent-runtime",
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

    knowledge_base_id = os.environ["KNOWLEDGE_BASE_ID"]
    # Configure vector search
    vector_search_config = {
        "numberOfResults": topK
    }
    if filters:
        vector_search_config["filter"] = filters

    # Prepare arguments
    kwargs = {
        'input': {
            'text': question
        },
        'retrieveAndGenerateConfiguration': {
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': knowledge_base_id,
                'modelArn': model_id,
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': vector_search_config
                }
            },
            'type': 'KNOWLEDGE_BASE'
        }
    }
    
    # Add sessionId if it exists (to continue a conversation)
    if sessionId:
        kwargs['sessionId'] = sessionId

    # Call the RetrieveAndGenerate API
    logger.info(f"Calling RetrieveAndGenerate with kwargs: {kwargs}")
    try_backup = false
    try:
        logger.info(f"Attempting Primary Model: {model_id}")
        return bedrock_agent_runtime.retrieve_and_generate(**kwargs)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code in ['ThrottlingException', 'ModelTimeoutException']:
            logger.warning(f"Primary model failed ({error_code}). Switch to Backup.")
            return try_backup(bedrock_agent_runtime, kwargs) # PASS THE CLIENT
        else:
            raise e

def try_backup(client, kwargs):
    # Hardcode a cheaper/faster model for backup
    backup_model_id = os.environ.get("MODEL_NOVA_MICRO") 
    kwargs['retrieveAndGenerateConfiguration']['knowledgeBaseConfiguration']['modelArn'] = backup_model_id
    
    logger.info(f"Attempting Backup Model: {backup_model_id}")
    return client.retrieve_and_generate(**kwargs)


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
    