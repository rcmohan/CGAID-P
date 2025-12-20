import os
import json
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

def lambda_handler(event, context):
    # Attempt to parse input from event directly (flat JSON)
    # The user's test event uses "input" for the question
    question = event.get('input') or event.get('question')
    sessionId = event.get('sessionId')
    topK = event.get('topK', 5)
    filters = event.get('filters')

    # Handle case where input might be in 'body' (API Gateway)
    if not question and 'body' in event:
        body = event['body']
        if isinstance(body, str):
            body = json.loads(body)
        question = body.get('input') or body.get('question')
        sessionId = body.get('sessionId')
        topK = body.get('topK', 5)
        filters = body.get('filters')

    response = call_bedrock(question, sessionId, topK, filters)
    parsed_response = parse_response(response)
    return parsed_response

def call_bedrock(question, sessionId, topK, filters):
    # Use bedrock-agent-runtime for Knowledge Base RAG operations
    bedrock_agent_runtime = boto3.client('bedrock-agent-runtime')
    knowledge_base = os.environ['KNOWLEDGE_BASE_ID']
    model_id = os.environ['BEDROCK_MODEL_ID']
    
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
                'knowledgeBaseId': knowledge_base,
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
    response = bedrock_agent_runtime.retrieve_and_generate(**kwargs)
    return response
    
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
    