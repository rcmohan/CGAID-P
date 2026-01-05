# Self Assessment Questions

## Task 1.1

1.1.1. **Your company needs to implement a customer support chatbot that can handle routine inquiries and escalate complex issues to human agents. The chatbot should integrate with your existing customer relationship management (CRM) system and maintain context throughout conversations.**

**Which integration pattern would be most appropriate for this solution?**

- Synchronous API integration with direct calls to Amazon Bedrock
- Asynchronous batch processing using Amazon SQS and Lambda
- Orchestrated workflow using AWS Step Functions with retrieval-augmented generation
- Streaming integration using Amazon Kinesis and real-time analytics

1.1.2 **A financial services company wants to implement a generative AI solution for summarizing lengthy financial documents and extracting key financial metrics. The solution must have high accuracy for financial terminology and maintain data privacy.**

**Which foundation model approach in Amazon Bedrock would be most appropriate?**

- Use Claude 3 Sonnet with few-shot prompting for financial summarization
- Use Llama 3 with RAG (Retrieval-Augmented Generation) for document processing
- Fine-tune Titan Text model on proprietary financial documents
- Use Amazon Bedrock Knowledge Bases with Anthropic Claude and company-specific financial data


1.1.3 **A customer service department wants to implement a generative AI solution to draft email responses to customer inquiries. The business requirements include maintaining brand voice, ensuring factual accuracy, and reducing response time.**

**Which architectural component would be LEAST important for meeting these requirements?**

- A content filtering system to detect and prevent inappropriate responses
- A human-in-the-loop review process for complex or sensitive inquiries
- A knowledge base integration to provide up-to-date product information
- A multi-modal foundation model that can process images and text

1.1.4 **Your company is developing a generative AI application that will process sensitive internal documents to answer employee questions. The application must remain within your company's network boundary and cannot send data to external services.**

**Which deployment strategy would best meet these requirements?**

- Use Amazon Bedrock with a private VPC endpoint and data encryption
- Deploy foundation models on Amazon SageMaker with model parallelism
- Implement Amazon Bedrock Knowledge Bases with cross-region replication
- Use AWS Inferentia instances with locally deployed open-source models

## Task 1.2 

1.2.1 **Your team needs to adapt a foundation model for a specialized medical domain without extensive retraining.**

**Which parameter-efficient adaptation technique would be most appropriate for this scenario?**

- Full model fine-tuning with gradient descent
- Low-Rank Adaptation (LoRA)
- Prompt engineering with few-shot examples
- Creating a new model from scratch using transfer learning

1.2.2 **Your company is deploying a critical AI application that uses Amazon Bedrock models. The application must remain operational even if a single AWS region experiences an outage.**

**Which approach would provide the most effective cross-region resilience?**

- Deploy the application in a single region with multiple Availability Zones
- Implement Amazon Bedrock Cross-Region Inference with automatic failover
- Create separate, independent deployments in multiple regions with DNS-based routing
- Use AWS Global Accelerator to route requests to the nearest available region

### KNOWLEDGE CHECK
Question
01/05
**In a system using AWS Step Functions for generative AI (GenAI) operations, which pattern BEST implements a circuit breaker for handling model failures?**

- Implementing exponential backoff within AWS Lambda functions
- Using Step Functions Choice states with error counts and timeouts
- Adding retries to Amazon API Gateway endpoints
- Implementing Amazon DynamoDB based error tracking
- Using Amazon EventBridge to trigger alternative workflows

02/05

**Which methods effectively support graceful degradation in a generative AI (GenAI) system? (Select TWO.)**

- Implementing fallback to more basic models with reduced capabilities
- Maintaining multiple model versions with capability tiering
- Returning cached responses for all requests during degradation
- Complete system shutdown during any model unavailability
- Switching to rule-based responses for all queries during degradation
- Routing all traffic to a single, generic model regardless of the original use case

03/05

**Which architectural patterns support dynamic model switching without code deployment when using Amazon Bedrock? (Select TWO.)**

- Using AWS AppConfig to store model configurations and AWS Lambda environment variables
- Implementing direct API calls with hardcoded model endpoints
- Using Amazon API Gateway with AWS Lambda integration and feature flags in AWS AppConfig
- Storing model selections in Amazon DynamoDB with direct access from applications
- Use AWS CloudFormation for model configuration

04/05

**Which combination of AWS services BEST supports Regional resilience for a mission-critical generative AI (GenAI) application?**

- Amazon Bedrock with multi-Region deployment and Amazon Route 53 health checks
- Cross-Region read replicas without active failover
- Amazon CloudFront distribution without Regional model deployment
- AWS Global Accelerator without Regional model redundancy

05/05

**When evaluating text summarization models in Amazon Bedrock for a production system that requires consistent performance, which evaluation result would indicate the BEST model selection?**
- The model demonstrates moderate fluctuation with a robustness score of 25.3%, producing notably different summaries when input text contains minor variations.
- The model shows minimal variation with a robustness score of 8.5%, maintaining consistent summary quality even when input text has slight variations or noise.
- The model exhibits high sensitivity with a robustness score of 42.1%, generating significantly different summaries for slightly modified input text.
- The model shows intermediate variation with a robustness score of 15.7%, with summary quality varying based on input text modifications.
- The model displays substantial inconsistency with a robustness score of 33.9%, producing unpredictable summaries when input text is slightly altered.


## Task 1.3


1.3.1. You're preparing to send a request to the Claude model in Amazon Bedrock.

**Which of the following JSON request formats is correctly structured?**

1. 
```json
{

 "prompt": "Summarize the benefits of cloud computing",

 "max_tokens_to_sample": 500,

 "temperature": 0.7

}
```

2. 
```json
{

 "anthropic_version": "bedrock-2023-05-31",

 "max_tokens": 500,

 "messages": [

   {

     "role": "user",

     "content": "Summarize the benefits of cloud computing"

   }

 ]

}
```

3. 
```json
{

 "inputs": "Summarize the benefits of cloud computing",

 "parameters": {

   "max_length": 500,

   "temperature": 0.7

 }

}
```

4. 
```json
{

 "text": "Summarize the benefits of cloud computing",

 "model": "claude-v2",

 "max_tokens": 500

}
```


### 1.3 Knowledge Check


### Knowledge Check

01/05
**Which approach is the MOST operationally efficient for implementing a data validation workflow for foundation model data?**

- Use AWS Glue Data Quality for automated validation and connect it with Amazon CloudWatch for monitoring.
- Use AWS Lambda functions to validate each data field individually.
- Implement all validations in Amazon SageMaker Data Wrangler.
- Write custom validation scripts in Amazon EC2 instances.

02/05

**You're designing a system that prepares customer interaction data from multiple sources for foundation model analysis.**

**Which approaches are the MOST operationally efficient for ensuring proper data formatting and validation? (Select TWO.)**

- Create a standardized formatting layer using AWS Lambda with caching and implement model-specific validation.
- Store all data in Amazon DynamoDB with JSON formatting.
- Implement centralized formatting with Amazon SageMaker AI endpoints and proper error handling.
- Use Amazon Aurora to format and validate all inputs.
- Process all inputs through Amazon EMR for formatting and validation.


03/05
Your team needs to process large volumes of multimodal data (text, images, and audio) for a foundation model.

**Which approach is MOST operationally efficient for handling this complex data processing?**

- Use separate AWS Lambda functions for each data type and coordinate with AWS Step Functions.
- Process all data types using Amazon EMR clusters.
- Implement Amazon SageMaker Processing jobs with custom containers for each modality.
- Use the multimodal capabilities of  Amazon Bedrock with Amazon SageMaker Processing for pre/post-processing.
- Store all data in Amazon S3 and process using Amazon Athena.


04/05

**You're developing a system to analyze customer service interactions that include both audio recordings and text chat logs.**

**Which approach is the MOST operationally efficient for extracting insights from both data sources?**

- Process all data through Amazon Comprehend only.
- Use Amazon Transcribe for audio and Amazon Comprehend for both transcribed and text data.
- Process audio with Amazon Polly and text with AWS Lambda functions.
- Use Amazon Translate for all text processing.
- Implement custom speech-to-text processing in Amazon EC2 instances.


05/05

**You're implementing a customer service system that needs to handle both one-time queries and multi-turn conversations with memory.**

**Which Amazon Bedrock API approaches can meet these requirements? (Select TWO.)**

- Use invokeModel for one-time queries and handle response parsing appropriately.
- Implement createModelCustomization for conversation handling.
- Use invokeModelWithResponseStream for multi-turn conversations with state management.
- Implement getModelCustomizations for all interactions.
- Use deleteModelCustomization for cleaning up conversations.

## Task 1.4: Design and implement vector store solutions.



### Knowledge Check

01/05

**You're designing a vector database architecture for a large-scale foundation model (FM) project that requires efficient semantic retrieval.**

**Which approaches are the MOST operationally efficient for this use case? (Select TWO.)**

- Store vectors directly in Amazon Neptune Analytics with graph relationships.
- Implement Amazon OpenSearch with Neural Plugin for topic-based segmentation and Amazon Bedrock integration.
- Use Amazon RDS PostgreSQL with pgvector extension and Amazon S3 for hybrid storage.
- Deploy Amazon DynamoDB with vector capabilities for metadata and embeddings management.
- Store all vectors in Amazon Timestream for time-series analysis.


02/05

**You're building a document processing system for a legal firm that needs to maintain strict versioning, authorship tracking, and domain classification for millions of legal documents used with foundation models (FMs). The system must support real-time updates and efficient retrieval based on document attributes.**

**Which approach is the MOST operationally efficient?**

- Store all metadata in separate Amazon DynamoDB tables with global secondary indexes (GSIs) for each attribute type.
- Use Amazon S3 Object Metadata with custom attributes and implement a tagging system for classification. 
- Implement a separate Amazon RDS instance with JSON columns for flexible metadata storage.
- Store metadata directly within document content using standardized headers.
- Use Amazon Kendra with custom metadata fields for extraction and storage.


03/05

**A research organization needs to optimize its vector database to handle semantic search across multiple scientific domains (biology, chemistry, physics) with different embedding models and query patterns. Each domain contains millions of vectors that need sub-second query response times.**

**Which approaches are MOST efficient for improving search performance? (Select TWO.)**

- Implement a single, large OpenSearch index with composite fields for all domains.
- Use OpenSearch sharding strategies based on domain-specific criteria with custom routing.
- Store all vectors in Amazon Aurora with partitioned tables per domain.
- Implement multi-index approaches in OpenSearch with specialized mappings for each domain.
- Use Amazon Redshift with materialized views for vector storage and querying.


04/05

**A company needs to maintain vector stores for its foundation models (FMs) with real-time updates, automated synchronization, and scheduled refreshes. The solution must ensure data consistency across different vector stores and knowledge bases.**

**Which approaches are most operationally efficient? (Select TWO.)**

- Implement scheduled full refreshes using AWS Batch.
- Use incremental update mechanisms with change detection in Amazon Bedrock Knowledge Bases.
- Deploy manual update processes with version control.
- Create automated synchronization workflows using AWS Glue with real-time triggers.
- Use Amazon Kinesis for all updates.


05/05

**A multinational corporation needs to integrate its document management systems, knowledge bases, and internal wikis into a unified system for its foundation models (FMs). The solution must maintain data freshness and support both batch and real-time updates.**

**Which approaches are MOST operationally efficient? (Select TWO.)**

- Use AWS Lake Formation for centralized access control and data integration.
- Implement individual AWS Lambda functions for each data source.
- Create a unified data pipeline using AWS Glue with incremental processing.
- Store all documents in Amazon OpenSearch for unified search.
- Use Amazon Managed Streaming for Apache Kafka (Amazon MSK) for all data integration.




## 1.5 Design Retrival Mechanisms for FM Augmentation



1.5.1. Your legal document retrieval system needs to process complex multi-part queries about case law and regulations, but users often submit broad questions that don't retrieve specific enough information.

**Which query handling approach would be MOST effective for improving retrieval quality in this scenario?**

- Implement a Lambda function that simplifies user queries by removing all but the most common legal terms
- Create a Step Function workflow that decomposes complex legal queries into multiple sub-queries, executes them in parallel, and aggregates the results
- Use Amazon Bedrock to expand user queries with relevant legal terminology and then perform vector search on the expanded queries
- Configure a static list of legal terms to append to all user queries regardless of content

1.5.2. Your team is developing a system that needs to integrate multiple foundation models with various vector stores for different use cases. Developers are struggling with inconsistent integration patterns.

**Which approach would create the MOST consistent access mechanism for seamless integration?**

- Create custom API endpoints for each foundation model and vector store combination
- Implement a standardized function calling interface that abstracts vector search operations across all foundation models
- Have each team develop their own integration approach based on their specific requirements
- Store all vector embeddings in a single database regardless of their source or purpose


### Knowledge Check

01/05

A publishing company needs to process diverse content types (books, articles, technical documentation) for its foundation model. The content varies greatly in structure and length, requiring adaptive chunking strategies.

**Which approaches are MOST operationally efficient? (Select TWO.)**


 - Use fixed-size chunking with Lambda functions for all document types.
 - Implement Amazon Bedrock's chunking capabilities with content-aware segmentation.
 - Store whole documents without segmentation.
 - Deploy custom hierarchical chunking based on document structure with AWS Lambda.
 - Use character-count-based splitting for all documents.

02/05

A financial services firm needs to implement embeddings for its market analysis documents, requiring high accuracy and domain-specific understanding.

**Which approach provides the MOST efficient solution?**

 - Use Amazon Titan Embeddings with domain-tuned parameters.
 - Generate embeddings using custom machine learning (ML) models on Amazon EC2.
 - Use pre-trained general-purpose embeddings.
 - Implement custom embedding generation in Amazon SageMaker AI.

03/05

A healthcare organization needs to implement semantic search across medical documents, clinical trials, and patient records.

**Which approach provides the MOST operationally efficient vector search solution?**

- Deploy Amazon OpenSearch Service with Neural Search plugin and vector search capabilities.
- Use Amazon Aurora with pgvector for all vector operations.
- Implement custom vector search using Amazon DynamoDB.
- Store vectors in Amazon RDS and implement search logic in AWS Lambda.
- Use Amazon Elasticsearch without vector search features.


04/05

A legal firm needs to implement a hybrid search system that must handle both exact citation matches and semantic similarity for case law documents. The system processes over 10,000 searches daily and must maintain sub-second response times while ensuring result accuracy.

**Which approach is MOST operationally efficient?**

 - Implement OpenSearch with hybrid retrieval and Amazon Bedrock reranker for result optimization.
 - Deploy separate Elasticsearch clusters for keyword and semantic search.
 - Build a custom search solution using Amazon RDS and AWS Lambda functions.
 - Implement pure vector search using pgvector in Amazon Aurora.

 05/05

 A research organization needs to process complex queries across scientific papers and experimental data.

**Which approach provides the MOST operationally efficient query handling system?**

 - Use Amazon Bedrock for query expansion with AWS Step Functions orchestration.
 - Implement all query processing in AWS Lambda functions.
 - Process queries directly without transformation.
 - Use Amazon Comprehend for query analysis.

 05/05

A research organization needs to process complex queries across scientific papers and experimental data.

**Which approach provides the MOST operationally efficient query handling system?**

 - Use Amazon Bedrock for query expansion with AWS Step Functions orchestration.
 - Implement all query processing in AWS Lambda functions.
 - Process queries directly without transformation.
 - Use Amazon Comprehend for query analysis.


# Task 2.3

### Knowledge Check

01/05

**A GenAI developer needs to implement real-time trading signal generation using foundation models (FMs).**

**Which combination of services provides the MOST efficient architecture for this requirement?**

 a. Amazon API Gateway with REST APIs and AWS Lambda functions for synchronous processing\
 b. WebSocket APIs with Amazon EventBridge and Amazon Kinesis Data Streams\
 c. HTTP APIs with AWS Step Functions for orchestration\
 d. Amazon Simple Queue Service (Amazon SQS) with periodic AWS Lambda polling


02/05

**A global financial services firm needs to deploy foundation models (FMs) for trading analytics while meeting data residency requirements and maintaining sub-millisecond latency.**

**Which combination of deployment approaches is MOST appropriate?**

 - Deploy all models to the cloud and access through virtual private network (VPN) connections.

 - Use AWS Outposts for data residency and AWS Wavelength for low-latency processing.

 - Replicate all data to the cloud for centralized processing.

 - Use Lambda@Edge for model inference and caching.


03/05


**A financial services firm needs to implement secure access controls for their AI model APIs while maintaining sub-millisecond latency.**

**Which security implementation is MOST appropriate?**

 - Amazon API Gateway with AWS Lambda authorizers
 - AWS Identity and Access Management (IAM) roles with resource-based policies and virtual private cloud (VPC) endpoints
 - Amazon Cognito User Pools with JSON web token (JWT) validation
 - Custom authorizers with database lookups


 04/05

**When implementing event-driven AI integration, what is the MOST effective approach for handling model inference errors while maintaining system reliability?**

 - Implement synchronous retries with exponential backoff.
 - Use dead-letter queues with automated fallback models.
 - Log errors and continue processing.
 - Route all errors to human review.

05/05

**A GenAI developer needs to implement real-time model monitoring across distributed environments.**

**Which combination of services provides the MOST comprehensive solution?**

 - Amazon CloudWatch with custom metrics and Amazon EventBridge rules
 - Prometheus with Grafana dashboards
 - Custom logging with Amazon OpenSearch Service
 - Amazon SageMaker Model Monitor






# ANSWER KEY

## Task 1.1 

#### 1.1.1 
An **orchestrated workflow using AWS Step Functions with retrieval-augmented generation** would be the most appropriate integration pattern for this customer support chatbot because:
- Step Functions can orchestrate the complex workflow needed for a chatbot that must maintain context, access external systems, and make decisions about escalation
- Retrieval-augmented generation (RAG) allows the chatbot to pull relevant information from the CRM system to provide accurate, contextual responses
- This pattern supports maintaining conversation state and context across multiple interactions
- Step Functions can handle the logic for determining when to escalate to human agents
- The pattern provides visibility into the execution flow and error handling capabilities

Synchronous API integration would be too simplistic for this complex use case involving multiple systems and decision points. Asynchronous batch processing isn't suitable for real-time conversational interfaces. Streaming integration is more appropriate for continuous data processing rather than conversational interfaces.

#### 1.1.2 
**Using Amazon Bedrock Knowledge Bases with Anthropic Claude and company-specific financial data** would be the most appropriate solution for this scenario because:

- Knowledge Bases allow the company to integrate their proprietary financial documents and terminology without exposing sensitive data for fine-tuning
- The solution maintains data privacy as the documents stay within the company's AWS environment
- Claude models have demonstrated strong capabilities in understanding and summarizing complex documents
- Knowledge Bases provide retrieval capabilities that improve accuracy for domain-specific information like financial metrics
- This approach doesn't require extensive model fine-tuning, which would be more complex and require more resources

While Claude 3 Sonnet with few-shot prompting could work for simple cases, it wouldn't have the deep integration with company-specific financial documents. Llama 3 with RAG is conceptually similar to Knowledge Bases but would require more custom implementation. Fine-tuning Titan Text would require sharing sensitive financial data for the training process, raising privacy concerns.

#### 1.1.3 
A multi-modal foundation model that can process images and text would be the LEAST important component for meeting the stated requirements because:

- The scenario specifically mentions drafting email responses to customer inquiries, which primarily involves text processing
- None of the stated requirements (maintaining brand voice, ensuring factual accuracy, reducing response time) necessitate image processing capabilities
- Adding multi-modal capabilities would increase complexity without directly addressing the core business needs

The other components are all important for meeting the requirements:
- Content filtering helps maintain appropriate brand voice and prevent problematic responses
- Human-in-the-loop review ensures factual accuracy for complex cases and provides quality control
- Knowledge base integration is crucial for ensuring factual accuracy about products and services

#### 1.1.4 
Using Amazon Bedrock with a private VPC endpoint and data encryption would best meet the requirements because:

- Amazon Bedrock can be accessed via VPC endpoints, keeping all traffic within your AWS network and not traversing the public internet
- This approach maintains data privacy while leveraging powerful foundation models without having to manage model infrastructure
- Data encryption can be implemented both in transit and at rest to protect sensitive documents
- It provides a balance between security requirements and implementation complexity

Deploying foundation models on SageMaker with model parallelism would be unnecessarily complex and resource-intensive when managed services like Bedrock are available. Amazon Bedrock Knowledge Bases with cross-region replication doesn't specifically address the network boundary requirement. Using AWS Inferentia with locally deployed models would require significant expertise in model deployment and optimization, which is likely beyond the scope needed for this use case.

## Task 1.2

#### 1.2.1
**Low-Rank Adaptation (LoRA)** would be most appropriate for adapting a foundation model to a specialized medical domain because:

- LoRA is a parameter-efficient fine-tuning technique that adds small, trainable rank decomposition matrices to existing model weights
- It requires significantly less computational resources than full model fine-tuning
- LoRA preserves most of the foundation model's general knowledge while adapting it to the medical domain
- It results in smaller adapter modules that can be easily swapped or combined
- LoRA typically requires less training data than full fine-tuning, which is valuable for specialized domains with limited data

Full model fine-tuning requires extensive computational resources and may lead to catastrophic forgetting. Prompt engineering with few-shot examples might not be sufficient for deep domain adaptation in medicine. Creating a new model from scratch would be unnecessary and resource-intensive when adaptation techniques like LoRA exist.

#### 1.2.2
Creating separate, independent deployments in multiple regions with DNS-based routing would provide the most effective cross-region resilience because:

- It ensures complete isolation between regional deployments, so a failure in one region doesn't affect others
- DNS-based routing (like Amazon Route 53) can detect regional outages and automatically direct traffic to healthy regions
- Each regional deployment can operate independently with its own resources and foundation model endpoints
- This approach provides true disaster recovery capabilities in case of a complete regional outage

Deploying in a single region with multiple AZs wouldn't protect against region-wide failures. Amazon Bedrock Cross-Region Inference helps with model availability but doesn't address the resilience of the entire application stack. AWS Global Accelerator improves network performance but requires healthy endpoints in multiple regions to route to, which this option doesn't specify.

## Task 1.3


1.3.2. You're developing a chatbot application using Amazon Bedrock.

**Which conversation formatting approach would produce the most consistent responses from the Claude model?**
1. Concatenate all previous messages with the current query
2. Send only the most recent user query to the model
3. Use the messages array with alternating user and assistant roles
4. Convert the conversation to a bullet-point summary before sending


### KNOWLEDGE CHECK
Question
01/05:  ✔️(b)\
❌ (a): Exponential backoff is a retry mechanism that does not maintain system-wide failure state or implement true circuit breaking.\
✔️ (b): Step Functions Choice states provide state management across executions, enabling proper threshold monitoring and automatic circuit breaking based on error patterns.\
❌ (c): API Gateway retries operate at the HTTP level without maintaining state between requests or implementing circuit breaker patterns.\
❌ (d): DynamoDB based tracking adds unnecessary latency and complexity while lacking immediate circuit breaking capabilities.\
❌ (e): EventBridge offers reactive responses rather than proactive circuit breaking and lack proper state management.

02/05: ✔️(a) ✔️(b)\
✔️ (a): Fallback to more basic models ensures service continuity while gracefully reducing capabilities based on system conditions.\
✔️ (b): Multiple model versions with capability tiering enables systematic degradation while maintaining critical functionalities.\
❌ (c): Cached responses cannot handle new queries and might return stale or inappropriate information.\
❌ (d): Complete shutdown violates graceful degradation principles and unnecessarily disrupts service.\
❌ (e): Rule-based responses eliminate AI capabilities and cannot handle complex queries effectively.\
❌ (f): Generic model routing compromises specialized use cases and reduces overall service quality.

03/05: ✔️(a) ✔️(c)\
✔️ (a): AWS AppConfig enables runtime configuration changes and model switching without code deployments.\
❌ (b): Hardcoded endpoints require code changes for any model modifications, defeating dynamic switching.\
✔️ (c): This combination enables dynamic routing and configuration updates while supporting A/B testing.\
❌ (d): Direct DynamoDB access adds complexity and does not provide proper configuration management.\
❌ (e): Using AWS CloudFormation for model configuration requires infrastructure updates and does not support runtime changes.

04/05: ✔️(a)\
✔️ (a): This approach combines Regional model availability with intelligent routing and automated health checking for maximum resilience.\
❌ (b): Read replicas without active failover do not provide immediate resilience during Regional issues.\
❌ (c): CloudFront alone does not address model availability in different Regions.\
❌ (d): Network optimization without model redundancy does not provide true Regional resilience.

05/05: ✔️(a)\
✔️ (a): A robustness score of 8.5% indicates strong model stability. This means that when the input text is slightly modified (perturbed), the model continues to produce consistently similar summaries. This level of consistency is crucial for production systems where reliability and predictability are essential.\
❌ (b): A 25.3% variation suggests the model is too sensitive to input changes. In production, this could lead to inconsistent user experiences where similar inputs produce noticeably different summaries.\
❌ (c): At 42.1%, this high variation indicates poor model stability. Such significant changes in output based on minor input modifications would make the model unreliable for production use.\
❌ (d): A 15.7% variation still indicates too much inconsistency for a production system requiring high reliability.\
❌ (e): A 33.9% variation demonstrates that the model is highly unstable. This level of inconsistency would make the model unsuitable for production deployment where consistent performance is required.

## Task 1.3

1.3.1. **Which of the following JSON request formats is correctly structured?**

Answer:2   

❌ 1. This is the legacy Anthropic Completion API. Not supported in Amazon Bedrock. Bedrock requires the Messages API, not prompt

✔️ 2. This is the correct format for the Messages API

❌ 3. This resembles Hugging Face–style inference payloads. Not valid for Claude in Bedrock

❌ 4. Bedrock does not accept model selection in the payload. Model is chosen via the API call / model ARN, not JSON body, `text` is not a valid Claude field

1.3.3. **Which conversation formatting approach would produce the most consistent responses from the Claude model?**

Answer:3
Using the messages array with alternating user and assistant roles would produce the most consistent responses because:

- This format maintains the full conversation context in a structured way that the model can understand
- The role-based format (user/assistant) helps the model understand the conversation flow and maintain appropriate responses
- It's the recommended approach in the Claude API documentation for conversation-based applications
- This approach preserves the natural back-and-forth structure of a conversation

Concatenating all messages loses the structured turn-taking information. Sending only the most recent query loses important conversation context. Converting to a bullet-point summary would lose the natural conversation flow.

### 1.3 Knowledge Check

01/05

Answer: 1

✔️ (a): This approach provides automated validation with built-in monitoring, reducing operational overhead and enabling quick issue detection.

❌ (b): This approach is resource-intensive and harder to maintain, requiring separate functions for each validation type.

❌ (c): Although visual tools are helpful, using only SageMaker Data Wrangler for all validations creates bottlenecks and lacks automation capabilities.

❌ (d): This approach requires managing infrastructure and lacks the automated scaling and integration features of managed services.


02/05

Answer: 1, 3

✔️ (a): This approach provides flexible formatting with efficient resource usage through caching while enabling custom validation logic for different model requirements. Lambda's serverless nature ensures cost-effective scaling and maintenance.

❌ (b): Although DynamoDB supports JSON, using it primarily for formatting adds unnecessary database operational overhead and complexity. It's not designed specifically for model input formatting.

✔️ (c): This method uses SageMaker's built-in capabilities for structured data preparation, ensuring consistent formatting across different model endpoints while maintaining proper error handling and validation.

❌ (d): Using Aurora for formatting adds unnecessary database management overhead and is not optimized for model input preparation. It lacks specific features needed for foundation model data formatting.

❌ (e): Processing all inputs through Amazon EMR introduces unnecessary complexity and operational overhead. It is better suited for large-scale data processing tasks rather than input formatting.


03/05

Answer: ✔️ (d)

❌ (a): Although flexible, this approach can lead to complex orchestration and potential synchronization issues between modalities.

❌ (b): Amazon EMR is powerful but introduces unnecessary cluster management overhead for this use case.

❌ (c): This approach works but requires more management of custom containers and might not be as efficient as using specialized multimodal services.

✔️ (d): This uses Amazon Bedrock's optimized multimodal processing with SageMaker's flexible pre/post-processing, providing an efficient, scalable solution with minimal operational overhead.

❌ (e): Athena is designed for query processing, not complex multimodal data processing for machine learning (ML) models.


04/05

Answer: ✔️ (b)

❌ (a): Amazon Comprehend is designed for text analysis and entity extraction but cannot process audio data directly. It lacks speech-to-text capabilities, making it insufficient for audio processing.

✔️ (b): Amazon Transcribe handles the unique challenges of speech recognition (such as background noise, multiple speakers, and temporal aspects), and Amazon Comprehend excels at understanding the meaning and entities within the text.

❌ (c): Amazon Polly is for text-to-speech synthesis, not speech recognition, making it inappropriate for this use case.

❌ (d): Amazon Translate is designed for language translation, not speech recognition or text analysis.

❌ (e): Custom implementations require significant development and maintenance effort, lacking the advanced features and continuous improvements of managed services.

05/05

Answer: ✔️ (a), ✔️ (d)

✔️ (a): invokeModel is the correct method for stateless interactions, providing efficient processing for one-time queries. It includes built-in response handling and error management capabilities.

❌ (b): createModelCustomization is an Amazon Bedrock API used for model customization, not conversation handling.

✔️ (c): invokeModelWithResponseStream is ideal for multi-turn conversations, allowing efficient streaming of responses while managing conversation state at the application layer.

❌ (d): getModelCustomizations is an Amazon Bedrock API that retrieves model customization information. It does not process conversations or queries.

❌ (e): deleteModelCustomization removes model customizations. It does not process conversations or queries.

## Task 1.4

### Knowlege Check

01/05


Answer: ✔️ (b), ✔️ (d)

❌ (a): Although Neptune Analytics supports vector storage, it's optimized for graph relationships and adds unnecessary complexity when primary needs are semantic search and retrieval. Its strength lies in relationship analysis rather than pure vector operations and would require additional management overhead.

✔️ (b): OpenSearch with Neural Plugin provides efficient semantic search and integrates well with Amazon Bedrock, offering scalable topic-based segmentation. The Neural Plugin specifically optimizes vector search operations with approximate k-nearest neighbors (k-NN) search, dynamic index updates, and efficient vector compression, making it ideal for large-scale semantic retrieval with minimal operational overhead.

❌ (c): Amazon RDS for PostgreSQL with pgvector and Amazon S3 can create an effective hybrid architecture, but it requires managing the PostgreSQL extension, handling database scaling, and implementing custom logic for efficient vector operations. Although it is viable, it introduces more operational complexity compared to managed vector search solutions and might face performance challenges at larger scales.

✔️  (d): DynamoDB with vector support efficiently manages metadata and embeddings, complementing the semantic search capabilities. It provides consistent performance at scale and supports atomic updates for real-time embedding management while maintaining millisecond latency for retrieval operations. The managed service aspects reduce operational overhead significantly.

❌ (e): Timestream is optimized for time-series data, not vector storage or semantic search. Its data model and query optimization are specifically designed for temporal data patterns, making it inappropriate for vector operations and semantic similarity searches. Using it would require significant custom development and would not use its core capabilities.

02/05

**Answer:** ✔️ (b)

❌ (a): Although DynamoDB with GSIs provides fast lookups, maintaining separate tables for metadata creates unnecessary complexity and potential consistency issues across tables. The GSI limits (20 per table) might also restrict flexibility for extensive metadata attributes needed in legal documents.

✔️ (b): Amazon S3 Object Metadata with custom attributes provides efficient, scalable metadata management directly tied to documents, and tagging enables flexible classification. This approach supports up to 10 tags for each object and unlimited user-defined metadata fields, perfect for legal document attributes. The metadata is automatically versioned with the objects, providing built-in audit trails and version control with minimal operational overhead.

❌ (c): An Amazon RDS instance with JSON columns offers flexibility but introduces additional operational complexity, backup management, and scaling concerns. Although it supports complex queries, it requires maintaining a separate database infrastructure and handling synchronization with document storage.

❌ (d): Embedding metadata within documents complicates retrieval and updates, requiring document parsing for metadata access. This approach would significantly impact performance for metadata-based searches and require reprocessing entire documents for metadata updates.

❌ (e): Although Amazon Kendra excels at intelligent search, using it primarily for metadata management is not efficient. Its metadata extraction capabilities are better suited as a complement to, rather than a replacement for, a dedicated metadata management system.

03/05

**Answer:** ✔️ (b), ✔️ (d)

❌ (a): A single large index would create contention across domains and limit the ability to optimize for domain-specific vector dimensions. This approach would also make index updates more challenging and impact query performance as the index grows beyond millions of vectors.

✔️ (b): Domain-based sharding with custom routing ensures optimal resource utilization and query performance. This approach allows for different shard counts and replica configurations for each domain, and custom routing ensures queries hit only relevant shards. It also enables independent scaling and maintenance for each scientific domain, which is crucial for handling different update patterns.

❌ (c): Although Aurora supports partitioning, it lacks specialized vector operations and optimization techniques like Hierarchical Navigable Small Worlds HNSW indexes. This would result in full table scans for similarity searches and poor performance for high-dimensional vectors common in scientific domains.

✔️ (d): Multi-index approaches with specialized mappings allow for optimized configurations for each domain (different vector dimensions, similarity algorithms, and index settings). This provides better resource isolation and enables domain-specific tuning of refresh intervals, merge policies, and similarity thresholds for maximum performance.

❌ (e): Although Amazon Redshift is powerful for analytical queries, it lacks native vector similarity search capabilities. Using materialized views would still require custom implementation of vector operations and would not provide the performance optimizations needed for sub-second similarity searches.


04/05

**Answer:** ✔️ (b), ✔️ (d)

❌ (a): Full refreshes are resource-intensive and do not efficiently handle incremental changes. This approach can lead to unnecessary processing and potential data inconsistencies.

✔️ (b): Amazon Bedrock Knowledge Bases provides efficient incremental updates with built-in change detection. This ensures vector stores remain current while minimizing processing overhead. The service handles version management and consistency automatically.

❌ (c): Manual processes do not scale and cannot maintain consistency across large-scale vector stores. This approach increases risk of errors and data inconsistencies.

✔️ (d): AWS Glue with real-time triggers enables efficient automated synchronization. It maintains data lineage, handles incremental updates automatically, and ensures consistent processing across different data sources. The built-in scheduling and monitoring capabilities make it ideal for maintaining vector store freshness.

❌ (e): Although Kinesis is good for streaming data, it is not optimized for maintaining vector stores and knowledge bases. It lacks the specific features needed for efficient vector store maintenance.


05/05

**Answer:** ✔️ (a), ✔️ (c)

✔️ (a): Lake Formation provides centralized governance and integration capabilities while maintaining fine-grained access control. It efficiently handles different data sources through blueprint-based ingestion and maintains data lineage automatically. This reduces operational overhead while ensuring consistent data handling across sources.

❌ (b): Individual Lambda functions create unnecessary complexity and make it difficult to maintain consistent processing patterns. This approach does not scale well for enterprise-wide integration.

✔️ (c): AWS Glue with incremental processing provides efficient extract, transform, and load (ETL) capabilities with built-in support for change detection and delta updates. It includes native connectors for various data sources and maintains metadata catalogs automatically, making it ideal for large-scale document integration.

❌ (d): Although OpenSearch is powerful for search, using it as the primary integration point creates unnecessary indexing overhead and does not efficiently handle different data source requirements.

❌ (e): Amazon MSK is optimized for real-time streaming data, not document-based integration. It would add unnecessary complexity for document management system integration.


## Task 1.5

1.5.1 
**Answer:** ✔️ (c)

Using Amazon Bedrock for query expansion leverages the foundation model's understanding of legal terminology to enhance the original query with relevant domain-specific terms. This approach helps bridge the gap between how users naturally phrase questions and the specialized terminology in legal documents, improving retrieval effectiveness without requiring complex decomposition logic.

1.5.2
**Answer:** ✔️ (b)

Implementing a standardized function calling interface creates consistency by abstracting vector search operations behind a common API. This approach allows different foundation models to interact with various vector stores through the same patterns, simplifying development and maintenance while ensuring seamless integration across the system.


### Knowledge Check

01/05

**Answer:** ✔️ (b), ✔️ (d)

❌ (a): Fixed-size chunking ignores document structure and can break contextual relationships. This approach lacks the flexibility needed for diverse content types.

✔️ (b): Bedrock's chunking capabilities provide intelligent segmentation that respects content boundaries and semantic units. It automatically handles different content types while maintaining context relationships.

❌ (c): Whole documents would exceed context windows and prevent efficient retrieval. This approach does not support granular context management.

✔️ (d): Custom hierarchical chunking preserves document structure and relationships while enabling flexible segment sizes. This approach optimizes retrieval by maintaining natural content boundaries.

❌ (e): Character-count splitting can break semantic units and ignore document structure, leading to poor context preservation.


02/05

**Answer:** ✔️ (a)

✔️ (a): Amazon Titan Embeddings provides optimized performance for domain-specific content, with built-in parameter tuning capabilities and efficient batch processing. It integrates natively with other AWS services and scales automatically.

❌ (b): Custom ML models require significant infrastructure management and ongoing maintenance. This approach adds unnecessary complexity without guaranteeing better performance.

❌ (c): General-purpose embeddings lack domain-specific understanding crucial for financial analysis, resulting in reduced accuracy and relevance.

❌ (d): SageMaker AI implementation requires managing endpoints and model deployment, adding operational overhead without clear benefits.


03/05

**Answer:** ✔️ (a)

✔️ (a): OpenSearch Service with Neural Search plugin provides optimized vector operations, automatic scaling, and integrated monitoring. It offers built-in support for approximate k-nearest neighbors (k-NN) search and efficient vector indexing crucial for healthcare data volumes.

❌ (b): Although Aurora with pgvector is viable, it requires more manual management and might face scaling challenges with large vector datasets in healthcare contexts.

❌ (c): Custom implementation in DynamoDB would require building vector search functionality from scratch, adding unnecessary complexity and reducing performance.

❌ (d): This approach lacks efficient vector search capabilities and would require significant custom development for basic operations.

❌ (e): Standard Elasticsearch without vector capabilities would not support the semantic search requirements effectively.


04/05


**Answer:** ✔️ (a)

✔️ (a): This solution optimally combines OpenSearch hybrid retrieval (supporting both exact matches and vector search) with Amazon Bedrock reranker. It maintains sub-second latency through efficient indexing while improving result quality through intelligent reranking. The managed service aspects reduce operational overhead.

❌ (b): Running separate clusters increases infrastructure costs and adds complexity in result aggregation. This approach also introduces additional latency when combining results and complicates maintenance.

❌ (c): A custom solution would require significant development effort and ongoing maintenance. It would lack the optimized performance and features available in managed services for hybrid search.

❌ (d): Pure vector search in Aurora with pgvector would not efficiently handle exact citation matches needed for legal documents. It also lacks the performance optimizations needed for sub-second response times at scale.

05/05

**Answer:** ✔️ (a)

✔️ (a): This combination provides intelligent query expansion through Amazon Bedrock while using Step Functions to manage complex query transformation workflows. It offers reliable orchestration and efficient processing.

❌ (b): Pure Lambda implementation lacks sophisticated query understanding and proper workflow management.

❌ (c): Direct processing without transformation would miss relevant results and fail to handle complex queries effectively.

❌ (d): Amazon Comprehend lacks specific scientific domain understanding and query expansion capabilities needed for research content.

❌ (e): Custom implementation requires significant infrastructure management and lacks the benefits of managed AI services.


## Task 1.6

### Self Assessment

1.6.1 Correct Answer: A. Use **Amazon Bedrock Guardrails for tone enforcement and Amazon Bedrock Prompt Management with role definitions**

**Explanation:**

 - Amazon Bedrock Guardrails is specifically designed to enforce responsible AI guidelines, including maintaining professional tone and preventing unwanted topics (like competitor discussions).

 - Amazon Bedrock Prompt Management with role definitions allows you to create consistent model instruction frameworks that define how the model should behave.

**Why other options are incorrect:**

 - Option B: Amazon Comprehend can analyze sentiment but doesn't enforce tone or prevent specific topics.

 - Option C: Lambda and CloudWatch can monitor but don't provide built-in guardrails functionality.

 - Option D: S3 and Step Functions don't provide direct control over model behavior.


1.6.2 **Correct Answer: B. Implement Amazon Bedrock Prompt Management with approval workflows, store templates in S3, and use CloudTrail for auditing**

Explanation:

 - Amazon Bedrock Prompt Management provides built-in versioning and approval workflows.
 - S3 is ideal for template storage in a prompt management system.
 - CloudTrail provides comprehensive audit capabilities for all API calls.

**Why other options are incorrect:**

 - Option A: Lacks proper versioning and approval workflows.

 - Option C: DynamoDB isn't optimized for template versioning.

 - Option D: CodePipeline is for application deployment, not primarily for prompt management.

 ### Knowledge Check
 
 01/05

 ✔️ (a): Amazon Bedrock Prompt Flows provides native support for complex chains and conditional processing, with built-in state management and monitoring. It enables efficient pre/post processing integration while maintaining operational efficiency through managed services.

❌ (b): Although Step Functions handles workflows well, this approach requires significant custom development for prompt management and lacks native chain optimization features provided by Amazon Bedrock.

❌ (c): Using Amazon MSK adds unnecessary complexity and requires managing streaming infrastructure. This approach lacks specialized prompt chain features and increases operational overhead.

❌ (d): Amazon SQS based sequential processing does not efficiently handle conditional branching and lacks prompt-specific optimizations. This creates additional complexity in managing chain state.

02/05

 ✔️ (a): Amazon Bedrock Prompt Management provides enterprise-grade template management with built-in versioning, and CloudTrail enables comprehensive audit trails. This combination ensures governance compliance with minimal operational overhead.

❌ (b): Git-based version control requires significant custom development for approval workflows and lacks native audit capabilities needed for financial services.

 ✔️ (c): Amazon S3 provides reliable template storage with versioning, and CloudWatch Logs enables detailed access tracking and analysis. This approach offers scalable template management with proper governance controls.

❌ (d): DynamoDB implementation requires complex custom logic for approvals and versioning, increasing development and maintenance overhead without specialized prompt-management features.

❌ (e): Parameter store is not designed for managing complex prompt templates and lacks necessary governance features for enterprise-scale deployment.

03/05

 ✔️ (a): This combination provides scalable workflow management through Step Functions, reliable intent recognition with Amazon Comprehend, and efficient state management in DynamoDB. It ensures sub-second performance while maintaining operational efficiency.

❌ (b): Aurora adds unnecessary database complexity and potential latency. Lambda with Aurora requires managing connection pools and lacks native conversation flow management.

❌ (c): Custom WebSocket implementation requires complex state management and does not provide built-in language understanding capabilities needed for clarification workflows.

❌ (d): Amazon SQS based state management does not efficiently handle real-time conversation requirements and lacks built-in support for context maintenance.

04/05

 ✔️ (a): Amazon Bedrock Prompt Flows provides native support for CoT patterns with built-in performance monitoring and feedback loops. This enables systematic improvement of reasoning steps while maintaining operational efficiency.

❌ (b): Custom Lambda implementation requires complex prompt management and lacks built-in support for reasoning pattern optimization and feedback integration.

❌ (c): Multiple model deployment unnecessarily increases complexity and cost without improving reasoning quality. This approach fragments the reasoning process across services.

❌ (d): Amazon SQS based processing does not provide specialized support for reasoning patterns and adds unnecessary queuing complexity to the workflow.

05/05

 ✔️ (a): DynamoDB provides optimal performance through TTL for session cleanup, GSIs for efficient queries, and automatic scaling. Its single-digit millisecond latency and native JSON support enable efficient conversation tracking while maintaining operational efficiency through managed services.

❌ (b): Amazon RDS requires managing connection pools, scaling, and cache invalidation. This approach introduces unnecessary complexity and potential bottlenecks when handling high concurrent sessions.

❌ (c): ElastiCache with Redis backup adds operational overhead for cache management and requires complex synchronization between cache and persistent storage. This increases maintenance complexity without providing additional benefits.

❌ (d): Amazon DocumentDB introduces unnecessary operational complexity and cost for conversation storage. Although it supports document storage, it lacks the automatic scaling and integrated TTL features needed for efficient session management.

# Task 2

## Task 2.1

**Question 2.1.1**

 ✔️ (b): Configure the agent with session-based memory to maintain context within a conversation.

Amazon Bedrock Agents supports session-based memory that allows the agent to maintain context across multiple turns in a conversation. This is the most effective approach as it handles state management automatically within the agent framework. While DynamoDB could store conversation history, it would require additional implementation to integrate effectively. S3 storage would introduce latency and complexity. Regenerating context from scratch would lose the continuity needed for effective customer service interactions.

**Question 2.1.2**

 ✔️ (d): 
 
 Option D provides the most comprehensive approach to safeguarding AI workflows in a regulated environment. 
  Step Functions with stopping conditions ensures the process can be halted if specific criteria are met (confidence below threshold). IAM policies enforce resource boundaries preventing unauthorized access. Circuit breakers mitigate cascading failures by stopping processes when errors exceed acceptable thresholds. While option A includes some safeguards, it lacks the orchestration and circuit breaker capabilities. Option B focuses on monitoring rather than prevention. Option C addresses network security but not behavioral controls.


### Knowledge Check

01/05

**Answer:** (a), (d)

Circuit breakers provide automated, real-time protection by monitoring error rates and taking immediate action when thresholds are exceeded. This approach maintains system responsiveness while ensuring safety through automated intervention.

Although logging is important, weekly review creates a significant delay between issues occurring and being addressed. This approach does not provide real-time protection.

Requiring human approval for all actions introduces unnecessary delays and creates operational bottlenecks. This defeats the purpose of automation.

Graduated access levels with automated risk assessment provide dynamic protection that scales with operation criticality. This approach maintains efficiency while ensuring appropriate controls are in place.

02/05

**Answer:** (a)

- The supervisor-agent pattern provides structured coordination through a hierarchical approach. It enables clear control flows, efficient task distribution, and centralized oversight while maintaining agent specialization.

- A single agent approach creates a monolithic system that lacks flexibility and becomes difficult to maintain as complexity increases. It does not effectively handle multiple specialized functions.

- Direct peer-to-peer communication between agents can lead to coordination conflicts and becomes exponentially complex as the number of agents increases. This makes the system difficult to manage and debug.

- Using Lambda functions alone for coordination does not provide the sophisticated orchestration capabilities needed for complex multi-agent systems. It lacks built-in state management and coordination patterns.


03/05

**Answer:** (a), (c)

- ReAct patterns with validation steps ensure thorough analysis of each proposed solution before implementation. This approach maintains quality standards while enabling automated decision-making in pharmaceutical manufacturing.

- Continuous human oversight creates unnecessary delays and does not scale well for real-time quality control in high-volume production environments.

- CoT reasoning enables the system to evaluate multiple solution strategies and their potential impacts on product quality. This is crucial for pharmaceutical manufacturing where decisions can affect patient safety.

- Direct interventions without validation pose significant risks in pharmaceutical manufacturing and could violate regulatory requirements.

- Basic if-then logic lacks the sophistication needed for complex pharmaceutical quality control decisions.

04/05

**Answer:** (a)

- This solution combines structured workflows with real-time communication capabilities. Amazon A2I provides organized human review when needed, and API Gateway enables immediate response to changing conditions across time zones.

- Email notifications are too slow for real-time logistics operations and do not provide the structured interaction needed for warehouse coordination.

- Manual approval requirements would create significant delays in operations across time zones and reduce the benefits of automation.

Completely autonomous management without human oversight capabilities poses risks in complex logistics operations where local knowledge is often crucial.

05/05

**Answer:** (c)

- Using only Lambda functions is not suitable for complex monitoring tools that require persistent connections or significant processing power.

- Hosting all tools on Amazon ECS is unnecessarily complex for basic monitoring tasks and increases operational overhead.

- This hybrid approach optimally matches tool requirements with infrastructure capabilities. Lambda handles basic, stateless operations efficiently, and Amazon ECS provides the necessary resources for complex monitoring tools.

- Running tools directly on manufacturing equipment lacks proper isolation and makes updates and maintenance difficult.

## 2.3

### Knowledge Check

01/05


02/05

**Answer:** (b)

a) - Lambda authorizers add variable latency that could exceed sub-millisecond requirements.\
b) - This provides secure access control with minimal latency overhead through direct VPC communication.\
c) - Token validation adds unnecessary overhead for machine-to-machine communication.\
d) - Database lookups introduce variable latency and potential performance bottlenecks.


04/05

Correct Answer: (b)

a) - Synchronous retries can block processing and impact system throughput.\
b) - This ensures failed inferences are captured and processed appropriately while maintaining system reliability.\
c) - Logging errors without proper handling could lead to data loss or incorrect results.\
d) - Human review introduces significant latency and does not scale for high-throughput systems.

05/05

Correct Answer: (a)

a) - This enables real-time monitoring with custom metrics and automated response to monitoring events.\
b) - Although powerful, this solution does not integrate natively with AWS services and might require additional maintenance.\
c) - This approach requires significant custom development and might not provide real-time monitoring capabilities.\
d) - Although useful for model drift, it does not provide comprehensive infrastructure and operational monitoring.


## Task 2.4

### Self Assessment

2.4.1 **Answer:** (b)

Using Amazon API Gateway with Lambda integration provides a scalable, managed solution for handling synchronous requests to foundation models. This approach allows for request validation, throttling, and authentication before the request reaches Lambda, which can then call Amazon Bedrock APIs. API Gateway automatically scales to handle varying loads, and Lambda provides compute that scales with demand. 

Option A lacks proper validation and management layers. Option C introduces unnecessary latency for synchronous requests by using a queue. Option D requires manual scaling of EC2 instances, which is less efficient than serverless options for this use case.


2.4.2 **Answer:** (b)

Implement Amazon Bedrock streaming APIs with WebSockets to deliver incremental responses

Amazon Bedrock streaming APIs combined with WebSockets provide the best solution for real-time, incremental response delivery. This approach allows the application to receive and display text tokens as they're generated by the foundation model, creating a fluid typing effect for users. WebSockets maintain a persistent connection that enables bidirectional communication. 

Option A would only deliver complete responses, not incremental ones. Option C (SNS) is for asynchronous notifications, not streaming content. Option D (Kinesis) is optimized for data analytics pipelines rather than real-time UI interactions.

2.4.3  Answer (b) Implement AWS SDK exponential backoff, circuit breakers, and fallback mechanisms

Implementing AWS SDK exponential backoff, circuit breakers, and fallback mechanisms creates a resilient system that can handle temporary service limitations. Exponential backoff automatically increases wait time between retries, preventing request flooding. Circuit breakers prevent continued calls to failing services. Fallback mechanisms provide alternative responses when the primary model is unavailable.

Option A addresses capacity but lacks intelligent handling of failures. Option C doesn't address the throttling issue, which is likely a service limit. Option D adds unnecessary complexity and cost when the issue is request management, not regional availability.

2.4.4 **Answer:** (c)

Request Validators with JSON Schema

API Gateway Request Validators with JSON Schema provide a way to validate incoming requests against a defined schema before they proceed to backend services. This ensures that requests contain all required fields and adhere to size limitations before reaching Amazon Bedrock, preventing unnecessary processing of invalid requests. 

Option A (caching) improves performance but doesn't validate requests. Option B (mapping templates) transforms data but doesn't validate it. Option D (usage plans) controls throttling and quotas but doesn't validate request content.


### Knowledge Check

01/05

**Answer:** (b)

- Amazon API Gateway WebSocket APIs with Amazon Bedrock streaming and chunked transfer encoding

a)     Although REST APIs with polling can provide basic functionality, this approach introduces unnecessary latency and overhead for real-time analysis. In a trading environment where immediate feedback is crucial, the polling mechanism would create additional system load and delay analysis results to traders, potentially impacting trading decisions.

b)    This combination delivers true real-time capabilities through persistent connections and incremental response delivery. WebSocket APIs maintain continuous connections, allowing immediate updates as analysis progresses. Amazon Bedrock streaming enables the model to return results incrementally, while chunked transfer encoding ensures efficient delivery of these results. This architecture minimizes latency and provides the immediate feedback essential for trading operations.

c)     This architecture introduces unnecessary complexity and latency into the real-time analysis pipeline. Step Functions adds orchestration overhead that is not needed for streaming responses, and HTTP APIs do not support the persistent connections needed for real-time updates. The additional queuing layer would further delay the delivery of analysis results to traders.

d)    Using Amazon SQS with periodic Lambda consumers fundamentally misaligns with real-time requirements. This architecture would introduce polling delays, increase system complexity, and create unnecessary latency in delivering analysis results. In a trading environment where milliseconds matter, this approach would significantly impact the system's ability to provide immediate document analysis feedback.


02/05

**Answer:** (b), (d)

a)     Using constant retry intervals during system issues can exacerbate problems by creating predictable load spikes. In a trading environment, this could lead to cascading failures across dependent systems and does not adapt to varying system conditions.

b)    Exponential backoff with jitter provides intelligent retry spacing that adapts to system stress. By progressively increasing delays between retries and adding randomization, this approach prevents retry storms during high-volume trading periods while maintaining system stability. The jitter component is particularly crucial in distributed systems to prevent synchronized retry attempts.

c)     Although immediate recovery might seem desirable, synchronous retries without appropriate delays can overwhelm already stressed systems. During peak trading periods, this approach could lead to resource exhaustion and broader system failures.

d)    Circuit breakers with fallback mechanisms provide sophisticated failure handling that maintains critical operations. This pattern automatically detects system stress and gracefully degrades service while maintaining essential trading functions. The fallback mechanisms ensure basic document processing continues even when primary analysis capabilities are impaired.

e)     Fixed timeouts with linear retry patterns do not account for varying system conditions and can lead to predictable resource consumption patterns. This approach lacks the flexibility needed for a dynamic trading environment and does not effectively protect system resources during stress periods.

03/05

**Answer:** (b)

a)     Long polling with Amazon SQS introduces unnecessary latency and does not support real-time streaming requirements for content moderation.

b)    BedrockRuntimeClient with streaming support enables the following:
  - a.     Immediate processing of content as it arrives
  - b.     Real-time feedback through WebSocket connections
  - c.     Efficient resource utilization through streaming
  - d.     Proper error handling for stream interruptions

c)     Standard synchronous calls do not support incremental results and would introduce delays in feedback delivery.

d)    Async Lambda with Amazon SQS creates additional complexity and latency that is not suitable for real-time moderation requirements.

04/05

**Answer:** (b), (d)

a) Basic CloudWatch metrics do not provide the detailed insights needed for complex distributed systems and lack cross-service trace context.

b) X-Ray tracing with custom subsegments provides the following:

  - Detailed visibility into model operations
  - Cross-service request tracking
  - Performance bottleneck identification
  - Error propagation patterns

c) Console logging lacks structured data and does not provide distributed tracing capabilities necessary for production systems.

d) Distributed tracing with custom attributes enables the following:
  - End-to-end request visualization
  - Detailed performance analysis
  - Service dependency mapping
  - Sophisticated error tracking

e) Standard access logs lack the context and detail needed for comprehensive system observability in a distributed environment.

05/05

**Answer:** (b)

a) Direct Lambda integrations lack sophisticated routing capabilities and make it difficult to implement complex decision logic based on document characteristics. This approach does not provide the flexibility needed for dynamic model selection.

b) This combination provides powerful routing capabilities through the following:

  - Dynamic Choice states that can evaluate document content and complexity
  - Visual workflow monitoring for complex routing patterns
  - Detailed tracing of routing decisions and model selection
  - Comprehensive observability of the entire routing process
  - Built-in error handling and retry mechanisms

c) Although Amazon SQS can handle message distribution, it is not designed for complex routing logic and lacks the orchestration capabilities needed for sophisticated model selection. Message groups do not provide the dynamic decision-making required.

d) EventBridge with static rules does not provide the flexibility needed for content-based routing decisions. This approach lacks the ability to make dynamic choices based on document analysis and system conditions.


## Task 2.5

### Self Assessment

2.5.1 

**Answer:** (b)
- Implement request batching and response caching

Implementing request batching and response caching is the most effective approach for optimizing performance in GenAI applications. Batching similar requests can reduce the number of API calls, while caching responses for common queries eliminates the need to call the foundation model repeatedly for the same input. This approach directly addresses the latency issue without compromising quality. Option A might help with Lambda processing but won't affect model API latency. Option C trades quality for speed, which may not be acceptable. Option D adds complexity and cost without necessarily addressing the core latency issue.

2.5.2

**Answer:** (a)
- Create a Lambda function that processes CRM events, retrieves customer data, and uses Amazon Bedrock to generate personalized emails

Creating a Lambda function that processes CRM events, retrieves customer data, and uses Amazon Bedrock to generate personalized emails is the most effective approach. Lambda can be triggered by CRM events (via EventBridge or webhooks), access customer data from various sources, and use foundation models to generate truly personalized content based on interaction notes. This serverless approach scales automatically and only runs when needed. Option B introduces unnecessary infrastructure management. Option C is more complex than needed when pre-trained foundation models are available. Option D supports templated emails but not dynamic generation based on interaction notes.

### Knowledge Check

01/05

**Answer:** (b)

Implement WebSocket APIs with AWS Lambda for token management.  

a) - REST APIs lack native support for streaming and using DynamoDB adds unnecessary complexity.\
b) - WebSocket APIs with Lambda provide efficient streaming capabilities, flexible token management, and proper scaling capabilities for foundation model applications.\
c) - HTTP APIs do not support long-lived connections needed for streaming responses effectively.\
d) - Client-side token management is less secure and reliable for foundation model workloads.


02/05

**Answer:** (b), (d)

a) - Custom components increase development overhead and might miss accessibility requirements.\
b) - Amplify UI provides tested, accessible components with built-in AWS service integration.\
c) - CLI-only interfaces severely limit accessibility and user adoption potential.\
d) - Prompt Flows enable business users to create AI workflows without complex coding.\
e) - Third-party solutions often lack deep integration with AWS services and increase maintenance complexity.

03/05

**Answer:** (b)

Use AWS Lambda and Amazon DynamoDB for all CRM processing.

a) - This combination lacks orchestration capabilities and intelligent knowledge integration.\
b) - This combination provides robust workflow management, flexible processing, and intelligent knowledge base capabilities needed for AI-enhanced CRM systems.\
c) - SageMaker AI is excessive for basic CRM enhancements and increases complexity.\
d) - AWS Glue alone lacks AI-driven insights and workflow management capabilities.

04/05

**Answer:** (b)

Use Amazon Q Developer context-aware suggestions with custom prompt templates.

a) - This approach misses the opportunity to use the advanced capabilities of Amazon Q Developer for custom scenarios.\
b) - By combining context-aware suggestions with custom prompt templates, developers can create sophisticated, tailored, multi-agent systems while benefiting from Amazon Q Developer AI-assisted coding.\
c) - Pre-built templates alone might not suffice for complex multi-agent systems and advanced prompt engineering needs.\
d) - Manual coding without AI assistance is inefficient and prone to errors in complex AI application development.

05/05

**Answer:** (a), (d)

a) - X-Ray with custom subsegments allows for detailed tracing across microservices, capturing AI-specific interactions and latencies.\
b) - Although useful, this approach might miss critical inter-service dependencies and AI-specific patterns.\
c) - Third-party tools might not integrate as seamlessly with AWS services and might lack AI-specific insights.\
d) - Amazon Q Developer AI-specific error recognition can identify subtle patterns in foundation model behavior that traditional monitoring might miss.\
e) - Application-level logging alone lacks the depth and correlation capabilities needed for complex, distributed AI systems.