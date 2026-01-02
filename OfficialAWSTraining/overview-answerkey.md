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

