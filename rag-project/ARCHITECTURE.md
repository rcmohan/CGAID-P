# Exam Framing

## Domain 1: Foundations & RAG

  * **Bedrock-managed RAG (KB + RetrieveAndGenerate)**

  Using Amazon Bedrock to manage the RAG pipeline simplifies the process of building and deploying RAG applications. It provides a unified interface for accessing and managing multiple LLMs and provides built-in support for vector stores, embeddings, and retrieval.

Bedrock implementation requires the following steps:

  * Create a S3 bucket for the documents that ground truth the RAG system
  * Create a vector store (OpenSearch, Aurora, Pinecone)
  * Create a Knowledge Base
  * Create a Lambda function to handle the RAG requests
  * Create a Bedrock Agent to handle the RAG requests

  * **FM Selection Guide (Exam Critical)**
The FM selection & routing is the process of selecting the right model for the right task. It is critical for the performance and cost of the RAG system. Foundation models are the models that are used to generate the responses, and application models are the models that are used to generate the responses for specific tasks. In this case, we are using a foundation model to generate the responses for the RAG system. 

    * **Amazon Titan:** 
      * *Best For:* General purpose, cost-efficiency, and **Embeddings** (Titan Embeddings v2).
      * *Use Case:* Summarization, text generation, and RAG vector creation.
    * **Anthropic Claude (Haiku/Sonnet/Opus):**
      * *Best For:* **Complex Reasoning**, Coding, Chain-of-Thought, and Nuance.
      * *Use Case:* "The application requires high accuracy for legal reasoning" or "Generating complex Python code."
      * *Note:* Claude 3 Haiku is the speed/cost leader for "intelligence" tasks.
    * **Meta Llama 3:**
      * *Best For:* **Open Weights** and General Purpose.
      * *Use Case:* When the requirement is "Fine-tuning an open model" or "Portability."
    * **Cohere Command & Embed:**
      * *Best For:* **Business/Enterprise** tasks and **Multilingual Embeddings**.
      * *Use Case:* "Connectors" to enterprise data sources or non-English retrieval tasks.
    * **AI21 Jurassic:**
      * *Best For:* **Instruction Following** and heavily text-centric tasks.

  The FM models are selected based on the following criteria:
    * Cost: FMs vary in cost. The cheapest FMs are Titan series. The most expensive FMs are the higher-end Claude (Opus and Sonnet) series.
    * Performance: Performance is defined as both accuracy and speed. The Claude 3 Family (especially Opus and Sonnet) outperforms Titan Text in complex reasoning, coding, and nuance. Claude 3 Haiku and Titan models are the most performant.
    * Latency: Latency is defined as the time it takes for the model to generate a response. Claude 3 Haiku is incredibly fast (low latency) and cheap, often beating Titan in speed/quality balance. Claude 3 Opus is the most accurate, but also the slowest.
    * Security and Compliance:Security is defined as the ability to protect the model from being used for malicious purposes. Compliance is defined as the ability to meet regulatory requirements. All models available via Amazon Bedrock (Titan, Claude, Llama, Command) fall within the same AWS security boundary.


  * **Vector store options and tradeoffs**
    * **Bedrock Integration:** When you create a **Knowledge Base** in Amazon Bedrock, you must choose where to store your vectors. Your options are:
        * Amazon OpenSearch Serverless *(AWS Native)*
        * Amazon Aurora *(AWS Native)*
        * **Pinecone** *(Third-Party)*
        * Redis Enterprise Cloud *(Third-Party)*
        * MongoDB Atlas *(Third-Party)*
    * **Hybrid search, reranking**: Hybrid search is the process of searching both the vector store and the document store. Reranking is the process of re-ranking the results of the hybrid search to improve the quality of the results.
    * **Embedding selection tradeoffs (dimension, cost, latency)**: Embedding selection tradeoffs are between the dimension of the embedding (vector size), the cost of generating the embedding, and the latency of generating the embedding. 

> **Pinecone** is a fully managed, cloud-native **Vector Database**. While it is **not** an AWS-owned service (it is a third-party SaaS), it is highly relevant to the exam because it is a **supported integration** for Amazon Bedrock Knowledge Bases. In an exam scenario, you might choose Pinecone if the requirements specify using a **"specialized vector database"** or if the customer is **"already using Pinecone"** and wants to connect it to Bedrock without migrating data.




  * **Cost & latency awareness**

    * Cost of embeddings: Cost of embeddings depends on the model used and the number of tokens processed. 
    * Cost of vector store: Cost of vector store depends on the vector store used and the number of tokens processed.
    * Cost of Lambda: Cost of Lambda depends on the number of requests processed.
    * Cost of Bedrock: Cost of Bedrock depends on the model used and the number of tokens processed.
    * Cost of S3: Cost of S3 depends on the number of objects stored.
    * Cost of OpenSearch: Cost of OpenSearch depends on the number of documents stored.
    * Cost of Aurora: Cost of Aurora depends on the number of documents stored.
    * Cost of Pinecone: Cost of Pinecone depends on the number of documents stored.
    * Cost of VPC: A VPC itself is free. What costs money are VPC Endpoints (PrivateLink) (hourly charge + data processing fee) and NAT Gateways.
    * Cost of PrivateLink: Cost of PrivateLink depends on the number of requests processed.
    * Cost of Cross-Region Inference: Cost of Cross-Region Inference depends on the number of requests processed.

* **IAM & Governance Boundaries**

    * **1. Bedrock Knowledge Base (KB) Service Role (The "Ingestion" Role)**

        * *Who assumes it:* The **Amazon Bedrock Service** (`bedrock.amazonaws.com`).
        * *Permissions Required:*
        * **S3 Access:** `s3:GetObject` and `s3:ListBucket` (To read the source documents).
        * **Embedding Model Access:** `bedrock:InvokeModel` (To turn those docs into vectors using Titan/Cohere).
        * **Vector Store Access:** Permission to write/index into the vector DB (e.g., `aoss:APIAccessAll` for OpenSearch Serverless).


        * *Correction:* The *Vector Store* does **not** access S3. Bedrock accesses S3, creates embeddings, and *pushes* them to the Vector Store.

    * **2. Bedrock Agent Service Role (The "Action" Role)**

        * *Who assumes it:* The **Amazon Bedrock Agent** (`bedrock.amazonaws.com`).
        * *Permissions Required:*
        * **Knowledge Base Access:** `bedrock:Retrieve` (To query the KB for answers).
        * **Foundation Model Access:** `bedrock:InvokeModel` (To reason and generate the final answer).
        * **Action Group Access:** `lambda:InvokeFunction` (To execute the tools defined in Action Groups).

    * **3. The `iam:PassRole` Requirement (Exam Critical!)**

        * *The Scenario:* You (the developer) are trying to create a Bedrock Agent in the console or via CLI.
        * *The Error:* "User is not authorized to perform: iam:PassRole".
        * *The Reason:* You are assigning a Service Role (from step #2) to the Agent. AWS needs to know you have permission to "pass" that powerful role to the service. You cannot assign a role to a service if you don't have `iam:PassRole` on that specific role ARN.

    * **4. OpenSearch Serverless (AOSS) Data Access Policies**

        * *The "Gotcha":* Standard IAM policies are **not enough** for OpenSearch Serverless.
        * *The Fix:* You must configure a **Data Access Policy** *inside* OpenSearch Serverless that explicitly grants the **Bedrock Service Role ARN** permission to read/write to the *collection*.
        * *Rule:* IAM grants "access to the API". Data Access Policy grants "access to the data inside the index".

    * **5. Cross-Service Confused Deputy Prevention**

        * *The Pattern:* In the Trust Policy of any Service Role (Bedrock, Lambda), you generally see a `Condition` block:
        ```json
        "Condition": {
            "StringEquals": {
                "aws:SourceAccount": "YOUR_ACCOUNT_ID"
            },
            "ArnLike": {
                "aws:SourceArn": "arn:aws:bedrock:us-east-1:123456789012:knowledge-base/*"
            }
        }
        ```
        * *The Reason:* This is a security measure to prevent a malicious service from assuming your role and performing actions on your behalf. This ensures that only your specific Knowledge Base (not a bad actor's KB) can assume the role to read your data.


 ### Out of scope (explicitly not built)

  * **Fine-tuning**
  Fine-tuning is the process of training a model on a specific dataset to improve its performance on a specific task. It is not covered in this exam. It also requires additional resources and expertise.

  * **Training pipelines**
  Training pipelines are the processes and tools used to train models. It is not covered in this exam. It also requires additional resources and expertise.

  * **Custom embedding training**
  Custom embedding training is the process of training a model on a specific dataset to improve its performance on a specific task. It is not covered in this exam. It also requires additional resources and expertise.

  * **Model internals**
  Model internals are the internal workings of a model. It is not covered in this exam. It also requires additional resources and expertise.