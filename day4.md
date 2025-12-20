## **Day 4: Prompt Engineering & Management**.

First, let's wrap up Day 3 with the answer to the Law Firm scenario:

* **The Solution:** You should use **Amazon Bedrock Knowledge Bases** backed by **Amazon OpenSearch Serverless**. This meets the "serverless" and "managed" requirement.
* **The "Case Year" Filter:** You would upload a `.metadata.json` file for each PDF (e.g., `case123.pdf.metadata.json` containing `{"year": "2020"}`). Then, in your application code, you pass a `generationConfiguration` or `retrievalConfiguration` filter to the `RetrieveAndGenerate` API to strictly include `year = 2020`.
* **The Deletion:** You must run a **Sync Job** (StartIngestionJob). Bedrock KB syncs with S3 to reflect adds, updates, *and* deletions.

---

### **Day 4: Prompt Engineering & Management**

**Goal:** Moving beyond "typing into a chatbox" to treating prompts as production code—versioned, tested, and optimized.

**1. The Techniques (Exam Vocabulary)**
You will likely see questions asking you to match a problem to a technique:

* **Zero-Shot:** Asking the model to do something without examples. (Good for simple tasks).
* **Few-Shot:** Providing 3-5 examples of "Input -> Output" inside the prompt to guide the model's style. (Crucial for specific formatting, like JSON output).
* **Chain-of-Thought (CoT):** Asking the model to "think step-by-step" or "explain your reasoning."
* *Exam Tip:* If a scenario says the model is failing at **complex math** or **multi-step logic**, the answer is almost always to **add Chain-of-Thought instructions**.


* **Retrieval Augmented Generation (RAG):** (We covered this) Injecting external data into the prompt context.

**2. Amazon Bedrock Prompt Management**
This is a relatively new feature heavily emphasized in the "Professional" level exam to separate code from prompts.

* **Prompt Library:** You can create and save prompts in the Bedrock console.
* **Versions:** You can create "Version 1" (Draft) and "Version 2" (Production).
* **Variables:** You use placeholders like `{{customer_name}}` or `{{transcript}}` in your saved prompt. Your code then just calls the prompt ID and fills in the variables.
* *Why use it?* It allows developers to update the prompt wording without redeploying the application code.



**3. Amazon Bedrock Prompt Flows**

* **What is it?** A visual builder to chain multiple prompts together.
* **Use Case:** If you need to:
1. Classify an email (Prompt A).
2. *If Complaint:* Extract order number (Prompt B).
3. *If Compliment:* Generate thank you note (Prompt C).


* Prompt Flows handles this branching logic without writing complex Python `if/else` glue code.



**4. Inference Profiles (System-defined vs. Application)**

* **System-defined:** Standard, cross-region routing (e.g., `us.anthropic.claude-3-5-sonnet-20240620-v1:0`).
* **Application Inference Profiles:** You create these to track metrics and costs for *specific* use cases (e.g., "MarketingBot-Profile" vs "SupportBot-Profile") while using the same underlying model.

---

### **Day 4 Scenario Quiz**

You are building a customer support agent that classifies incoming emails as "Urgent" or "Routine."
You tried a simple prompt ("Classify this email..."), but the model is inconsistent. It often marks angry customers as "Routine" if they don't use specific swear words.
You want to improve accuracy without changing the underlying model (Claude 3 Haiku).

**Question:**

1. Which **Prompt Engineering technique** is the most effective next step to fix this classification issue?
2. Your team wants to A/B test a new, friendlier prompt for the "Reply" generation phase without redeploying the backend application code. Which **Bedrock feature** allows you to manage this switch externally?