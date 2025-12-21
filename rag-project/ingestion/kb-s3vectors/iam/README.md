# IAM Roles and Permissions for Bedrock KB (S3 Vectors)

This folder documents the IAM setup required to create and operate
a Bedrock Knowledge Base using Amazon S3 Vectors.

---

## 1. Bedrock Execution Role

### Role Name
`AmazonBedrockExecutionRoleForKnowledgeBase`

### Trusted Entity
```json
{
  "Service": "bedrock.amazonaws.com"
}
````

### Permissions (attached to role)

* Read access to source S3 bucket (`askmydocs-dev`)
* Full access to S3 Vectors operations 
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3Vectors",
            "Effect": "Allow",
            "Action": "s3vectors:*",
            "Resource": "*"
        },
        {
            "Sid": "AllowReadDocsBucket",
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:GetObject"
            ],
            "Resource": [
                "arn:aws:s3:::BUCKET_NAME",
                "arn:aws:s3:::BUCKET_NAME/*"
            ]
        }
    ]
}
```

* Permission to write logs

Key permissions include:

* `s3:GetObject`
* `s3:ListBucket`
* `s3vectors:*`

This role is assumed by Bedrock during ingestion and retrieval.

---

## 2. Console / User Role

The IAM user or role used in the AWS Console must have:

* `AmazonBedrockFullAccess`
* `AmazonS3FullAccess`
* Custom inline policy allowing:

  * `iam:PassRole` for the execution role

Example:

```json
{
  "Effect": "Allow",
  "Action": "iam:PassRole",
  "Resource": "arn:aws:iam::<ACCOUNT_ID>:role/AmazonBedrockExecutionRoleForKnowledgeBase"
}
```

---

## 3. Common Failure Modes

| Symptom                            | Cause                          |
| ---------------------------------- | ------------------------------ |
| KB creation fails                  | Missing `iam:PassRole`         |
| Vector bucket created but KB fails | Missing `s3vectors:*`          |
| Access denied during sync          | Execution role missing S3 read |

---

## Summary

Two identities are always involved:

1. **You** (creating/configuring the KB)
2. **Bedrock** (executing ingestion & retrieval)

Permissions must be granted to both.

