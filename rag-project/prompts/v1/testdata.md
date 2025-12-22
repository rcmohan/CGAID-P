## Test Data

### Data Set 1

**Default Model Tier**

* **Test Data**
```json
{
  "input": "Which was the top grossing movie of May 2025?",
  "topK": 5
}
```



**Performance Model Tier**

* **Test Data**
```json
{
  "input": "Which was the top grossing movie of May 2025?",
  "topK": 5,
  "modelTier": "performance"
}
```

**Deep Model Tier**

* **Test Data**
```json
{
  "input": "Which was the top grossing movie of May 2025?",
  "topK": 5,
  "modelTier": "deep"
}
```

* **Test Response**
```json
{
  "answer": "The top grossing movie of May 2025 was \"Lilo & Stitch,\" which grossed $260,128,724 and was released on May 23 by Walt Disney Studios Motion Pictures.",
  "citations": [
    {
      "generatedResponsePart": {
        "textResponsePart": {
          "span": {
            "end": 148,
            "start": 0
          },
          "text": "The top grossing movie of May 2025 was \"Lilo & Stitch,\" which grossed $260,128,724 and was released on May 23 by Walt Disney Studios Motion Pictures"
        }
      },
      "retrievedReferences": [
        {
          "content": {
            "text": "Domestic Box Office For May 2025\r By Month By Month May May 2025 2025 Calendar grosses Calendar grosses Key:Estimated Rank\tRelease\tGross\tTheaters\tTotal Gross\tRelease Date\tDistributor 1\tLilo & Stitch\t$260,128,724\t4,410\t$423,778,855\tMay 23\tWalt Disney Studios Motion Pictures 2\tThunderbolts*\t$180,498,688\t4,330\t$190,274,328\tMay 2\tWalt Disney Studios Motion Pictures 3\tSinners\t$123,840,604\t3,518\t$279,653,537\tApr 18\tWarner Bros. 4\tMission: Impossible - The Final Reckoning\t$114,424,705\t3,861\t$197,413,515\tMay 23\tParamount Pictures International 5\tFinal Destination: Bloodlines\t$108,594,467\t3,523\t$138,130,814\tMay 16\tWarner Bros. 6\tA Minecraft Movie\t$39,170,628\t4,289\t$423,949,195\tApr 4\tWarner Bros. 7\tThe Accountant 2\t$34,317,505\t3,610\t$65,523,366\tApr 25\tAmazon MGM Studios 8\tKarate Kid: Legends\t$",
            "type": "TEXT"
          },
          "location": {
            "s3Location": {
              "uri": "s3://askmydocs-dev/may2025.md"
            },
            "type": "S3"
          },
          "metadata": {
            "x-amz-bedrock-kb-source-file-modality": "TEXT",
            "x-amz-bedrock-kb-chunk-id": "cbc1d886-785f-474a-a446-ba6c60d5a0a1",
            "x-amz-bedrock-kb-data-source-id": "ZHF1LHZQ8N"
          }
        }
      ]
    }
  ],
  "sessionId": "26a0459b-e3ff-4cda-81d0-4cde04477e62"
}
```

**Cheap Model Tier**

* **Test Data**
```json
{
  "input": "Which was the top grossing movie of May 2025?",
  "topK": 5,
  "modelTier": "cheap"
}
```
* **Test Response**

>  Status: Succeeded
>
>  Test Event Name: TopGrosserMay25Fast
> 
>  Response:
> 
```json
  {
    "answer": "Answer: Lilo & Stitch",
    "citations": [
      {
        "generatedResponsePart": {
          "textResponsePart": {
            "span": {
              "end": 21,
              "start": 0
            },
            "text": "Answer: Lilo & Stitch"
          }
        },
        "retrievedReferences": [
          {
            "content": {
              "text": "Domestic Box Office For May 2025\r By Month By Month May May 2025 2025 Calendar grosses Calendar grosses Key:Estimated Rank\tRelease\tGross\tTheaters\tTotal Gross\tRelease Date\tDistributor 1\tLilo & Stitch\t$260,128,724\t4,410\t$423,778,855\tMay 23\tWalt Disney Studios Motion Pictures 2\tThunderbolts*\t$180,498,688\t4,330\t$190,274,328\tMay 2\tWalt Disney Studios Motion Pictures 3\tSinners\t$123,840,604\t3,518\t$279,653,537\tApr 18\tWarner Bros. 4\tMission: Impossible - The Final Reckoning\t$114,424,705\t3,861\t$197,413,515\tMay 23\tParamount Pictures International 5\tFinal Destination: Bloodlines\t$108,594,467\t3,523\t$138,130,814\tMay 16\tWarner Bros. 6\tA Minecraft Movie\t$39,170,628\t4,289\t$423,949,195\tApr 4\tWarner Bros. 7\tThe Accountant 2\t$34,317,505\t3,610\t$65,523,366\tApr 25\tAmazon MGM Studios 8\tKarate Kid: Legends\t$",
              "type": "TEXT"
            },
            "location": {
              "s3Location": {
                "uri": "s3://askmydocs-dev/may2025.md"
              },
              "type": "S3"
            },
            "metadata": {
              "x-amz-bedrock-kb-source-file-modality": "TEXT",
              "x-amz-bedrock-kb-chunk-id": "cbc1d886-785f-474a-a446-ba6c60d5a0a1",
              "x-amz-bedrock-kb-data-source-id": "ZHF1LHZQ8N"
            }
          }
        ]
      }
    ],
    "sessionId": "7752479e-da60-494f-8eea-0951ffff97a1"
  }
  ```
