# How to Train Your Dragon (LLM) for Product Recommendation

In this post, i have explained my approach for building a product recommendation system using Large Language Models (LLMs). We begin with our current method using Retrieval-Augmented Generation (RAG), identify its limitations, and then present a structured path toward training and evolving a more personalized recommender system using Reinforcement Learning from Human Interaction (RLHI).

---

## Current System: RAG with ChatGPT

my existing setup uses a RAG pipeline where ChatGPT (gpt-3.5-turbo) acts as a reasoning engine to score products retrieved from a vector database.

### Workflow

**Data Sources**:
- Product catalog
- User preferences and Browsing history collected from the UI

**Vector Search**:
- All product descriptions and recent user history are embedded.
- The top-K similar products are retrieved using FAISS.

**Relevance Scoring**:
- These products are passed to ChatGPT in a structured prompt.
- ChatGPT returns relevance scores which are used to recommend the top-N products.

---

## Observed Limitations

1. **Cold Start Problem**: Sparse or missing browsing history leads to unreliable recommendations.
2. **Lack of Diversity**: Over-specialization (e.g., viewing a monitor yields more monitors) with little exploration into complementary products.
3. **Static Behavior**: No ability to learn from real-time feedback or user-level interactions.

---

## Why Move Beyond RAG?

While the RAG+LLM approach is effective for general-purpose recommendations, it lacks adaptability. Training a task-specific model or lightweight LLM variant enables more fine-tuned behavior. But even supervised fine-tuning has limitations—it cannot efficiently adapt to new trends or individual user preferences over time.

---

## Background

Traditional recommender systems use collaborative filtering, clustering, or graph-based methods. While effective at scale, they often miss nuance in user behavior.

With LLMs, we can reframe recommendation as a sequence prediction or language modeling problem—treating browsing and purchase behavior as structured narratives. This opens the door for both zero-shot reasoning and continual adaptation.

---
## Pipeline Overview

```
         +---------------------+
         | Browsing History    |
         +----------+----------+
                    |
                    v
         +---------------------+
         | LLM (Base or SFT)   |  <-- Initial supervised model
         +----------+----------+
                    |
          Generate Recommendations
                    |
                    v
         +---------------------+
         | User Feedback Logs  |  <-- Hover, click, purchase, etc.
         +----------+----------+
                    |
                    v
         +---------------------+
         | Reward Model (optional)     |
         +----------+----------+
                    |
                    v
         +---------------------+
         | RLHF Fine-Tuning    |  <-- PPO or DPO
         +----------+----------+
                    |
                    v
         +---------------------+
         | Updated Recommender |
         +---------------------+
```
The pipeline starts with the user’s browsing history as input to a base or fine-tuned language model, which generates product recommendations. As users interact—click, hover, purchase, or scroll past—these actions are logged as feedback. A reward model (or simple heuristics) assigns value to these interactions. Using this reward signal, the model is fine-tuned with methods like PPO or DPO to better align with user preferences. This updated model is then redeployed, creating a feedback loop that enables real-time personalization and continuous improvement.


## Supervised Training with Sequence Inputs

If the dataset contains user sessions with clear transitions from view to cart to purchase, we can train a model with simple input-output patterns like:

```
Input: "User viewed prod1, prod2, added prod5, purchased prod6"  
Output: "Recommend: prod8, prod9"
```

Such sequences can be used to fine-tune a base transformer using standard supervised objectives.

---

## Reinforcement Learning from Human Interaction (RLHI)

To personalize and adapt over time, we move beyond supervised learning. RLHI allows the model to improve using real user interactions as feedback.

These interactions include:

- Click
- Hover
- Scroll past
- Add to cart
- Remove from cart
- Purchase
- Review after purchase

Each action can be treated as an implicit signal of user intent. Strong signals like purchases or reviews can reinforce useful recommendations, while clicks and hovers offer fine-grained feedback.

### Example Interaction Log

```json
{
  "user_browsing_history": ["prod_21", "prod_13", "prod_09"],
  "llm_recommendations": ["prod_31", "prod_45", "prod_57"],
  "user_action": "clicked prod_45"
}
```

Such logs can serve as reward signals for fine-tuning, allowing the model to learn user preferences in real time.

---

## Why RLHI Works Better Than Just Fine-Tuning

Supervised fine-tuning captures general trends across many users, but:

- It cannot learn from individual user behavior in real time.
- It becomes outdated as product catalogs or trends change.

RLHI, on the other hand, enables dynamic, user-specific adaptation—allowing the system to evolve with every interaction.

---

## Closing Thoughts

By combining LLM reasoning with continual feedback via RLHI, we can move toward a truly adaptive, user-centric recommendation system—one that personalizes based on context, history, and evolving intent.

This approach not only improves short-term accuracy but ensures long-term robustness in dynamic e-commerce environments.
