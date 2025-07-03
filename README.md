# How to Train Your Dragon (LLM) for Product Recommendation





In this post, I explain my previous approach for building a product recommendation system using Large Language Models (LLMs), the limitations encountered, and a proposed direction for training a smaller model that can improve performance over time.

---

## Old Approach: RAG with ChatGPT Wrapper

Our existing solution is built using a Retrieval-Augmented Generation (RAG) approach, where the main idea is to use ChatGPT (gpt-3.5-turbo) as a reasoning wrapper rather than training it directly.

### How It Works

**Data Sources**:
- **Product dataset**: A list of available products.
- **User preferences**: Provided separately.
- **Browsing history**: Collected from the React-based UI.

**Vector Search**:
- All product descriptions are embedded and stored in a vector database (I use FAISS).
- The user’s most recent browsing history is also embedded.
- We retrieve the top-K nearest products from the vector database based on this embedding.

**Relevance Scoring**:
- These top-K products are sent to ChatGPT with a structured prompt.
- ChatGPT returns a relevance score for each item.
- We sort and fetch the top-N products based on these scores and recommend them to the user.

---

## Limitations

While this approach works to some extent, it has a few important drawbacks:

- **Cold Start Problem**: If a user has no or very little browsing history, recommendations become unreliable.
- **Lack of Diversity**: For example, if the browsing history contains a monitor, the system tends to recommend only similar items (more monitors). In many cases, suggesting related accessories like keyboards or mouse devices would be more effective.

---

## Why Train a Model?

Instead of using ChatGPT solely as a wrapper, training a dedicated model can yield better performance. A model fine-tuned on our dataset can learn user behavior patterns more effectively and provide diverse recommendations based on both browsing history and general trends.

---
## Background

There are many machine learning models available for clustering all the products and making recommendations, but in this post, I will focus on how Large Language Models (LLMs) can be used for product recommendation—particularly how we’ve adapted them for personalized suggestions based on user behavior
---
## Dataset Preparation

The quality and structure of the dataset play a critical role in building a successful recommendation system. Most existing solutions rely heavily on the structure of the data. To move forward, I have created a dataset that includes:

- Product metadata  
- Simulated browsing histories  
- Annotated preferred recommendations (including edge cases and anomalies)

This dataset mirrors the decisions made by ChatGPT in our current pipeline. The goal is to train a smaller model using this data in a teacher-student setup, where ChatGPT serves as the teacher.

---


## Alternative Training Approaches Based on Dataset

There are many approaches we can take, depending on the kind of dataset we have. For example, if we have data with `user_id`, `last_viewed`, `last_add_to_cart`, and `last_purchased` items, we can fine-tune an LLM for next-product recommendation using supervised learning.

**Example Input-Output Pair**:
```
Input: “User viewed prod1, prod2, added prod5, purchased prod6”  
Output: “Recommend: prod8, prod9”
```

Any transformer or LLM can be fine-tuned on this data using supervised learning. With enough training samples, the model can learn strong recommendation capabilities.

### Reinforcement Learning from Human Interaction (RLHI)

Once the model is deployed, we can refine it using RLHI. Whenever a product is actually purchased or clicked by the user, this interaction can act as a **reward signal**.

You're capturing real intent through user actions—which is more reliable than any synthetic reward.

**Example JSON format**:
```json
{
  "user_browsing_history": "['prod_21', 'prod_13', 'prod_09']",
  "llm_recommendations": "['prod_31', 'prod_45', 'prod_57']",
  "user_action": "clicked prod_45"
}
```

These interactions can be used to continuously fine-tune the model and personalize recommendations further.

---

## Towards an Online Learning System

Once we have a trained model, we can improve it over time using Reinforcement Learning from Human Interaction (RLHI) or simplified forms of it. As users interact more with the system, the model can be updated regularly to:

- Reflect shifting user preferences  
- Improve accuracy based on live signals  
- Adapt to new product inventory and trends
