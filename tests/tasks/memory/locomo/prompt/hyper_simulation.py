LOCOMO_HYPER_PROMPT = """
### Context (Ranked by Relevance, Highest to Lowest):
{context_text}

Answer with a short phrase only (no full sentence, no explanation, no markdown) based on the above context.
Question: {question}
Short answer:
"""

LOCOMO_HYPER_RAG_PROMPT = """
### Context (Ranked by Relevance, Highest to Lowest):
{context_text}

Answer with a short phrase only (no full sentence, no explanation, no markdown) based on the above context.
Question: {question}
Short answer:
"""

# LOCOMO_HYPER_RAG_PROMPT = """
# ### Context (Ranked by Relevance, Highest to Lowest):
# {context_text}

# You are an intelligent memory assistant tasked with retrieving accurate information from ranked conversation chunks.

# # INSTRUCTIONS:
# 1. Read across all ranked chunks before answering, not just the first one.
# 2. Analyze all provided chunks and combine evidence when multiple chunks are relevant.
# 3. Prefer direct evidence, but when the wording is paraphrased, partial, or indirect, return the most likely answer supported by the context.
# 4. Pay special attention to timestamps, names, and concrete attributes when the question asks for them.
# 5. Use exact words from the context whenever possible.
# 6. Do not say "Not mentioned", "Unknown", or refuse the question. Return the best short answer supported by the provided context.
# 7. If the evidence is imperfect, still provide the most likely answer instead of refusing.
# 8. The final answer must be brief, direct, and under 5-6 words.

# Question: {question}
# Short answer:
# """
