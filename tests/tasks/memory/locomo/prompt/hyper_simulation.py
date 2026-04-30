LOCOMO_HYPER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
{context_text}

# INSTRUCTIONS:
1. Carefully analyze all provided conversation snippets.
2. The snippets are re-ranked by graph simulation from higher priority to lower priority.
3. For each snippet, `critical` is the number of unique data-side nodes in that snippet matched by critical query nodes, and `total` is the number of unique data-side nodes matched by all query nodes.
4. Treat `critical` and `total` as soft relevance hints rather than proof that a snippet answers the question.
5. Pay close attention to the `Session Date`. All utterances inside the same session block happened on that date.
6. If the question involves relative time references such as "last year", "yesterday", or "two months ago", convert them to specific dates, months, or years using the session date.
7. If snippets contain contradictory information, prefer the snippet with the clearest direct evidence; if still tied, prefer the more recent dated snippet.
8. Use exact words from the context whenever possible.
9. The final answer must be brief, direct, and under 5-6 words.

# APPROACH:
1. First, inspect the highest-ranked snippets that directly mention the question topic.
2. Compare dates, entities, events, and attributes carefully.
3. Resolve relative time expressions using the corresponding session date.
4. Return a precise short answer with no explanation.

Question: {question}

Answer:
"""

LOCOMO_HYPER_PROMPT_CAT_5 = """
You are an intelligent memory assistant tasked with judging whether the conversation supports a candidate answer.

# CONTEXT:
{context_text}

# INSTRUCTIONS:
1. Carefully analyze all provided conversation snippets.
2. The snippets are re-ranked by graph simulation from higher priority to lower priority.
3. For each snippet, `critical` is the number of unique data-side nodes in that snippet matched by critical query nodes, and `total` is the number of unique data-side nodes matched by all query nodes.
4. Treat `critical` and `total` as soft relevance hints rather than proof.
5. One option states that no information is available; the other gives a candidate answer.
6. Choose the option best supported by the context only.
7. Reply with only `(a)` or `(b)`.

Question: {question}

Answer:
"""
