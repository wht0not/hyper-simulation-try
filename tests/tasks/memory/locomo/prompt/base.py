# QA_PROMPT = """
# Answer with a short phrase.\n\n

# Question: {}\n
# Short answer:\n
# """

# QA_PROMPT_CAT_5 = """
# Answer with only "(a)" or "(b)".

# Question: {}
# Reply with only (a) or (b).
# """

# QA_PROMPT = """
# You are an advanced reflective memory-assistant. Your goal is to provide an answer by balancing literal retrieval, inferred intent, conversational nuance, and possible implicit assumptions from two speakers.

# # CONTEXT:
# You have access to conversation memories from two speakers. The memories may include timestamps, incomplete statements, emotionally colored wording, indirect references, and potentially conflicting fragments.

# # INSTRUCTIONS:
# 1. First consider direct evidence, but also consider indirect implications if direct evidence appears sparse.
# 2. If memories conflict, do not immediately discard older memories; compare chronology, tone, and possible speaker uncertainty.
# 3. When answering time-related questions, use timestamps when present, but if timestamps are noisy, you may infer a likely timeframe from narrative flow.
# 4. Use exact words from memory whenever possible, but prefer semantically clearer paraphrases when exact wording seems ambiguous.
# 5. Consider both speaker-specific facts and cross-speaker context, including implicit relations not explicitly stated in one line.
# 6. Prefer concise answers, but include disambiguating detail if multiple plausible interpretations exist.
# 7. Avoid overcommitting unless evidence is strong, yet still provide the most likely answer instead of refusing.
# 8. When uncertain, prioritize coherence with the overall conversation trajectory rather than isolated snippets.
# 9. Keep the final answer short in most cases, but permit slightly longer phrases when precision would otherwise be lost.
# 10. Use your best judgment to trade off faithfulness, completeness, and readability.
# 11. Only output the answer itself. Do not write any analysis, reasoning, or explanation. The final answer must be brief, direct, and under 5-6 words.

# Question: {}

# Answer:
# """

QA_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. If memories conflict, prefer the most recent direct evidence.
2. Analyze all provided memories from both speakers.
3. Pay special attention to timestamps when the question asks about time.
4. Focus only on the provided memories from the two speakers.
5. Use exact words from the memories whenever possible.
6. The final answer must be brief, direct, and under 5-6 words.

Question: {} Answer:
"""
