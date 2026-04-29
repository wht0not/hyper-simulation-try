# LOCOMO_HYPER_PROMPT = """
# {context_text}

# Based on the above context, write an answer in the form of a short phrase.
# For the following question, Answer with exact words from the context whenever possible.
# If you are sure the answer cannot be found in the context, just output "no information available".
# \n\n

# ### Question:
# {question}

# ### Hyper-simulation Hints:
# The following key elements are closely related to the question:
# {non_conflict_items}
# """

# LOCOMO_HYPER_PROMPT = """
# {context_text}

# You are given a question and a set of hints that are highly relevant keywords extracted from the context.

# Instructions:
# 1. Answer the question with a short phrase, using exact words from the context whenever possible.
# 2. If the question asks about a time, date, or location, provide the absolute value (e.g., "February 7th", "London", "3pm") – never use relative terms like "yesterday", "tomorrow", "next week", or "here".
# 3. The "Hyper-simulation Hints" are key elements strongly related to the question. Focus on the segments of the context that contain these hints – they are your most reliable guide.
# 4. Only respond with "no information available" if you are **absolutely certain** that the context contains no relevant information to answer the question, AND the hints provide no usable clues, OR the question directly contradicts the established facts in the context. Do not output "no information available" just because the answer is not obvious or requires inference. Use it only as a last resort.

# ### Question:
# {question}

# ### Hyper-simulation Hints:
# {non_conflict_items}
# """

# LOCOMO_HYPER_PROMPT = """
# {context_text}

# You must answer the following question based ONLY on the context above. Follow these rules strictly:

# 1. **Always provide a short answer** (a few words) using exact words from the context.
# 2. For questions about time, date, or location, give the **absolute value** (e.g., "February 7th", "3 PM") – never relative words like "yesterday", "tomorrow", "here".
# 3. The "Hyper-simulation Hints" are keywords strongly related to the question. They appeared in the context and can help you locate the answer.
# 4. If the answer requires combining multiple sentences or some reasoning, still try to produce the most specific answer you can.
# 5. If you are unsure but there is some relevant information, give what you think is most likely – do not refuse.

# Question: {question}
# Hyper-simulation Hints: {non_conflict_items}
# Answer:
# """

LOCOMO_HYPER_PROMPT = """
{context_text}

Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.
The conversation snippets above are re-ranked by relevance from higher priority to lower priority, so prefer earlier snippets when evidence conflicts.
Within each session block, all utterances belong to the `Session Date` shown at the top of that block.
For questions about time, date, or location, answer with absolute values rather than relative terms like "yesterday" or "tomorrow".

Question: {question} Short answer:\n
"""

QA_PROMPT = (
    "Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.\n\n"
    "Question: {} Short answer:\n"
)
