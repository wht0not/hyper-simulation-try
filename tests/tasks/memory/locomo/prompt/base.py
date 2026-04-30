QA_PROMPT = (
    "Based on the above context, write an answer in the form of a short phrase for the following question. "
    "Answer with exact words from the context whenever possible.\n\n"
    "Question: {} Short answer:\n"
)

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.
One option states that no information is available and the other gives a candidate answer.

Question: {}
Reply with only (a) or (b).
"""
