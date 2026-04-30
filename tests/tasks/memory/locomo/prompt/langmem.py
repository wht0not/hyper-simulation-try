LOCOMO_LANGMEM_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers.
2. Pay special attention to timestamps when the question asks about time.
3. If memories conflict, prefer the most recent direct evidence.
4. Convert relative time references such as "last year" or "next month" into concrete dates or years using the memory timestamp.
5. Focus only on the provided memories from the two speakers.
6. Use exact words from the memories whenever possible.
7. The final answer must be brief, direct, and under 5-6 words.

Memories for user {speaker_1_user_id}:
{speaker_1_memories}

Memories for user {speaker_2_user_id}:
{speaker_2_memories}

Question: {question}

Answer:
"""

LOCOMO_LANGMEM_PROMPT_CAT_5 = """
You are an intelligent memory assistant tasked with judging whether the conversation memories support a candidate answer.

# CONTEXT:
You have access to memories from two speakers in a conversation.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers.
2. One option states that no information is available and the other gives a candidate answer.
3. Choose the option best supported by the memories only.
4. Reply with only `(a)` or `(b)`.

Memories for user {speaker_1_user_id}:
{speaker_1_memories}

Memories for user {speaker_2_user_id}:
{speaker_2_memories}

Question: {question}

Answer:
"""
