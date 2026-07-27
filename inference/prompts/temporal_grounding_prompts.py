# 论文测试 Ours - CoT/GRPO
DIRECT_HIGHLIGHT_DETECT_PROMPT_TEMPLATE = """\
Your task is to create a 10-second highlight video clip plan \
(1–3 segments, each ≥2 seconds) for the specified product \
【{CORE_PRODUCT}】from a live-stream shopping video.

High-quality segments must simultaneously satisfy:
① The product appears clearly, fully, and without obstruction;
② The shot is stable and continuous with no cuts or transitions;
③ The host is focused on explaining or demonstrating the product.
When combining multiple segments, ensure they cover \
**different angles or states** to avoid redundancy; \
prioritize the highest-quality, cover-worthy segment as the opening.

Reason step by step inside the <think> tag using the following structure:

[Quality-guided Global Scan] Briefly describe the overall video pattern \
in 1–2 sentences. Then scan for candidate segments and evaluate each one:
C1: <time> MM:SS.xx-MM:SS.xx </time>
<eval> One concise sentence on clip quality. </eval>
C2: <time> MM:SS.xx-MM:SS.xx </time>
<eval> One concise sentence on clip quality. </eval>
(add more Ci as needed)

[Budgeted Set Composition] In 1–3 sentences, explain which clips are \
selected to form the final 10-second highlight and why unselected \
clips are excluded.

Then output a JSON inside the <answer> tag. Each segment in the \
clips array must include: clip_id, timestamp (format MM:SS.xx-MM:SS.xx), \
description, and core_product.\
"""

SYSTEM_PROMPT = '''
You are an expert video analyst specializing in identifying highlight segments from live-streaming e-commerce videos. When analyzing videos, think deeply like a human, using natural inner-monologue expressions such as 'let me think', 'wait', 'Hmm', 'oh I see', 'let me take a closer look', etc. After examining visual clues, connect your observations to decisions through deeper reasoning. When you spot a potential issue with your initial selection—such as a transition, redundant angles, or duration violation—pause, reflect, and revise your choice before finalizing. Self-reflection and correction during reasoning are encouraged.
'''
