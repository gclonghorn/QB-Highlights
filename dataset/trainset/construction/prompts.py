# -*- coding: utf-8 -*-
"""
Prompt templates for QB-Highlights training-set construction.

The three teacher prompts (candidate selection / composition / reasoning-trace
generation) are transcribed **verbatim from the paper (Appendix B.2)**. The
training-time message templates (system prompt + user prompt for SFT / GRPO)
reproduce exactly what appears in the released `sft_cot_17k.jsonl` /
`grpo_49k.jsonl`.
"""

# ============================================================================
# Teacher pipeline prompts (paper Appendix B.2) — used with Gemini-2.5-Pro
# ============================================================================

# Stage 1: coarse candidate selection (360p). `{query}` and `{video}` are
# substituted with the product-presentation query and the video handle.
CANDIDATE_SELECTION_PROMPT = """\
Identify 1--10 high-quality candidate clips from the input video for downstream highlight composition. \
Select clips whose visual content directly fulfills the {query}, considering the entire {video} when \
making your choice. The goal is to extract candidate segments that are visually strong, semantically \
aligned with the query, and suitable for later composition into a concise highlight. Use visual \
information only and ignore all audio, including speech, music, and sound effects. Determine the \
query-relevant content from the provided query together with the visual content of the video. Each \
selected clip should satisfy the following quality requirements:
1. Query relevance as the visual focus. The clip should clearly fulfill the query, with the query being \
actively shown as the main visual content.
2. Clear and complete visibility. The query-relevant content should remain clearly and completely \
visible throughout the clip, without major occlusion or poor framing.
3. Continuous shot. The clip should come from a single continuous shot, without cuts, abrupt \
transitions, or severe content changes.
4. Strong visual quality. The clip should be stable, sharp, and well lit, with minimal blur, shake, or \
distracting background content.
5. Reasonable clip length. Each clip should be 3--10 seconds long.
6. Content diversity across clips. Different clips should capture distinct visual segments rather than \
adjacent, overlapping, or repetitive moments. Favor clips that provide complementary coverage of the \
query across the video.
The output JSON should include three parts: (1) an analysis of the screening and selection process, \
explaining how the query is interpreted and how clips are selected; (2) the timestamps of the selected \
candidate clips; and (3) a description of the visual content of each selected clip."""

# Stage 2: fine composition (720p). `{query}` and `{candidate segments}` are
# substituted; `{candidate segments}` carries the Stage-1 output.
COMPOSITION_PROMPT = """\
Select and organize the best clips from the {candidate_segments} to form a compact 10-second highlight. \
The final composition should be visually strong, tightly paced, and consistently fulfill the {query} \
throughout. Use visual information only and ignore all audio. All selected clips must fulfill the same \
query. Each selected clip should satisfy the following quality requirements:
1. Query consistency. All clips must fulfill the same query, and the query-relevant content should be \
clearly visible from the beginning of each clip.
2. Sustained quality throughout. Every selected clip and the entire composition should consistently \
satisfy strong visual quality, including stability, sharpness, lighting, clear visibility of the subject, \
and absence of blur, shake, or distracting background content.
3. High information density. Each selected clip should carry substantive query-relevant visual content, \
avoiding static, idle, or low-information moments where little informative change occurs.
4. Content diversity. The selected clips should differ in viewpoint, action, framing, or information \
content while maintaining high overall quality.
5. Logical composition. The clip order should balance quality, diversity, and viewing coherence. Clips \
may be reordered from their original temporal sequence, but they must not overlap or repeat.
6. Duration control. The final composition should be approximately 10 seconds long. Internal trimming \
within a candidate clip is allowed, as long as the extracted segment remains continuous and natural.
Output three parts: 1. An analysis of the screening and selection process, explaining how the final clips \
are chosen and balanced for quality, diversity, and temporal structure. 2. The timestamps of the selected \
clips, including the final trimmed time ranges used from the original video. 3. A description of each \
selected clip, covering the visual content, action, and framing relevant to the query."""

# Stage 4: reasoning-trace generation (17K subset). `{candidate_selection}` and
# `{composition}` carry the Stage-1 / Stage-2 outputs. Run with Qwen3-Max.
REASONING_TRACE_PROMPT = """\
Under the results of {candidate_selection} and {composition}, rewrite the original intermediate reasoning \
into a concise and standardized reasoning trace. The rewritten trace should be compact, structured, and \
method-like, with minimal narrative style.
Output a JSON object with three fields:
1. scan_overview: summarize the overall visual pattern of the video in 1--2 concise sentences, focusing \
on the dominant visual style, camera setup, and the typical way content relevant to the target query \
appears across the video. Do not make final selection judgments here.
2. segment_qualification: for each candidate clip from the candidate selection, provide its timestamp and \
a brief reasoning sentence grounded only in observable visual properties such as clarity, stability, \
framing, visibility of the target query content, and information density. Do not compare clips against \
each other or make any keep/remove decision in this field.
3. budgeted_composition: summarize the final set-level reasoning under the 10-second budget using the \
fine-stage composition result. Explain why the selected clips form an effective final highlight and why \
each unselected candidate is excluded, using concrete reasons such as overlap, weaker framing, lower \
visual quality, lower information density, or limited additional content relevant to the target query.
Keep the language concise and structured. Separate clip-level quality assessment from final budget-aware \
composition reasoning. Avoid long narrative monologues, subjective storytelling, and unsupported candidate \
invention. Preserve the core reasoning logic of the coarse and fine stages while rewriting it."""


# ============================================================================
# Training-time message templates (must match the released JSONL exactly)
# ============================================================================

# System prompt used for the GRPO split (inner-monologue reasoning style).
GRPO_SYSTEM_PROMPT = (
    "You are an expert video analyst specializing in identifying highlight segments "
    "from live-streaming e-commerce videos. When analyzing videos, think deeply like a "
    "human, using natural inner-monologue expressions such as 'let me think', 'wait', "
    "'Hmm', 'oh I see', 'let me take a closer look', etc. After examining visual clues, "
    "connect your observations to decisions through deeper reasoning. When you spot a "
    "potential issue with your initial selection—such as a transition, redundant angles, "
    "or duration violation—pause, reflect, and revise your choice before finalizing. "
    "Self-reflection and correction during reasoning are encouraged."
)

# User prompt for both SFT and GRPO. `{product_block}` is e.g.
# "10cm-12cm Ornamental Fish Assortment ... Product Category: Fish".
TRAIN_USER_PROMPT = """<video>
Your task is to create a 10-second highlight video clip plan (1–3 segments, each ≥2 seconds) for the specified product from a live-stream shopping video.
[{product_block}]

High-quality segments must simultaneously satisfy:
① The product appears clearly, fully, and without obstruction;
② The shot is stable and continuous with no cuts or transitions;
③ The host is focused on explaining or demonstrating the product.
When combining multiple segments, ensure they cover **different angles or states** to avoid redundancy; prioritize the highest-quality, cover-worthy segment as the opening.

Reason step by step inside the <think> tag using the following structure:

[Quality-guided Global Scan] Briefly describe the overall video pattern in 1–2 sentences. Then scan for candidate segments and evaluate each one:
C1: <time> MM:SS.xx-MM:SS.xx </time>
<eval> One concise sentence on clip quality. </eval>
C2: <time> MM:SS.xx-MM:SS.xx </time>
<eval> One concise sentence on clip quality. </eval>
(add more Ci as needed)

[Budgeted Set Composition] In 1–3 sentences, explain which clips are selected to form the final 10-second highlight and why unselected clips are excluded.

Then output a JSON inside the <answer> tag. Each segment in the clips array must include:
clip_id, timestamp (format MM:SS.xx-MM:SS.xx), description, and core_product."""
