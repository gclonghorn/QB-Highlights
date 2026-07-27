import json
import re
import math
import logging
from typing import List

import numpy as np
from swift.rewards import ORM, orms

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 解析工具函数
# ══════════════════════════════════════════════════════════════════════════════

def parse_time_str_to_seconds(time_str: str):
    """MM:SS.xx → 秒"""
    try:
        parts = time_str.split(':')
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
        return None
    except (ValueError, IndexError):
        return None


def get_timestamps_from_str(json_str: str):
    """从 JSON 字符串解析 clips 时间戳"""
    try:
        if json_str.strip().startswith("```json"):
            json_str = json_str.strip()[7:].strip()
            if json_str.endswith("```"):
                json_str = json_str[:-3].strip()
        data  = json.loads(json_str)
        clips = data.get("clips", [])
        timestamps_sec = []
        for clip in clips:
            ts_str = clip.get("timestamp")
            if not ts_str:
                continue
            parts = ts_str.split('-')
            if len(parts) != 2:
                continue
            start_sec = parse_time_str_to_seconds(parts[0].strip())
            end_sec   = parse_time_str_to_seconds(parts[1].strip())
            if start_sec is not None and end_sec is not None and start_sec < end_sec:
                timestamps_sec.append([start_sec, end_sec])
        return timestamps_sec
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def get_timestamps_from_deepseek_format(text: str):
    """从 <answer>...</answer> 标签内解析时间戳"""
    if not isinstance(text, str):
        return None
    match          = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    answer_content = match.group(1).strip() if match else text.strip()
    answer_content = re.sub(r'^```json\s*', '', answer_content)
    answer_content = re.sub(r'\s*```$',     '', answer_content)
    return get_timestamps_from_str(answer_content)


def get_time_tags_from_think(text: str) -> List[List[float]]:
    """从 think 标签内提取所有 <time>xx-xx</time> 时间段"""
    if not isinstance(text, str):
        return []
    THINK_OPEN  = "<" + "think" + ">"
    THINK_CLOSE = "<" + "/" + "think" + ">"
    TIME_OPEN   = "<" + "time" + ">"
    TIME_CLOSE  = "<" + "/" + "time" + ">"

    think_pattern = re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE)
    think_match   = re.search(think_pattern, text, re.DOTALL)
    think_content = think_match.group(1) if think_match else text

    time_pattern = re.escape(TIME_OPEN) + r"(.*?)" + re.escape(TIME_CLOSE)
    time_matches  = re.findall(time_pattern, think_content, re.DOTALL)

    results = []
    for ts_str in time_matches:
        ts_str = ts_str.strip()
        parts  = ts_str.split("-")
        if len(parts) != 2:
            continue
        start_sec = parse_time_str_to_seconds(parts[0].strip())
        end_sec   = parse_time_str_to_seconds(parts[1].strip())
        if start_sec is not None and end_sec is not None and end_sec > start_sec:
            results.append([start_sec, end_sec])
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 几何工具函数
# ══════════════════════════════════════════════════════════════════════════════

def calculate_union_of_segments(segments: List[List[float]]) -> List[List[float]]:
    if not segments:
        return []
    segments.sort(key=lambda x: x[0])
    merged = [segments[0][:]]
    for current in segments[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current[:])
    return merged


def total_duration(segments: List[List[float]]) -> float:
    return sum(e - s for s, e in segments)


def calculate_ngiou(pred_ts: List[float], gt_ts: List[float]) -> float:
    pred_start, pred_end = min(pred_ts), max(pred_ts)
    gt_start,   gt_end   = min(gt_ts),   max(gt_ts)
    inter_len = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    pred_len  = pred_end - pred_start
    gt_len    = gt_end   - gt_start
    union_len = pred_len + gt_len - inter_len
    if union_len == 0:
        return 1.0 if pred_len == 0 and gt_len == 0 else 0.0
    iou   = inter_len / union_len
    c_len = max(pred_end, gt_end) - min(pred_start, gt_start)
    if c_len == 0:
        return 1.0
    giou = iou - (c_len - union_len) / c_len
    return (giou + 1) / 2.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. r-global
# ══════════════════════════════════════════════════════════════════════════════

class GlobalQualityReward(ORM):
    """
    所有预测片段的并集 vs 所有 GT 片段的并集 → Union IoU
    使用 <answer> 标签解析
    """
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gt_solutions_batch = kwargs.get('solution')
        if gt_solutions_batch is None:
            return [0.0] * len(completions)

        batch_rewards = []
        for i, completion_str in enumerate(completions):
            pred_ts = get_timestamps_from_deepseek_format(completion_str)
            gt_ts   = get_timestamps_from_deepseek_format(gt_solutions_batch[i])
            if pred_ts is None or gt_ts is None:
                batch_rewards.append(0.0)
                continue
            batch_rewards.append(self._global_iou(pred_ts, gt_ts))
        return batch_rewards

    def _global_iou(self, pred_list, gt_list):
        if not pred_list and not gt_list:
            return 1.0
        pred_union = calculate_union_of_segments(pred_list[:])
        gt_union   = calculate_union_of_segments(gt_list[:])
        total_inter = 0.0
        for p in pred_union:
            for g in gt_union:
                total_inter += max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
        total_union = (total_duration(pred_union)
                       + total_duration(gt_union)
                       - total_inter)
        return total_inter / total_union if total_union > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. r-local
# ══════════════════════════════════════════════════════════════════════════════

class LocalQualityReward(ORM):
    """
    逐片段：NGIoU × exp(-k × (δs + δe) / min(gt_dur, cap))
    使用 <answer> 标签解析
    """
    def __init__(self, boundary_penalty_k: float = 1.0,
                 duration_cap: float = 5.0, **kwargs):
        super().__init__(**kwargs)
        self.k            = boundary_penalty_k
        self.duration_cap = duration_cap

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        gt_solutions_batch = kwargs.get('solution')
        if gt_solutions_batch is None:
            return [0.0] * len(completions)

        batch_rewards = []
        for i, completion_str in enumerate(completions):
            pred_ts = get_timestamps_from_deepseek_format(completion_str)
            gt_ts   = get_timestamps_from_deepseek_format(gt_solutions_batch[i])
            if pred_ts is None or gt_ts is None:
                batch_rewards.append(0.0)
                continue
            batch_rewards.append(self._compute(pred_ts, gt_ts))
        return batch_rewards

    def _compute(self, pred_list, gt_list):
        if not pred_list and not gt_list:
            return 1.0
        num_pairs   = max(len(pred_list), len(gt_list))
        total_score = 0.0
        for i in range(num_pairs):
            pred_ts = pred_list[i] if i < len(pred_list) else None
            gt_ts   = gt_list[i]   if i < len(gt_list)   else None
            if pred_ts is None or gt_ts is None:
                continue
            ngiou   = calculate_ngiou(pred_ts, gt_ts)
            gt_dur  = gt_ts[1] - gt_ts[0]
            eff_dur = min(gt_dur, self.duration_cap)
            if eff_dur > 1e-6:
                ds = abs(pred_ts[0] - gt_ts[0]) / eff_dur
                de = abs(pred_ts[1] - gt_ts[1]) / eff_dur
            else:
                val = 0.0 if (pred_ts[1] - pred_ts[0]) < 1e-6 else 1.0
                ds  = de = val
            total_score += ngiou * math.exp(-self.k * (ds + de))
        return total_score / num_pairs


# ══════════════════════════════════════════════════════════════════════════════
# 3. r-format
# ══════════════════════════════════════════════════════════════════════════════

class FormatReward(ORM):
    """
    格式奖励（总分 1.0）：
      0.2  think 在 answer 之前（顺序正确）
      0.4  think 内 <time>/<eval> 严格交替且非空
      0.4  answer 标签存在
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        batch_rewards = []
        for text in completions:
            score = 0.0
            try:
                THINK_PAT  = r'<' + r'think>(.*?)</' + r'think>'
                ANSWER_PAT = r'<answer>(.*?)</answer>'
                think_m    = re.search(THINK_PAT,  text, re.DOTALL)
                answer_m   = re.search(ANSWER_PAT, text, re.DOTALL)

                # 0.2: 顺序正确
                if think_m and answer_m and think_m.end() <= answer_m.start():
                    score += 0.2

                # 0.4: think 内部 time/eval 交替
                if think_m:
                    inner = think_m.group(1)
                    tags  = re.findall(r'<(time|eval)>', inner)
                    if len(tags) >= 2 and len(tags) % 2 == 0:
                        alternating = all(
                            tags[j] == 'time' and tags[j + 1] == 'eval'
                            for j in range(0, len(tags), 2)
                        )
                        times = re.findall(r'<time>(.*?)</time>', inner, re.DOTALL)
                        evals = re.findall(r'<eval>(.*?)</eval>', inner, re.DOTALL)
                        if (alternating
                                and all(t.strip() for t in times)
                                and all(e.strip() for e in evals)):
                            score += 0.4

                # 0.4: answer 标签存在
                if answer_m:
                    score += 0.4

            except Exception:
                score = 0.0
            batch_rewards.append(score)
        return batch_rewards


# ══════════════════════════════════════════════════════════════════════════════
# 4. r-ground
# ══════════════════════════════════════════════════════════════════════════════

class GroundingReward(ORM):
    """
    answer 时间戳不脱离 think 中推理过的候选区间。
    coverage(a, t) = |a∩t| / |a|
    reward = mean over answer clips of max_t coverage(a, t) ^ alpha
    """
    def __init__(self, alpha: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [self._compute(c) for c in completions]

    def _coverage(self, ans_ts: List[float], think_ts: List[float]) -> float:
        a_s, a_e = ans_ts
        t_s, t_e = think_ts
        a_len = a_e - a_s
        if a_len <= 1e-6:
            return 0.0
        inter = max(0.0, min(a_e, t_e) - max(a_s, t_s))
        return inter / a_len

    def _compute(self, text: str) -> float:
        think_ts  = get_time_tags_from_think(text)
        answer_ts = get_timestamps_from_deepseek_format(text)
        if not think_ts or not answer_ts:
            return 0.0
        scores = [
            max(self._coverage(a, t) for t in think_ts) ** self.alpha
            for a in answer_ts
        ]
        return float(np.clip(sum(scores) / len(scores), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 5. r-comp
# ══════════════════════════════════════════════════════════════════════════════

class CompositionReward(ORM):
    """
    Gap + Stage Coverage 版本：
    gap_score      : 相邻片段不要太近（local anti-crowding）
    coverage_score : 片段不要都集中在同一时间阶段（global stage coverage）

    diversity = weight_gap * gap_score + weight_cov * coverage_score
    """
    def __init__(
        self,
        target_gap:            float = 5.0,
        num_bins:              int   = 3,
        weight_gap:            float = 0.6,
        weight_cov:            float = 0.4,
        single_clip_cov_mode:  str   = "normal",
        single_clip_cov_value: float = 0.33,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_gap            = target_gap
        self.num_bins              = num_bins
        self.weight_gap            = weight_gap
        self.weight_cov            = weight_cov
        self.single_clip_cov_mode  = single_clip_cov_mode
        self.single_clip_cov_value = single_clip_cov_value
        logger.info(
            f"CompositionReward | target_gap={target_gap}s, "
            f"num_bins={num_bins}, w_gap={weight_gap}, w_cov={weight_cov}, "
            f"single_clip_cov_mode={single_clip_cov_mode}, "
            f"single_clip_cov_value={single_clip_cov_value}"
        )

    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        # 支持多种字段名
        video_durations = (
            kwargs.get("video_durations")
            or kwargs.get("durations")
            or kwargs.get("video_duration")
        )

        # 单个 float → 扩展为列表
        if isinstance(video_durations, (int, float)):
            video_durations = [float(video_durations)] * len(completions)

        if video_durations is None:
            logger.warning(
                "CompositionReward: missing video_duration, return 0.0"
            )
            return [0.0] * len(completions)

        if len(video_durations) != len(completions):
            logger.warning(
                f"CompositionReward: len(video_durations)="
                f"{len(video_durations)} != len(completions)={len(completions)}"
            )
            return [0.0] * len(completions)

        return [
            self._compute(c, float(video_durations[i]))
            for i, c in enumerate(completions)
        ]

    # ── gap 分数 ──────────────────────────────────────────────────────────
    def _gap_score(self, pred_timestamps: List[List[float]]) -> float:
        """相邻片段 gap 分数；单片段返回 1.0（gap 约束不适用）"""
        if len(pred_timestamps) <= 1:
            return 1.0
        pts    = sorted(pred_timestamps, key=lambda x: x[0])
        scores = []
        for i in range(len(pts) - 1):
            gap   = max(0.0, pts[i + 1][0] - pts[i][1])
            score = min(gap / (self.target_gap + 1e-6), 1.0)
            scores.append(score)
        return float(sum(scores) / len(scores)) if scores else 1.0

    # ── bin 覆盖 ──────────────────────────────────────────────────────────
    def _clip_to_bins(
        self,
        ts: List[float],
        video_duration: float,
    ) -> set:
        """返回该片段覆盖到的 bin 索引集合"""
        s, e = ts
        if video_duration <= 1e-6 or e <= s:
            return set()
        s = max(0.0, min(s, video_duration))
        e = max(0.0, min(e, video_duration))
        if e <= s:
            return set()
        bin_len = video_duration / self.num_bins
        if bin_len <= 1e-6:
            return set()
        covered = set()
        for b in range(self.num_bins):
            b_s   = b * bin_len
            b_e   = (b + 1) * bin_len
            inter = max(0.0, min(e, b_e) - max(s, b_s))
            if inter > 1e-6:
                covered.add(b)
        return covered

    def _coverage_score(
        self,
        pred_timestamps: List[List[float]],
        video_duration:  float,
    ) -> float:
        """stage coverage 分数 = 覆盖到的 unique bins / num_bins"""
        if not pred_timestamps or video_duration <= 1e-6:
            return 0.0
        if len(pred_timestamps) == 1 and self.single_clip_cov_mode == "fixed":
            return float(np.clip(self.single_clip_cov_value, 0.0, 1.0))
        covered_bins = set()
        for ts in pred_timestamps:
            covered_bins |= self._clip_to_bins(ts, video_duration)
        return float(len(covered_bins) / max(self.num_bins, 1))

    # ── 单条计算 ──────────────────────────────────────────────────────────
    def _compute(self, text: str, video_duration: float) -> float:
        pred_timestamps = get_timestamps_from_deepseek_format(text)
        if not pred_timestamps:
            return 0.0

        valid_preds = [
            [float(ts[0]), float(ts[1])]
            for ts in pred_timestamps
            if (isinstance(ts, (list, tuple))
                and len(ts) == 2
                and ts[1] - ts[0] > 1e-6)
        ]
        if not valid_preds:
            return 0.0

        gap_score = self._gap_score(valid_preds)
        cov_score = self._coverage_score(valid_preds, video_duration)
        diversity = self.weight_gap * gap_score + self.weight_cov * cov_score
        return float(np.clip(diversity, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# 注册
# ══════════════════════════════════════════════════════════════════════════════

orms['r-global']          = GlobalQualityReward
orms['r-local'] = LocalQualityReward
orms['r-format']                              = FormatReward
orms['r-ground']                              = GroundingReward
orms['r-comp']                       = CompositionReward

