"""Shared helpers for QCR inference: timestamp/JSON utils + ffmpeg clip extraction."""
import os
import json
import subprocess
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_time_str(time_str: str):
    """
    将 MM:SS, HH:MM:SS 或 SS 格式的时间字符串转换为秒数。
    支持浮点秒数，例如 "00:01.5"。
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)

    parts = str(time_str).split(':')
    if len(parts) == 3:
        # HH:MM:SS
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        # MM:SS
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 1:
        # SS
        dot_parts = parts[0].split('.')
        if len(dot_parts) == 3:
            # MM.SS.xx
            return int(dot_parts[0]) * 60 + float(f"{dot_parts[1]}.{dot_parts[2]}")
        elif len(dot_parts) <= 2:
            # SS or SS.xx
            return float(parts[0])
        else:
            raise ValueError(f"Invalid Timestamp: {time_str}")
    else:
        raise ValueError(f"Invalid Timestamp: {time_str}")


def save_json(data, output_path):
    """
    保存当前所有处理结果到JSON文件。
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _execute_media_command(
    command: List[str],
    description: str = "执行命令",
    input_file_context: str = None, # 原始输入文件路径，用于更详细的错误报告
    capture_stdout: bool = False, # 是否捕获标准输出 (True for ffprobe, False for ffmpeg)
    encoding: str = None # 解码输出时使用的编码
) -> Union[subprocess.CompletedProcess, None]:
    """
    通用的 FFmpeg/FFprobe 命令执行器。
    封装了 subprocess.run 调用、标准输出/错误流捕获、以及常见的 CalledProcessError 和 FileNotFoundError 处理。
    """
    process_kwargs = {
        "check": True,  # 遇到非零退出码时抛出 CalledProcessError
        "text": True,   # 将输出解码为文本
    }

    if capture_stdout:
        # 同时捕获 stdout 和 stderr 到 result.stdout 和 result.stderr
        process_kwargs["capture_output"] = True
    else:
        # 对于不需要捕获 stdout 的情况 (如 ffmpeg)，丢弃 stdout，但捕获 stderr 以便调试
        process_kwargs["stdout"] = subprocess.DEVNULL
        process_kwargs["stderr"] = subprocess.PIPE

    if encoding:
        process_kwargs["encoding"] = encoding

    try:
        # logging.debug(f"Executing command for '{description}': {' '.join(command)}") # 可选：用于调试
        result = subprocess.run(command, **process_kwargs)
        return result
    except FileNotFoundError:
        program_name = command[0]
        logger.critical(f"Fatal Error: Command '{program_name}' not found. Please ensure {program_name} is installed and added to the system PATH environment variable.")
        raise
    except subprocess.CalledProcessError as e:
        base_input_filename = os.path.basename(input_file_context) if input_file_context else "（未知文件）"
        logger.error(f"'{description}' failed. File: '{base_input_filename}'. Exit code: {e.returncode}")
        logger.error(f"Command details: {' '.join(e.cmd)}")
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr.strip()}")
        else:
            logger.error("Error output is empty.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during '{description}': {e}", exc_info=True)
        raise


def clip_video_from_segments(
    input_video_path: str,
    segments: List[Tuple[float, float]],
    output_video_path: str,
    overwrite: bool = True
) -> str:
    """
    使用 FFmpeg 从单个视频中剪辑出多个时间段，并将它们合并成一个新视频。
    """
    if not os.path.isfile(input_video_path):
        logger.error(f"Error: Input file not found -> {input_video_path}")
        raise FileNotFoundError(f"❌ 错误：输入文件未找到 -> {input_video_path}")
        
    if not segments:
        raise ValueError("❌ 错误：时间段列表 'segments' 不能为空。")
    for i, (start, end) in enumerate(segments):
        if start >= end:
            raise ValueError(f"❌ 错误：第 {i+1} 个时间段 `({start}, {end})` 无效，开始时间必须小于结束时间。")
    
    filter_parts = []
    concat_inputs = ""
    
    for i, (start, end) in enumerate(segments):
        # [0:v] -> 输入文件的视频流
        # trim -> 剪辑指定时间; setpts -> 重置时间戳以确保无缝拼接
        filter_parts.append(f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]")
        concat_inputs += f"[v{i}]"

    # 将所有滤镜链组合起来
    filter_script = ";".join(filter_parts)
    num_segments = len(segments)
    
    # 完整的 filter_complex 字符串，注意 a=0 表示没有音频流
    final_filter_complex = f"{filter_script};{concat_inputs}concat=n={num_segments}:v=1:a=0[outv]"

    # 3. 构建完整的 FFmpeg 命令列表
    command = ['ffmpeg']
    if overwrite:
        command.append('-y')

    command.extend([
        '-i', input_video_path,
        '-filter_complex', final_filter_complex,
        '-map', '[outv]',         # 只映射处理后的视频流
        '-c:v', 'libx264',        # 使用兼容性好的 H.264 编码
        '-preset', 'fast',      # 编码速度与质量的平衡
        '-an',                    # 明确告诉 FFmpeg 输出文件不要包含音频 (No Audio)
        output_video_path
    ])

    # 4. 执行命令
    logger.info(f"开始从 '{os.path.basename(input_video_path)}' 剪辑 {len(segments)} 个【视频】片段...")
    _execute_media_command(
        command,
        description="剪辑并合并视频片段 (仅视频)",
        input_file_context=input_video_path,
        capture_stdout=False
    )
    
    logger.info(f"视频剪辑成功！新视频已保存至: {output_video_path}")
    return output_video_path


def is_video_valid(file_path: str) -> bool:
    """通过 ffprobe 快速检查视频文件是否有效且完整。"""
    # 1. 检查文件是否存在且大小大于一个基本阈值（例如1KB），避免对空文件执行ffprobe
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
        return False
        
    # 2. 使用ffprobe检查视频流
    # 确保ffprobe在您的系统PATH中，或者提供完整路径
    command = [
        "ffprobe",  # 或者 "/path/to/your/ffprobe"
        "-v", "error",
        "-select_streams", "v:0",  # 只选择视频流
        "-show_entries", "stream=codec_name",  # 请求显示视频流的编解码器名称
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ]
    try:
        # 运行命令，如果ffprobe成功找到视频流并输出了codec_name，则result.stdout会有内容
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # 如果stdout不为空，说明视频流有效
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果ffprobe命令失败或未找到，说明文件损坏或不是有效的视频
        return False

def clip_single_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    crop_bbox: Optional[Union[Dict[str, int], List[int]]] = None, # 接受字典或列表
    overwrite: bool = True
) -> str:
    """
    使用 FFmpeg 从输入视频中剪辑单个时间段，并可选择性地应用裁剪。
    此函数智能处理两种 bbox 格式。

    Args:
        input_path (str): 输入视频文件的路径。
        output_path (str): 输出剪辑视频文件的路径。
        start_time (float): 剪辑的开始时间（秒）。
        end_time (float): 剪辑的结束时间（秒）。
        crop_bbox (Optional[Union[Dict[str, int], List[int]]]): 
            可选的裁剪框，支持两种格式：
            1. 字典: {'x': int, 'y': int, 'w': int, 'h': int} (FFmpeg crop filter format)
            2. 列表: [x1, y1, x2, y2] (坐标点格式)
            如果为 None，则不进行裁剪。
        overwrite (bool): 如果输出文件已存在，是否覆盖。

    Returns:
        str: 输出视频文件的路径。
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found -> {input_path}")
        raise FileNotFoundError(f"输入文件未找到 -> {input_path}")
    
    if start_time >= end_time:
        raise ValueError(f"Invalid segment `({start_time}, {end_time})`: start time must be less than end time.")

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    command = ['ffmpeg']
    if overwrite:
        command.append('-y')

    command.extend(['-i', input_path, '-ss', str(start_time), '-to', str(end_time)])

    # --- 智能 BBox 转换和处理逻辑 ---
    if crop_bbox:
        ffmpeg_crop_params = None
        # 检查是否为字典格式 {'x': ..., 'w': ...}
        if isinstance(crop_bbox, dict) and all(k in crop_bbox for k in ['x', 'y', 'w', 'h']):
            ffmpeg_crop_params = crop_bbox
        # 检查是否为列表格式 [x1, y1, x2, y2]
        elif isinstance(crop_bbox, (list, tuple)) and len(crop_bbox) == 4:
            try:
                x1, y1, x2, y2 = map(int, crop_bbox)
                ffmpeg_crop_params = {
                    'x': x1,
                    'y': y1,
                    'w': x2 - x1,
                    'h': y2 - y1
                }
                # 检查计算出的宽高是否为正数
                if ffmpeg_crop_params['w'] <= 0 or ffmpeg_crop_params['h'] <= 0:
                     logger.warning(f"Invalid BBox coordinates {crop_bbox} result in non-positive width/height. Skipping crop.")
                     ffmpeg_crop_params = None

            except (ValueError, TypeError):
                logger.warning(f"BBox list {crop_bbox} contains non-numeric values. Skipping crop.")
                ffmpeg_crop_params = None
        else:
            logger.warning(f"Unsupported bbox format: {crop_bbox}. Skipping crop.")

        if ffmpeg_crop_params:
            p = ffmpeg_crop_params
            crop_filter = f"crop={p['w']}:{p['h']}:{p['x']}:{p['y']}"
            command.extend(['-vf', crop_filter])
    # --- BBox 处理结束 ---

    command.extend(['-c:v', 'libx264', '-preset', 'fast', '-an', output_path])

    action_desc = "Clipping single video segment"
    if 'crop=' in ' '.join(command):
        action_desc += " with cropping"
        
    logger.info(f"Starting to {action_desc.lower()} from '{os.path.basename(input_path)}'...")
    _execute_media_command(
        command,
        description=action_desc,
        input_file_context=input_path,
        capture_stdout=False
    )
    
    logger.info(f"Successfully created clip: {output_path}")
    return output_path