"""Shared helpers for QCR inference: timestamp/JSON utils + ffmpeg clip extraction."""
import json
import logging
import os
import subprocess
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def parse_time_str(time_str):
    """Parse HH:MM:SS / MM:SS / SS (float allowed) into seconds."""
    if isinstance(time_str, (int, float)):
        return float(time_str)

    parts = str(time_str).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 1:
        dot_parts = parts[0].split('.')
        if len(dot_parts) == 3:
            # MM.SS.xx
            return int(dot_parts[0]) * 60 + float(f"{dot_parts[1]}.{dot_parts[2]}")
        elif len(dot_parts) <= 2:
            return float(parts[0])
        else:
            raise ValueError(f"Invalid Timestamp: {time_str}")
    else:
        raise ValueError(f"Invalid Timestamp: {time_str}")


def save_json(data, output_path):
    """Dump `data` as pretty JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _execute_media_command(
    command: List[str],
    description: str = "media command",
    input_file_context: str = None,
    capture_stdout: bool = False,
    encoding: str = None,
) -> Union[subprocess.CompletedProcess, None]:
    """Thin wrapper around subprocess.run for ffmpeg / ffprobe calls."""
    process_kwargs = {
        "check": True,
        "text": True,
    }

    if capture_stdout:
        process_kwargs["capture_output"] = True
    else:
        # ffmpeg: drop stdout, keep stderr for debugging
        process_kwargs["stdout"] = subprocess.DEVNULL
        process_kwargs["stderr"] = subprocess.PIPE

    if encoding:
        process_kwargs["encoding"] = encoding

    try:
        result = subprocess.run(command, **process_kwargs)
        return result
    except FileNotFoundError:
        program_name = command[0]
        logger.critical(
            f"Fatal Error: Command '{program_name}' not found. "
            f"Please ensure {program_name} is installed and on PATH."
        )
        raise
    except subprocess.CalledProcessError as e:
        base = os.path.basename(input_file_context) if input_file_context else "(unknown file)"
        logger.error(f"'{description}' failed. File: '{base}'. Exit code: {e.returncode}")
        logger.error(f"Command details: {' '.join(e.cmd)}")
        if e.stderr:
            logger.error(f"Error output:\n{e.stderr.strip()}")
        else:
            logger.error("Error output is empty.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during '{description}': {e}", exc_info=True)
        raise


def clip_video_from_segments(
    input_video_path: str,
    segments: List[Tuple[float, float]],
    output_video_path: str,
    overwrite: bool = True,
) -> str:
    """Extract multiple time segments from a single video and concat them."""
    if not os.path.isfile(input_video_path):
        logger.error(f"Error: Input file not found -> {input_video_path}")
        raise FileNotFoundError(f"Input file not found -> {input_video_path}")

    if not segments:
        raise ValueError("'segments' must not be empty.")
    for i, (start, end) in enumerate(segments):
        if start >= end:
            raise ValueError(
                f"Segment #{i+1} `({start}, {end})` is invalid: start must be < end."
            )

    filter_parts = []
    concat_inputs = ""

    for i, (start, end) in enumerate(segments):
        # [0:v] = input video stream; trim + setpts for seamless concat.
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
        )
        concat_inputs += f"[v{i}]"

    filter_script = ";".join(filter_parts)
    num_segments = len(segments)

    # a=0 -> no audio stream in the concat filter
    final_filter_complex = (
        f"{filter_script};{concat_inputs}concat=n={num_segments}:v=1:a=0[outv]"
    )

    command = ['ffmpeg']
    if overwrite:
        command.append('-y')

    command.extend([
        '-i', input_video_path,
        '-filter_complex', final_filter_complex,
        '-map', '[outv]',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-an',
        output_video_path,
    ])

    logger.info(
        f"Extracting {len(segments)} segment(s) from "
        f"'{os.path.basename(input_video_path)}'..."
    )
    _execute_media_command(
        command,
        description="clip & concat segments (video-only)",
        input_file_context=input_video_path,
        capture_stdout=False,
    )

    logger.info(f"Saved concatenated highlight to: {output_video_path}")
    return output_video_path


def is_video_valid(file_path: str) -> bool:
    """Return True if the file exists, is >= 1KB, and has a decodable video stream."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
        return False

    command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path,
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clip_single_video(
    input_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
    crop_bbox: Optional[Union[Dict[str, int], List[int]]] = None,
    overwrite: bool = True,
) -> str:
    """
    Extract a single time segment from a video and optionally spatially crop it.

    Args:
        input_path:  source video path.
        output_path: destination clip path.
        start_time:  segment start (seconds).
        end_time:    segment end (seconds).
        crop_bbox:   optional bbox, either
                     - {'x': int, 'y': int, 'w': int, 'h': int}  (ffmpeg crop format), or
                     - [x1, y1, x2, y2]                          (corner-points format).
                     None -> no cropping.
        overwrite:   pass -y to ffmpeg if True.
    """
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found -> {input_path}")
        raise FileNotFoundError(f"Input file not found -> {input_path}")

    if start_time >= end_time:
        raise ValueError(
            f"Invalid segment `({start_time}, {end_time})`: start must be < end."
        )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command = ['ffmpeg']
    if overwrite:
        command.append('-y')

    command.extend(['-i', input_path, '-ss', str(start_time), '-to', str(end_time)])

    # ---- bbox normalization ----
    if crop_bbox:
        ffmpeg_crop_params = None
        if isinstance(crop_bbox, dict) and all(k in crop_bbox for k in ['x', 'y', 'w', 'h']):
            ffmpeg_crop_params = crop_bbox
        elif isinstance(crop_bbox, (list, tuple)) and len(crop_bbox) == 4:
            try:
                x1, y1, x2, y2 = map(int, crop_bbox)
                ffmpeg_crop_params = {
                    'x': x1,
                    'y': y1,
                    'w': x2 - x1,
                    'h': y2 - y1,
                }
                if ffmpeg_crop_params['w'] <= 0 or ffmpeg_crop_params['h'] <= 0:
                    logger.warning(
                        f"BBox {crop_bbox} yields non-positive w/h. Skipping crop."
                    )
                    ffmpeg_crop_params = None
            except (ValueError, TypeError):
                logger.warning(f"BBox list {crop_bbox} has non-numeric values. Skipping crop.")
                ffmpeg_crop_params = None
        else:
            logger.warning(f"Unsupported bbox format: {crop_bbox}. Skipping crop.")

        if ffmpeg_crop_params:
            p = ffmpeg_crop_params
            crop_filter = f"crop={p['w']}:{p['h']}:{p['x']}:{p['y']}"
            command.extend(['-vf', crop_filter])

    command.extend(['-c:v', 'libx264', '-preset', 'fast', '-an', output_path])

    action_desc = "Clipping single video segment"
    if 'crop=' in ' '.join(command):
        action_desc += " with cropping"

    logger.info(f"Starting to {action_desc.lower()} from '{os.path.basename(input_path)}'...")
    _execute_media_command(
        command,
        description=action_desc,
        input_file_context=input_path,
        capture_stdout=False,
    )

    logger.info(f"Successfully created clip: {output_path}")
    return output_path
