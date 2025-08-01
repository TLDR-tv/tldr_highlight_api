from typing import Optional

from src.infrastructure.media.ffmpeg_integration import FFmpegProcessor


class ClipGenerator:
    def __init__(self, ffmpeg_processor: Optional[FFmpegProcessor] = None):
        self.ffmpeg_processor = ffmpeg_processor or FFmpegProcessor()

    async def generate_clip(
        self, source_path: str, output_dir: str, start_time: float, duration: float
    ) -> str:
        output_path = f"{output_dir}/clip_{start_time}_{duration}.mp4"
        await self.ffmpeg_processor.run_ffmpeg_async(
            [
                "-i",
                source_path,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c",
                "copy",
                output_path,
            ]
        )
        return output_path
