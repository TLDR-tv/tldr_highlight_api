from typing import Optional

from src.infrastructure.media.ffmpeg_integration import FFmpegProcessor


class ThumbnailGenerator:
    def __init__(self, ffmpeg_processor: Optional[FFmpegProcessor] = None):
        self.ffmpeg_processor = ffmpeg_processor or FFmpegProcessor()

    async def generate_thumbnail(self, video_path: str, output_dir: str) -> str:
        output_path = f"{output_dir}/thumbnail_{video_path.split('/')[-1]}.jpg"
        await self.ffmpeg_processor.run_ffmpeg_async(
            [
                "-i",
                video_path,
                "-ss",
                "00:00:01.000",
                "-vframes",
                "1",
                output_path,
            ]
        )
        return output_path
