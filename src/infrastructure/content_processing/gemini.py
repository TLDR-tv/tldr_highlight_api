import asyncio

import google.generativeai as genai

class GeminiCaptionGenerator:
    def __init__(self, api_key: str, model_name: str = "gemini-pro-vision"):
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

    async def generate_caption(self, video_path: str) -> str:
        model = genai.GenerativeModel(self.model_name)
        video_file = await asyncio.to_thread(genai.upload_file, path=video_path)
        response = await model.generate_content_async(["Describe this video", video_file])
        return response.text
