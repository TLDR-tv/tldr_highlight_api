import asyncio
from typing import Optional

import boto3
from botocore.exceptions import NoCredentialsError


class S3Storage:
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region_name = region_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
        )

    async def upload_file(self, file_path: str, object_name: str) -> str:
        try:
            await asyncio.to_thread(
                self.s3_client.upload_file, file_path, self.bucket_name, object_name
            )
            return f"{self.endpoint_url}/{self.bucket_name}/{object_name}"
        except NoCredentialsError:
            raise Exception("AWS credentials not found.")
