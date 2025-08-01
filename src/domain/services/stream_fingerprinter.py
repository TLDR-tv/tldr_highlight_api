"""Stream fingerprinting service for unique streamer identification."""
import hashlib
from urllib.parse import urlparse
from typing import Optional


class StreamFingerprinter:
    """
    Service to generate unique fingerprints for streamers.
    
    The fingerprint identifies a unique streamer on a platform, allowing us to:
    1. Group all streams/highlights from the same streamer
    2. Provide access to all content from a specific streamer
    3. Track usage per streamer without managing their accounts
    """
    
    @staticmethod
    def generate_streamer_fingerprint(
        organization_id: str,
        platform: str,
        streamer_id: str
    ) -> str:
        """
        Generate a unique fingerprint for a streamer.
        
        This creates a consistent identifier for a streamer that:
        - Is unique per organization (multi-tenant isolation)
        - Is unique per platform (youtube, twitch, etc.)
        - Is unique per streamer on that platform
        
        Args:
            organization_id: The organization ID (for tenant isolation)
            platform: The streaming platform (e.g., "twitch", "youtube", "custom")
            streamer_id: The unique identifier for the streamer on that platform
        
        Returns:
            A 16-character hex fingerprint
        """
        # Normalize inputs
        components = [
            f"org:{organization_id}",
            f"platform:{platform.lower()}",
            f"streamer:{streamer_id.lower()}"
        ]
        
        # Create stable fingerprint
        fingerprint_data = '|'.join(components).encode('utf-8')
        full_hash = hashlib.sha256(fingerprint_data).hexdigest()
        
        # Return first 16 characters for a manageable fingerprint
        return full_hash[:16]
    
    @staticmethod
    def extract_streamer_info(stream_url: str) -> dict[str, Optional[str]]:
        """
        Extract streamer information from stream URL.
        
        Returns:
            Dict with platform and streamer_id
        """
        parsed = urlparse(stream_url)
        path_parts = [p for p in parsed.path.strip('/').split('/') if p]
        
        result = {
            'platform': None,
            'streamer_id': None,
            'display_name': None  # Human-readable name if available
        }
        
        # Identify platform and extract streamer info
        if parsed.netloc in ('twitch.tv', 'www.twitch.tv'):
            result['platform'] = 'twitch'
            if path_parts and path_parts[0] not in ('directory', 'videos', 'clip'):
                # twitch.tv/username
                result['streamer_id'] = path_parts[0].lower()
                result['display_name'] = path_parts[0]
                
        elif parsed.netloc in ('kick.com', 'www.kick.com'):
            result['platform'] = 'kick'
            if path_parts and path_parts[0] not in ('category', 'clips'):
                # kick.com/username
                result['streamer_id'] = path_parts[0].lower()
                result['display_name'] = path_parts[0]
                
        elif 'youtube.com' in parsed.netloc:
            result['platform'] = 'youtube'
            if len(path_parts) >= 2:
                if path_parts[0] == 'channel':
                    # youtube.com/channel/CHANNEL_ID
                    result['streamer_id'] = path_parts[1]
                elif path_parts[0] == 'c':
                    # youtube.com/c/customname
                    result['streamer_id'] = f"c:{path_parts[1].lower()}"
                    result['display_name'] = path_parts[1]
                elif path_parts[0] == '@':
                    # youtube.com/@handle
                    result['streamer_id'] = f"@{path_parts[1].lower()}"
                    result['display_name'] = f"@{path_parts[1]}"
                elif path_parts[0] == 'user':
                    # youtube.com/user/username
                    result['streamer_id'] = f"user:{path_parts[1].lower()}"
                    result['display_name'] = path_parts[1]
            elif path_parts and path_parts[0] == 'watch':
                # For video URLs, we'd need to make an API call to get channel info
                # For now, we'll use the video ID as a fallback
                video_id = None
                if '?v=' in stream_url:
                    video_id = stream_url.split('?v=')[1].split('&')[0]
                if video_id:
                    result['streamer_id'] = f"video:{video_id}"
                    
        else:
            # Custom platform - use domain and first path component
            result['platform'] = parsed.netloc.replace('www.', '').split('.')[0]
            if path_parts:
                # Assume first meaningful path component is streamer
                meaningful_parts = [p for p in path_parts if p not in ('live', 'stream', 'watch', 'user')]
                if meaningful_parts:
                    result['streamer_id'] = meaningful_parts[0].lower()
                    result['display_name'] = meaningful_parts[0]
        
        return result
    
    @staticmethod
    def generate_fingerprint_from_url(
        stream_url: str,
        organization_id: str,
        platform_override: Optional[str] = None,
        streamer_id_override: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a streamer fingerprint from a stream URL.
        
        Args:
            stream_url: The stream URL
            organization_id: The organization ID
            platform_override: Override the detected platform
            streamer_id_override: Override the detected streamer ID
        
        Returns:
            Fingerprint if streamer can be identified, None otherwise
        """
        # Extract streamer info from URL
        info = StreamFingerprinter.extract_streamer_info(stream_url)
        
        # Use overrides if provided
        platform = platform_override or info['platform']
        streamer_id = streamer_id_override or info['streamer_id']
        
        # Generate fingerprint if we have both platform and streamer ID
        if platform and streamer_id:
            return StreamFingerprinter.generate_streamer_fingerprint(
                organization_id=organization_id,
                platform=platform,
                streamer_id=streamer_id
            )
        
        return None