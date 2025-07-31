# Highlights API

The Highlights API provides endpoints for accessing, managing, and downloading AI-detected highlights from processed streams.

## Overview

Highlights are automatically detected segments from streams that represent interesting, important, or entertaining moments. Each highlight:

- Has a confidence score indicating detection certainty
- Contains multi-dimensional scoring (action, emotion, etc.)
- Includes extracted video clips and thumbnails
- Can be accessed via API or signed URLs
- Supports custom metadata and tags

## Endpoints

### Get Highlights Service Status

Check if the highlights service is operational.

```http
GET /api/v1/highlights/status
```

#### Response

```json
{
  "status": "Highlights service operational",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### List Highlights

Get a list of highlights with flexible filtering options.

```http
GET /api/v1/highlights
```

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `stream_id` | integer | No | Filter by specific stream |
| `organization_id` | integer | No | Filter by organization (admin only) |
| `page` | integer | No | Page number (default: 1) |
| `per_page` | integer | No | Items per page (default: 20, max: 100) |
| `sort` | string | No | Sort by: `created_at`, `confidence_score`, `duration`, `start_time` |
| `order` | string | No | Sort order: `asc`, `desc` (default: `desc`) |
| `min_confidence` | float | No | Minimum confidence score (0.0-1.0) |
| `max_confidence` | float | No | Maximum confidence score (0.0-1.0) |
| `min_duration` | integer | No | Minimum duration in seconds |
| `max_duration` | integer | No | Maximum duration in seconds |
| `type` | string | No | Filter by highlight type |
| `tags` | array | No | Filter by tags (comma-separated) |
| `created_after` | datetime | No | Filter highlights created after this date |
| `created_before` | datetime | No | Filter highlights created before this date |

#### Response

```json
{
  "items": [
    {
      "id": 67890,
      "stream_id": 12345,
      "title": "Epic clutch 1v5 ace",
      "description": "Incredible 1v5 clutch in overtime to win the match",
      "start_time": 3600,
      "end_time": 3645,
      "duration": 45,
      "confidence_score": 0.95,
      "type": "gameplay_highlight",
      "video_url": "https://cdn.tldr.tv/highlights/67890/video.mp4",
      "thumbnail_url": "https://cdn.tldr.tv/highlights/67890/thumbnail.jpg",
      "created_at": "2024-01-15T11:00:00Z",
      "tags": ["clutch", "ace", "valorant", "epic"],
      "metadata": {
        "game": "VALORANT",
        "map": "Haven",
        "round": 24,
        "score": "12-11"
      },
      "dimensions": {
        "action_intensity": 0.98,
        "audience_reaction": 0.92,
        "skill_display": 0.96,
        "emotional_peak": 0.89
      },
      "stream_info": {
        "title": "Shroud VALORANT Ranked",
        "platform": "twitch",
        "streamer_name": "shroud"
      }
    },
    {
      "id": 67891,
      "stream_id": 12345,
      "title": "Funny fail moment",
      "description": "Hilarious failed jump attempt",
      "start_time": 4200,
      "end_time": 4230,
      "duration": 30,
      "confidence_score": 0.87,
      "type": "funny_moment",
      "video_url": "https://cdn.tldr.tv/highlights/67891/video.mp4",
      "thumbnail_url": "https://cdn.tldr.tv/highlights/67891/thumbnail.jpg",
      "created_at": "2024-01-15T11:10:00Z",
      "tags": ["funny", "fail", "valorant"],
      "dimensions": {
        "humor": 0.91,
        "audience_reaction": 0.88,
        "unexpectedness": 0.85
      }
    }
  ],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 234,
    "pages": 12,
    "has_next": true,
    "has_prev": false
  },
  "aggregations": {
    "total_duration": 11700,
    "average_confidence": 0.86,
    "by_type": {
      "gameplay_highlight": 89,
      "funny_moment": 56,
      "emotional_moment": 45,
      "educational_segment": 44
    }
  }
}
```

### Get Highlight Details

Get detailed information about a specific highlight.

```http
GET /api/v1/highlights/{highlight_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `highlight_id` | integer | path | Yes | Highlight ID |

#### Response

```json
{
  "id": 67890,
  "stream_id": 12345,
  "title": "Epic clutch 1v5 ace",
  "description": "Incredible 1v5 clutch in overtime to win the match",
  "start_time": 3600,
  "end_time": 3645,
  "duration": 45,
  "confidence_score": 0.95,
  "type": "gameplay_highlight",
  "video_url": "https://cdn.tldr.tv/highlights/67890/video.mp4",
  "thumbnail_url": "https://cdn.tldr.tv/highlights/67890/thumbnail.jpg",
  "gif_url": "https://cdn.tldr.tv/highlights/67890/preview.gif",
  "created_at": "2024-01-15T11:00:00Z",
  "updated_at": "2024-01-15T11:00:00Z",
  "tags": ["clutch", "ace", "valorant", "epic"],
  "metadata": {
    "game": "VALORANT",
    "map": "Haven",
    "round": 24,
    "score": "12-11",
    "weapons": ["vandal", "phantom"],
    "agent": "Jett"
  },
  "dimensions": {
    "action_intensity": {
      "score": 0.98,
      "confidence": 0.95,
      "contributing_factors": ["rapid_kills", "movement_speed", "health_critical"]
    },
    "audience_reaction": {
      "score": 0.92,
      "confidence": 0.88,
      "contributing_factors": ["chat_velocity", "emote_spam", "caps_ratio"]
    },
    "skill_display": {
      "score": 0.96,
      "confidence": 0.91,
      "contributing_factors": ["headshot_ratio", "movement_precision", "game_sense"]
    },
    "emotional_peak": {
      "score": 0.89,
      "confidence": 0.85,
      "contributing_factors": ["voice_excitement", "heart_rate_proxy", "celebration"]
    }
  },
  "analysis": {
    "video_features": {
      "dominant_colors": ["#FF4655", "#0F1923", "#FFFFFF"],
      "motion_intensity": 0.87,
      "scene_changes": 12,
      "face_detected": true,
      "objects_detected": ["weapon", "character", "ui_elements"]
    },
    "audio_features": {
      "peak_loudness": -12.3,
      "speech_ratio": 0.34,
      "music_detected": false,
      "excitement_level": 0.91
    },
    "chat_features": {
      "message_velocity": 145,
      "emote_ratio": 0.67,
      "unique_chatters": 89,
      "sentiment": "very_positive"
    }
  },
  "stream_context": {
    "stream_id": 12345,
    "stream_title": "Shroud VALORANT Ranked",
    "platform": "twitch",
    "streamer": {
      "name": "shroud",
      "id": "37402112",
      "avatar_url": "https://static-cdn.jtvnw.net/jtv_user_pictures/shroud.png"
    },
    "vod_url": "https://twitch.tv/videos/1234567890",
    "vod_timestamp": 3600
  },
  "file_info": {
    "video": {
      "format": "mp4",
      "codec": "h264",
      "resolution": "1920x1080",
      "fps": 60,
      "bitrate": "8000kbps",
      "size_bytes": 45678900
    },
    "thumbnail": {
      "format": "jpg",
      "resolution": "1280x720",
      "size_bytes": 123456
    }
  },
  "engagement": {
    "views": 12345,
    "shares": 89,
    "downloads": 234,
    "feedback": {
      "accurate": 156,
      "inaccurate": 12
    }
  }
}
```

#### Error Responses

- `404 Not Found` - Highlight not found
- `403 Forbidden` - User doesn't have access to this highlight

### Get Highlight Download URL

Get a presigned URL to download the highlight video file.

```http
GET /api/v1/highlights/{highlight_id}/download
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `highlight_id` | integer | path | Yes | Highlight ID |
| `quality` | string | query | No | Video quality: `original`, `1080p`, `720p`, `480p` (default: `original`) |
| `format` | string | query | No | Video format: `mp4`, `webm`, `gif` (default: `mp4`) |

#### Response

```json
{
  "download_url": "https://cdn.tldr.tv/highlights/67890/video.mp4?token=eyJ0eXAi...",
  "expires_at": "2024-01-15T12:00:00Z",
  "file_info": {
    "format": "mp4",
    "quality": "1080p",
    "size_bytes": 45678900,
    "duration": 45
  }
}
```

#### Error Responses

- `404 Not Found` - Highlight not found
- `403 Forbidden` - User doesn't have access
- `400 Bad Request` - Invalid quality or format

### Update Highlight

Update highlight metadata, tags, or title.

```http
PATCH /api/v1/highlights/{highlight_id}
```

#### Request Body

```json
{
  "title": "Updated highlight title",
  "description": "New description",
  "tags": ["new", "tags", "here"],
  "metadata": {
    "custom_field": "custom_value"
  }
}
```

#### Response

Returns the updated highlight object.

#### Error Responses

- `404 Not Found` - Highlight not found
- `403 Forbidden` - User doesn't own this highlight
- `422 Unprocessable Entity` - Invalid data

### Delete Highlight

Permanently delete a highlight and its associated files.

```http
DELETE /api/v1/highlights/{highlight_id}
```

#### Parameters

| Name | Type | Location | Required | Description |
|------|------|----------|----------|-------------|
| `highlight_id` | integer | path | Yes | Highlight ID |

#### Response

```http
204 No Content
```

#### Error Responses

- `404 Not Found` - Highlight not found
- `403 Forbidden` - User doesn't own this highlight

## Highlight Types

The system automatically categorizes highlights into types based on detected content:

| Type | Description | Common Dimensions |
|------|-------------|-------------------|
| `gameplay_highlight` | Impressive gameplay moments | action_intensity, skill_display |
| `funny_moment` | Humorous or unexpected events | humor, unexpectedness |
| `emotional_moment` | Emotional peaks or reactions | emotional_intensity, authenticity |
| `educational_segment` | Informative or tutorial content | educational_value, clarity |
| `achievement` | Game achievements or milestones | significance, rarity |
| `interaction` | Memorable viewer interactions | engagement, authenticity |
| `technical_play` | High-skill technical gameplay | complexity, execution |
| `story_moment` | Narrative or story highlights | narrative_impact, emotion |
| `custom` | User-defined highlight types | varies |

## Dimension Scoring

Each highlight is scored across multiple dimensions:

### Standard Dimensions

| Dimension | Description | Range |
|-----------|-------------|-------|
| `action_intensity` | Level of action/excitement | 0.0-1.0 |
| `audience_reaction` | Viewer engagement level | 0.0-1.0 |
| `emotional_peak` | Emotional intensity | 0.0-1.0 |
| `skill_display` | Technical skill demonstrated | 0.0-1.0 |
| `humor` | Comedy/entertainment value | 0.0-1.0 |
| `educational_value` | Learning/informational content | 0.0-1.0 |
| `narrative_impact` | Story significance | 0.0-1.0 |
| `visual_quality` | Visual appeal/production value | 0.0-1.0 |

### Custom Dimensions

Organizations can define custom dimensions:

```json
{
  "brand_relevance": {
    "description": "How relevant to brand messaging",
    "weight": 0.3,
    "scoring_prompt": "Score based on brand guidelines"
  },
  "sponsor_visibility": {
    "description": "Sponsor product visibility",
    "weight": 0.2,
    "detection_rules": ["logo_visible", "product_mentioned"]
  }
}
```

## Batch Operations

### Export Highlights

Export multiple highlights metadata to CSV or JSON.

```http
POST /api/v1/highlights/export
```

#### Request Body

```json
{
  "highlight_ids": [67890, 67891, 67892],
  "format": "csv",
  "fields": ["id", "title", "confidence_score", "duration", "tags"]
}
```

#### Response

```json
{
  "export_url": "https://cdn.tldr.tv/exports/highlights_export_123.csv",
  "expires_at": "2024-01-15T12:00:00Z",
  "record_count": 3
}
```

### Bulk Update

Update multiple highlights at once.

```http
PATCH /api/v1/highlights/bulk
```

#### Request Body

```json
{
  "highlight_ids": [67890, 67891, 67892],
  "updates": {
    "tags": {
      "add": ["reviewed", "approved"],
      "remove": ["pending"]
    },
    "metadata": {
      "reviewed_by": "john.doe",
      "review_date": "2024-01-15"
    }
  }
}
```

## Content Delivery

### Signed URLs

Generate signed URLs for secure, time-limited access:

```http
POST /api/v1/highlights/{highlight_id}/share
```

#### Request Body

```json
{
  "expires_in_hours": 24,
  "allow_download": false,
  "max_views": 100
}
```

#### Response

```json
{
  "share_url": "https://highlights.tldr.tv/v/67890?token=eyJ0eXAi...",
  "expires_at": "2024-01-16T10:30:00Z",
  "share_id": "shr_abc123"
}
```

### Embed Codes

Get embed codes for highlights:

```http
GET /api/v1/highlights/{highlight_id}/embed
```

#### Response

```json
{
  "iframe": "<iframe src=\"https://embed.tldr.tv/h/67890\" width=\"800\" height=\"450\" frameborder=\"0\" allowfullscreen></iframe>",
  "responsive": "<div style=\"position:relative;padding-bottom:56.25%;\"><iframe src=\"https://embed.tldr.tv/h/67890\" style=\"position:absolute;top:0;left:0;width:100%;height:100%;\" frameborder=\"0\" allowfullscreen></iframe></div>",
  "script": "<script async src=\"https://embed.tldr.tv/embed.js\" data-highlight-id=\"67890\"></script>"
}
```

## Analytics

### Highlight Performance

Track highlight engagement and performance:

```http
GET /api/v1/highlights/{highlight_id}/analytics
```

#### Response

```json
{
  "highlight_id": 67890,
  "period": "last_30_days",
  "metrics": {
    "views": {
      "total": 12345,
      "unique": 8901,
      "by_source": {
        "api": 5678,
        "embed": 4567,
        "share": 2100
      }
    },
    "engagement": {
      "average_watch_percentage": 87.5,
      "completion_rate": 0.76,
      "shares": 234,
      "downloads": 89
    },
    "feedback": {
      "accurate": 156,
      "inaccurate": 12,
      "accuracy_rate": 0.929
    }
  },
  "daily_breakdown": [
    {
      "date": "2024-01-15",
      "views": 456,
      "shares": 12,
      "downloads": 5
    }
    // ... more days
  ]
}
```

## Examples

### Search and Filter Highlights

```python
from tldr_highlight_api import TLDRClient

client = TLDRClient(api_key="tldr_sk_your_api_key")

# Get high-confidence gaming highlights
highlights = client.highlights.list(
    min_confidence=0.9,
    type="gameplay_highlight",
    tags=["epic", "clutch"],
    sort="confidence_score",
    order="desc",
    per_page=10
)

for highlight in highlights.items:
    print(f"{highlight.title} - Score: {highlight.confidence_score}")
    print(f"Duration: {highlight.duration}s")
    print(f"Tags: {', '.join(highlight.tags)}")
    print(f"Video: {highlight.video_url}\n")
```

### Download Highlights

```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({ apiKey: 'tldr_sk_your_api_key' });

// Get download URLs for highlights
async function downloadHighlights(streamId) {
  const highlights = await client.highlights.list({ 
    streamId: streamId,
    minConfidence: 0.8 
  });
  
  for (const highlight of highlights.items) {
    const download = await client.highlights.getDownloadUrl(highlight.id, {
      quality: '1080p',
      format: 'mp4'
    });
    
    console.log(`Downloading: ${highlight.title}`);
    console.log(`URL: ${download.downloadUrl}`);
    console.log(`Size: ${(download.fileInfo.sizeBytes / 1024 / 1024).toFixed(2)} MB`);
    
    // Use fetch or axios to download the file
    // await downloadFile(download.downloadUrl, `${highlight.id}.mp4`);
  }
}
```

### Create Highlight Compilation

```bash
#!/bin/bash

# Get top highlights from multiple streams
stream_ids=(12345 12346 12347)
api_key="tldr_sk_your_api_key"

# Collect highlight video URLs
urls=()
for stream_id in "${stream_ids[@]}"; do
  highlights=$(curl -s -H "X-API-Key: $api_key" \
    "https://api.tldr.tv/api/v1/highlights?stream_id=$stream_id&min_confidence=0.85&per_page=3")
  
  video_urls=$(echo $highlights | jq -r '.items[].video_url')
  urls+=($video_urls)
done

# Download videos
mkdir -p highlights
for i in "${!urls[@]}"; do
  curl -s "${urls[$i]}" -o "highlights/clip_$(printf "%03d" $i).mp4"
done

# Create compilation with ffmpeg
ffmpeg -f concat -safe 0 -i <(for f in highlights/*.mp4; do echo "file '$PWD/$f'"; done) \
  -c copy compilation.mp4
```

## Best Practices

### Performance Optimization

1. **Use Pagination** - Don't fetch all highlights at once
2. **Filter Efficiently** - Use query parameters to reduce results
3. **Cache Results** - Highlights don't change after creation
4. **Batch Operations** - Use bulk endpoints when possible

### Content Management

1. **Tag Consistently** - Use standardized tag taxonomy
2. **Add Metadata** - Enrich highlights with context
3. **Monitor Accuracy** - Track feedback to improve detection
4. **Regular Cleanup** - Delete unwanted highlights to save storage

### Integration Tips

1. **Use Webhooks** - Get notified of new highlights
2. **Implement Caching** - Reduce API calls for static data
3. **Handle Errors** - Implement retry logic for failures
4. **Monitor Usage** - Track API usage against quotas

## Troubleshooting

### Common Issues

1. **Highlights Not Found**
   - Verify stream has completed processing
   - Check confidence threshold isn't too high
   - Ensure proper permissions for the stream

2. **Download URLs Expired**
   - Presigned URLs expire after set time
   - Request new URL when needed
   - Consider implementing refresh logic

3. **Missing Highlight Types**
   - Not all streams produce all types
   - Check dimension scoring for insights
   - Adjust detection thresholds

### Support

For highlight-related issues:

- **Detection accuracy**: ml-support@tldr.tv
- **Technical issues**: support@tldr.tv
- **Feature requests**: product@tldr.tv

---

*See also: [Streams API](./streams.md) | [Content Security](../content_delivery/content_security.md) | [API Overview](./overview.md)*