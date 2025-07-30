# Quick Start Guide

Get up and running with the TL;DR Highlight API in under 10 minutes.

## 🚀 5-Minute Setup

### 1. Get Your API Key (2 minutes)

1. **Sign up** at [https://app.tldr.tv](https://app.tldr.tv)
2. **Verify your email** and complete onboarding
3. **Navigate to Settings** → **API Keys**
4. **Click "Generate New Key"** with these scopes:
   - `streams:read`
   - `streams:write` 
   - `highlights:read`
5. **Copy your API key** (starts with `tldr_sk_`)

### 2. Make Your First Request (1 minute)

```bash
# Test your API key
curl -H "X-API-Key: tldr_sk_your_api_key_here" \
  "https://api.tldr.tv/api/v1/streams"
```

**Expected Response:**
```json
{
  "success": true,
  "data": [],
  "pagination": {
    "page": 1,
    "per_page": 20,
    "total": 0,
    "total_pages": 0
  }
}
```

### 3. Process Your First Stream (2 minutes)

```bash
# Start processing a stream
curl -X POST "https://api.tldr.tv/api/v1/streams" \
  -H "X-API-Key: tldr_sk_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "processing_options": {
      "enable_video_analysis": true,
      "enable_audio_analysis": true,
      "confidence_threshold": 0.7
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 123,
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "status": "pending",
    "created_at": "2024-01-15T10:30:00Z",
    "estimated_completion": "2024-01-15T10:35:00Z"
  }
}
```

✅ **Success!** Your stream is now being processed. Continue to the next steps to see your highlights.

## 📈 Next Steps (Optional)

### Check Processing Status

```bash
# Check your stream status
curl -H "X-API-Key: tldr_sk_your_api_key_here" \
  "https://api.tldr.tv/api/v1/streams/123"
```

Wait for `"status": "completed"` (usually 2-5 minutes for short videos).

### Get Your Highlights

```bash
# Retrieve detected highlights
curl -H "X-API-Key: tldr_sk_your_api_key_here" \
  "https://api.tldr.tv/api/v1/highlights?stream_id=123"
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": 456,
      "title": "High Energy Moment",
      "confidence_score": 0.89,
      "start_time": 45.2,
      "duration": 12.5,
      "video_url": "https://cdn.tldr.tv/highlights/456.mp4",
      "thumbnail_url": "https://cdn.tldr.tv/thumbnails/456.jpg"
    }
  ]
}
```

## 🛠️ SDK Integration (Recommended)

### Python
```bash
pip install tldr-highlight-api
```

```python
from tldr_highlight_api import TLDRClient

# Initialize client
client = TLDRClient(api_key="tldr_sk_your_api_key_here")

# Process stream
stream = await client.streams.create(
    url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    processing_options={
        "enable_video_analysis": True,
        "enable_audio_analysis": True
    }
)

print(f"Stream {stream.id} is {stream.status}")

# Wait for completion and get highlights
highlights = await client.highlights.list(stream_id=stream.id)
print(f"Found {len(highlights)} highlights!")
```

### Node.js
```bash
npm install tldr-highlight-api
```

```javascript
const { TLDRClient } = require('tldr-highlight-api');

const client = new TLDRClient({
  apiKey: 'tldr_sk_your_api_key_here'
});

// Process stream
const stream = await client.streams.create({
  url: 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
  processingOptions: {
    enableVideoAnalysis: true,
    enableAudioAnalysis: true
  }
});

console.log(`Stream ${stream.id} is ${stream.status}`);

// Get highlights
const highlights = await client.highlights.list({ streamId: stream.id });
console.log(`Found ${highlights.length} highlights!`);
```

## 🔔 Set Up Webhooks (Optional)

Get real-time notifications when processing completes:

```bash
# Create webhook
curl -X POST "https://api.tldr.tv/api/v1/webhooks" \
  -H "X-API-Key: tldr_sk_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://your-app.com/webhooks/tldr",
    "events": ["stream.completed", "highlight.detected"],
    "secret": "your-webhook-secret-key"
  }'
```

**Webhook payload example:**
```json
{
  "event": "stream.completed",
  "data": {
    "stream_id": 123,
    "status": "completed",
    "highlights_count": 5,
    "processing_duration": 142.5
  },
  "timestamp": "2024-01-15T10:35:00Z"
}
```

## 🎯 Common Use Cases

### Gaming Highlights
```python
# Gaming-optimized processing
stream = await client.streams.create(
    url="https://twitch.tv/your_gaming_stream",
    processing_options={
        "enable_video_analysis": True,
        "enable_audio_analysis": True,
        "enable_chat_analysis": True,  # Great for gaming!
        "confidence_threshold": 0.8,   # Higher threshold for quality
        "dimension_set_id": "gaming_preset"
    }
)
```

### Educational Content
```python
# Education-optimized processing
stream = await client.streams.create(
    url="https://youtube.com/watch?v=educational_video",
    processing_options={
        "enable_video_analysis": True,
        "enable_audio_analysis": True,
        "confidence_threshold": 0.6,   # Lower threshold for learning moments
        "dimension_set_id": "education_preset"
    }
)
```

### Corporate Meetings
```python
# Corporate meeting processing
stream = await client.streams.create(
    url="https://your-meeting-recording.mp4",
    processing_options={
        "enable_video_analysis": False,  # Focus on audio for meetings
        "enable_audio_analysis": True,
        "confidence_threshold": 0.7,
        "dimension_set_id": "corporate_preset"
    }
)
```

## 🔧 Testing & Development

### Development Environment
```bash
# Use development base URL for testing
export TLDR_API_BASE_URL="https://api-dev.tldr.tv/api/v1"
export TLDR_API_KEY="tldr_sk_your_dev_key_here"
```

### Test with Sample Content
```bash
# Process a short test video
curl -X POST "${TLDR_API_BASE_URL}/streams" \
  -H "X-API-Key: ${TLDR_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
    "processing_options": {
      "enable_video_analysis": true,
      "confidence_threshold": 0.5
    }
  }'
```

### Monitor Processing
```python
import asyncio
from tldr_highlight_api import TLDRClient

async def monitor_processing(stream_id: int):
    client = TLDRClient(api_key="your_key_here")
    
    while True:
        stream = await client.streams.get(stream_id)
        print(f"Status: {stream.status}")
        
        if stream.status in ["completed", "failed"]:
            break
            
        await asyncio.sleep(10)  # Check every 10 seconds
    
    if stream.status == "completed":
        highlights = await client.highlights.list(stream_id=stream_id)
        print(f"🎉 Processing complete! Found {len(highlights)} highlights")
        for highlight in highlights:
            print(f"- {highlight.title} ({highlight.confidence_score:.2f})")
    else:
        print(f"❌ Processing failed: {stream.error_message}")

# Usage
await monitor_processing(123)
```

## ❓ Need Help?

### Quick Resources
- **📚 Full Documentation**: [https://docs.tldr.tv](https://docs.tldr.tv)
- **🔍 API Reference**: [https://docs.tldr.tv/api](https://docs.tldr.tv/api)
- **💬 Community**: [Discord](https://discord.gg/tldr-tv)
- **📧 Support**: [support@tldr.tv](mailto:support@tldr.tv)

### Common Issues
- **Authentication failed**: Check your API key format and scopes
- **Stream processing failed**: Verify the video URL is accessible
- **No highlights detected**: Try lowering the confidence threshold
- **Rate limit exceeded**: Upgrade your plan or wait for reset

### What's Next?
1. **Explore custom dimensions** for your specific content type
2. **Set up webhooks** for real-time processing notifications  
3. **Integrate with your application** using our SDKs
4. **Configure monitoring** and usage analytics
5. **Scale up** with higher-tier plans as needed

---

🎉 **Congratulations!** You're now ready to build amazing applications with AI-powered highlight detection. Happy coding!