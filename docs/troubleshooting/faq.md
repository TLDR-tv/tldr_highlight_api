# Frequently Asked Questions (FAQ)

## 🚀 Getting Started

### Q: What is the TL;DR Highlight API?
**A:** The TL;DR Highlight API is an enterprise B2B service that provides AI-powered highlight extraction from livestreams and video content. It uses Google Gemini AI to analyze video, audio, and chat data to automatically identify the most engaging moments in streams.

### Q: What platforms are supported?
**A:** We support:
- **Twitch** streams and VODs
- **YouTube** live streams and videos  
- **RTMP** streams from any source
- **Custom** video URLs and uploads
- **100ms** live streaming platform integration

### Q: What makes this different from other highlight detection services?
**A:** Our key differentiators include:
- **Flexible AI analysis** with customizable dimensions
- **Multi-modal processing** (video + audio + chat)
- **Enterprise-grade architecture** with multi-tenancy
- **Real-time processing** with sub-5-minute detection
- **Comprehensive API** with webhooks and SDKs

## 🔧 Technical Questions

### Q: What are processing dimensions?
**A:** Dimensions are configurable scoring criteria that define what makes content "highlight-worthy" for your use case. Examples:
- **Gaming**: action_intensity, skill_display, emotional_peaks
- **Education**: learning_moments, key_concepts, engagement_peaks
- **Sports**: exciting_plays, scoring_moments, crowd_reactions
- **Corporate**: important_points, decision_moments, engagement_metrics

### Q: How accurate is the AI highlight detection?
**A:** Accuracy varies by content type and dimension configuration:
- **Gaming streams**: 85-95% accuracy for action moments
- **Educational content**: 80-90% accuracy for key concepts
- **Sports**: 90-95% accuracy for scoring events
- **General content**: 75-85% accuracy depending on configuration

You can adjust confidence thresholds and customize dimensions to optimize for your specific content.

### Q: What's the processing time for streams?
**A:** Processing times depend on stream length and complexity:
- **Real-time streams**: 2-5 minutes behind live
- **VOD processing**: ~10-20% of video duration
- **Example**: 60-minute stream processes in 6-12 minutes

### Q: Can I customize the AI analysis?
**A:** Yes! You can:
- Create custom **dimension sets** for your content type
- Define **highlight type registries** with specific criteria
- Choose **analysis strategies** (AI-only, rule-based, hybrid)
- Set **confidence thresholds** and processing options
- Use **industry presets** as starting points

### Q: What video formats and quality are supported?
**A:** We support:
- **Video formats**: MP4, WebM, FLV, HLS, RTMP
- **Resolutions**: Up to 4K (processed at 1080p for analysis)
- **Frame rates**: Any (analyzed at 1 FPS by default)
- **Duration limits**: Up to 6 hours per stream

## 🏢 Business & Pricing

### Q: What subscription plans are available?
**A:** Currently, all organizations have unlimited access:

| Plan | Processing Hours/Month | API Calls/Month | Features |
|------|----------------------|-----------------|----------|
| **All Plans** | Unlimited | Unlimited | Full feature access, comprehensive support |

*Note: We're currently offering unlimited usage for our first clients while we optimize the platform.*

### Q: How is usage calculated and billed?
**A:** Usage is tracked for statistics but not billed currently:
- **Processing time**: Tracked for analytics (no limits)
- **API calls**: Tracked for analytics (no limits)
- **Storage**: Tracked for analytics (no limits)
- **AI analysis**: Tracked for analytics (no limits)

*Note: All usage is currently unlimited for our first clients.*

### Q: Is there a free trial available?
**A:** Yes! We offer:
- **Unlimited access** for early clients
- **No credit card required** 
- **Full feature access** with unlimited usage
- **Priority support** during early access period

### Q: Can I upgrade or downgrade my plan?
**A:** Currently all plans have unlimited access:
- **Plan changes**: Available but all have same unlimited limits
- **Custom configurations**: Contact us for specific needs
- **Enterprise features**: Available to all during early access
- **No usage limits**: No overage charges or restrictions

## 🔐 Security & Compliance

### Q: How secure is my data?
**A:** We implement enterprise-grade security:
- **Encryption**: All data encrypted in transit (TLS 1.3) and at rest (AES-256)
- **Access control**: API key authentication with scoped permissions
- **Isolation**: Multi-tenant architecture with organization-level separation
- **Monitoring**: Comprehensive audit logs and security monitoring
- **Compliance**: SOC 2 Type II, GDPR compliant

### Q: Where is my data stored?
**A:**
- **Video processing**: Temporary processing only, deleted after analysis
- **Highlights**: Stored in your dedicated S3 bucket
- **Metadata**: Encrypted PostgreSQL database
- **Geographic options**: US, EU, and Asia-Pacific regions available

### Q: Can I use my own storage?
**A:** Yes, for Enterprise plans:
- **Bring your own S3 bucket** with cross-account access
- **On-premise deployment** options available
- **Custom storage integrations** can be developed
- **Data residency** requirements can be met

### Q: What about content privacy?
**A:**
- **No content retention**: Video content is not stored after processing
- **Anonymized analysis**: Personal data is not extracted or stored  
- **Webhook security**: All notifications use HMAC signatures
- **Access logs**: Complete audit trail of all data access

## 🛠️ Integration & Development

### Q: How do I get started with the API?
**A:**
1. **Sign up** for an account at https://app.tldr.tv
2. **Generate an API key** in your dashboard
3. **Read the docs** at https://docs.tldr.tv
4. **Try the SDKs** (Python, Node.js, Go, Java, C#)
5. **Test with a sample stream** to see results

### Q: What SDKs and libraries are available?
**A:** Official SDKs:
- **Python**: `pip install tldr-highlight-api`
- **Node.js**: `npm install tldr-highlight-api`
- **Go**: `go get github.com/tldr-tv/go-sdk`
- **Java**: Maven/Gradle integration
- **C#**: NuGet package
- **REST API**: Direct HTTP integration for any language

### Q: How do webhooks work?
**A:** Webhooks provide real-time notifications:
- **Events**: stream.created, stream.completed, highlight.detected
- **Security**: HMAC-SHA256 signatures for verification
- **Reliability**: Automatic retries with exponential backoff
- **Payload**: JSON with event data and metadata
- **Testing**: Webhook tester available in dashboard

### Q: Can I process multiple streams simultaneously?
**A:** Yes, unlimited concurrent processing:
- **All Plans**: Unlimited concurrent streams
- **No restrictions**: Process as many streams as needed
- **High performance**: Optimized for concurrent processing
- **Queue management**: All streams get priority processing

*Note: Unlimited concurrent processing during early access.*

### Q: What's the rate limiting policy?
**A:** Very generous rate limits for all users:
- **Authentication endpoints**: 1000 requests/minute
- **Stream creation**: 1000 requests/minute (effectively unlimited)
- **Data retrieval**: 10,000 requests/minute (effectively unlimited)
- **Webhooks**: No limits on receiving
- **Headers**: Rate limit info still in response headers

*Note: Extremely high limits during early access period.*

## 🎥 Content Processing

### Q: What types of content work best?
**A:** Optimal content characteristics:
- **Gaming**: Fast-paced action, clear audio, visible UI elements
- **Education**: Clear speech, visual presentations, structured content
- **Sports**: HD video, multiple camera angles, commentary
- **Corporate**: Professional audio, slide presentations, Q&A sessions

### Q: How can I improve highlight detection accuracy?
**A:** Optimization strategies:
1. **Custom dimensions**: Define specific criteria for your content
2. **Confidence thresholds**: Adjust to balance precision vs recall
3. **Multi-modal analysis**: Enable video, audio, and chat analysis
4. **Training examples**: Provide sample highlights for better understanding
5. **Feedback loop**: Rate generated highlights to improve future results

### Q: What happens if processing fails?
**A:** Our failure handling:
- **Automatic retries**: Up to 3 retry attempts with exponential backoff
- **Partial processing**: Successful segments are preserved
- **Error reporting**: Detailed error messages via API and webhooks
- **Manual retry**: Option to retry failed streams
- **Support escalation**: Failed streams are automatically flagged for review

### Q: Can I process pre-recorded content?
**A:** Absolutely:
- **Upload via API**: Direct file upload for processing
- **URL processing**: Provide URLs to video files
- **Batch processing**: Process multiple videos in sequence
- **Historical streams**: Process past broadcasts from platforms
- **Custom scheduling**: Schedule processing for optimal resource usage

## 🔄 Workflow & Automation

### Q: How do I integrate highlights into my application?
**A:** Common integration patterns:
1. **Webhook notifications** → Process highlights when ready
2. **Polling API** → Check stream status periodically  
3. **Direct embedding** → Use presigned URLs for video players
4. **Metadata sync** → Import highlight data into your database
5. **Custom workflows** → Chain multiple processing steps

### Q: Can I export highlights to other platforms?
**A:** Export options:
- **Direct URLs**: Presigned URLs for immediate access
- **Bulk download**: ZIP archives of highlight collections
- **API integration**: Programmatic access to all highlight data
- **Platform connectors**: YouTube, TikTok, Twitter integration
- **Custom formats**: JSON, CSV, XML export formats

### Q: What about automated publishing workflows?
**A:** Automation capabilities:
- **Auto-publishing**: Automatically post highlights to social media
- **Content moderation**: Filter highlights by confidence and content type
- **Scheduling**: Publish highlights at optimal times
- **A/B testing**: Test different highlight variations
- **Analytics integration**: Track highlight performance metrics

## 📊 Analytics & Reporting

### Q: What analytics and insights are provided?
**A:** Comprehensive analytics:
- **Processing metrics**: Success rates, processing times, error analysis
- **Content insights**: Most common highlight types, confidence distributions
- **Usage analytics**: API usage, storage consumption, cost analysis
- **Performance data**: Response times, throughput metrics
- **Business metrics**: Highlight engagement, conversion rates

### Q: Can I access raw processing data?
**A:** Data access options:
- **API endpoints**: Detailed highlight metadata and scores
- **Data export**: CSV, JSON exports of processing results
- **Webhook payloads**: Real-time access to all processing data
- **Custom reporting**: Enterprise customers can access raw logs
- **Analytics integration**: Connect to your BI tools and dashboards

### Q: How do I monitor system performance?
**A:** Monitoring tools:
- **Dashboard**: Real-time system status and metrics
- **Health checks**: API endpoints for service monitoring
- **Alerting**: Email/SMS alerts for issues or thresholds
- **SLA reporting**: Uptime and performance SLA tracking
- **Custom monitoring**: Integrate with your monitoring stack

## 🆘 Support & Troubleshooting

### Q: What support options are available?
**A:** Support by plan tier:
- **Starter**: Email support, documentation, community forum
- **Professional**: Priority email, chat support, SLA response times
- **Enterprise**: Dedicated support team, phone support, custom SLAs
- **Custom**: On-site support, dedicated engineers, 24/7 availability

### Q: How do I troubleshoot common issues?
**A:** Troubleshooting resources:
1. **Status page**: Check system status at https://status.tldr.tv
2. **Documentation**: Comprehensive guides at https://docs.tldr.tv
3. **Health checks**: Use `/health` endpoint to verify service status
4. **Logs**: Check application logs for detailed error information
5. **Support**: Contact support with specific error messages

### Q: What if I need custom features?
**A:** Custom development:
- **Feature requests**: Submit via GitHub or support channels
- **Enterprise features**: Custom development for Enterprise customers
- **Consulting services**: Implementation and integration support
- **White-label solutions**: Fully branded implementations
- **On-premise deployment**: Custom deployment options

### Q: How do I migrate from another service?
**A:** Migration support:
- **Data migration**: Tools to import existing highlight data
- **API compatibility**: Wrapper APIs for common services
- **Parallel processing**: Run both services during transition
- **Migration team**: Dedicated team for Enterprise migrations
- **Documentation**: Step-by-step migration guides

## 📚 Additional Resources

### Q: Where can I find more information?
**A:** Useful resources:
- **Documentation**: https://docs.tldr.tv
- **API Reference**: https://docs.tldr.tv/api
- **SDKs**: https://github.com/tldr-tv
- **Status Page**: https://status.tldr.tv
- **Blog**: https://blog.tldr.tv
- **Community**: https://discord.gg/tldr-tv

### Q: How do I stay updated on new features?
**A:** Stay informed:
- **Newsletter**: Subscribe for product updates
- **Release notes**: Detailed changelog for all releases
- **Blog**: Feature announcements and tutorials
- **Social media**: Follow @TLDRHighlights for updates
- **Webhooks**: API version and feature notifications

---

## 💬 Still Have Questions?

If you don't see your question here:

1. **Search the documentation** at https://docs.tldr.tv
2. **Check GitHub issues** for technical questions
3. **Contact support** at support@tldr.tv
4. **Join our community** on Discord for peer help
5. **Schedule a demo** for complex integration questions

We're here to help make your integration successful! 🎉