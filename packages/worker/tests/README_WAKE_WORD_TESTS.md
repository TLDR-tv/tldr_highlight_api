# Wake Word Detection Tests

This document describes the comprehensive test suite for the wake word detection system.

## Test Coverage

### 1. Domain Model Tests (`test_wake_word_model.py`)
- **Basic functionality**: Creation, normalization, matching
- **Edge cases**: Empty phrases, special characters, unicode, very long phrases
- **Cooldown logic**: Trigger timing and enforcement
- **Matching modes**: Exact vs fuzzy, case sensitive vs insensitive
- **Configuration limits**: Validation of thresholds and distances

### 2. Repository Tests (`test_wake_word_repository.py`)
- **CRUD operations**: Create, read, update, delete
- **Filtering**: By organization, active status
- **Constraints**: Unique phrase per organization
- **Bulk operations**: Performance with many wake words
- **Edge cases**: Null handling, special characters

### 3. FFmpeg Processor Tests (`test_enhanced_ffmpeg_processor.py`)
- **Dual extraction**: Video segments and audio chunks
- **Ring buffer management**: Memory efficiency
- **Stream format detection**: RTMP, HLS, DASH, HTTP, File
- **Error recovery**: Process failure and restart
- **Context manager**: Resource cleanup
- **Audio chunk overlap**: Boundary detection

### 4. Wake Word Detection Task Tests (`test_wake_word_detection.py`)
- **Transcription**: Integration with faster-whisper
- **Matching algorithms**: Fuzzy matching with Levenshtein-Damerau distance
- **Cooldown enforcement**: Preventing rapid re-triggering
- **Multi-word phrases**: Support for complex wake phrases
- **Error handling**: Transcription failures, missing files
- **Timestamp adjustment**: Chunk offset handling

### 5. Stream Processing Integration Tests (`test_stream_processing_with_wake_words.py`)
- **Task queuing**: Wake word detection tasks properly queued
- **Audio chunk handling**: Processing with and without audio
- **Error handling**: Database and processing failures
- **Resource cleanup**: Temporary directory management

### 6. End-to-End Integration Tests (`test_wake_word_detection_integration.py`)
- **Full pipeline**: Stream → FFmpeg → Transcription → Detection
- **Chunk boundaries**: Wake words spanning audio chunks
- **Concurrent processing**: Multiple streams
- **Performance**: Large transcripts, many wake words
- **Memory management**: Ring buffer effectiveness

## Test Scenarios Covered

### Edge Cases
1. **Empty/whitespace-only phrases**
2. **Special regex characters** (properly escaped)
3. **Unicode characters** in wake words and transcripts
4. **Very long phrases** (50+ words)
5. **Rapid concurrent triggers** (cooldown testing)
6. **Wake words at audio chunk boundaries**
7. **Missing or corrupted audio files**
8. **Transcription failures**
9. **Zero-duration segments**
10. **Process crashes and recovery**

### Performance Tests
1. **Large transcripts** (4000+ words)
2. **Many wake words** (100+ per organization)
3. **Long streams** (10+ minutes)
4. **Concurrent stream processing**
5. **Memory usage with ring buffers**

### Integration Points
1. **FFmpeg process management**
2. **faster-whisper transcription**
3. **Database operations**
4. **Celery task queuing**
5. **File system operations**

## Running the Tests

### Run all wake word tests:
```bash
python packages/worker/tests/run_wake_word_tests.py
```

### Run individual test files:
```bash
# Unit tests
pytest packages/shared/tests/domain/test_wake_word_model.py -v
pytest packages/worker/tests/test_wake_word_detection.py -v

# Integration tests
pytest packages/worker/tests/integration/test_wake_word_detection_integration.py -v
```

### Run with coverage:
```bash
pytest packages/worker/tests/test_wake_word_*.py --cov=worker.tasks.wake_word_detection --cov=worker.services.ffmpeg_processor
```

## Test Data Requirements

Some tests require:
- **NumPy**: For generating test audio files
- **Mock audio files**: Created dynamically in tests
- **Database**: Tests use async SQLAlchemy sessions
- **Temporary directories**: Automatically cleaned up

## Mocking Strategy

The tests use extensive mocking to:
- Avoid dependency on external services
- Simulate transcription results
- Control timing for cooldown tests
- Inject failures for error handling tests
- Track method calls for verification

## Key Test Patterns

1. **Async context managers**: Proper resource cleanup
2. **Parametrized tests**: Multiple scenarios with same logic
3. **Fixtures**: Reusable test data and mocks
4. **Integration markers**: `@pytest.mark.integration`
5. **Performance timing**: Ensuring operations complete quickly