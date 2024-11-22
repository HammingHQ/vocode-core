# HME NEXEO Voice AI Order-Taking (VAIO) Reference

## Key Components

- NEXEO base station
- Speaker/mic units at drive-thru lanes
- Crew headsets
- HME Cloud for configuration
- AOT (Automatic Order Taking) Provider

## Communication Channels

### WebSocket Connections
- `/audio` - Drive-thru audio streams
- `/message` - Control messages and telemetry
- `/bot` - Employee-BOT communication

### Authentication
- Uses JWT tokens encrypted with RSA keys
- Public key configured in HME Cloud
- Private key held by AOT Provider
- Token refresh required before expiration

### Audio Formats

Drive-thru Audio:
- 16kHz PCM
- 16-bit signed integer
- Little-endian
- Single lane: 2 channels (mic + speaker)
- Dual lane: 4 channels (2x mic + 2x speaker)
- Fixed chunks: 1920 bytes (single) or 3840 bytes (dual)

Employee-BOT Audio:
- 16kHz PCM
- 16-bit signed integer
- Little-endian
- Single channel
- 960 byte chunks every 30ms

### Key Message Types

Vehicle Events:
- `NEXEO/request/lane{1,2}/arrive`
- `NEXEO/request/lane{1,2}/depart`

Status/Telemetry:
- `NEXEO/status/configuration`
- `NEXEO/status/telemetry`
- `NEXEO/audio/telemetry`
- `NEXEO/status/error`

Control:
- `aot/request/auto-escalation/lane{1,2}`
- `aot/request/reconnect`
- `aot/request/audio-interruption`

### Audio Modes

Inbound:
- Continuous - Always streaming
- Vehicle Present - Only when vehicle detected

Outbound:
- Fixed - 30ms chunks with strict timing
- Variable - 5ms to 30s chunks with flexible timing

## Configuration Settings

Required Provider Info:
- Public key and expiration
- Token expiration time
- WebSocket URL and port
- Audio modes
- Reconnect interval
- Max concurrent BOT connections

## Key Features

- Bi-directional audio for order taking
- Crew takeover capability
- Employee-BOT communication
- BOT audio broadcasting
- Network latency monitoring
- Telemetry and diagnostics

## Expanded Details and Examples

### WebSocket Connections

To enhance the Python application's design, ensure it manages the following WebSocket connections concurrently:
1. /message: Handles control messages such as arrive and depart events.
2. /audio: Manages audio byte streams between NEXEO and the AOT Provider.
3. /bot: Manages audio byte streams between NEXEO and the BOT.

### Asynchronous Event Management

Implement asynchronous handling to coordinate interactions across multiple WebSocket connections effectively. For instance:
- Sending an "arrive" Event:
  - Dispatch the arrive event through the /message WebSocket.
- Receiving Audio Response:
  - The AOT Provider responds with audio bytes on the /audio WebSocket.

### Key Considerations

- Concurrency: Utilize Python's asyncio to run parallel tasks for each WebSocket, ensuring non-blocking operations.
- Event Coordination: Maintain separate asynchronous event loops or use synchronized queues to manage interconnection events seamlessly.
- Error Handling: Gracefully handle disconnections and retries for each WebSocket to maintain robust communication channels.
- Logging: Implement comprehensive logging for monitoring the status and performance of each WebSocket connection.

### Example Workflow

1. Establish Connections:
  - Connect to /message, /audio, and /control WebSockets asynchronously.
2. Event Dispatching:
  - When a vehicle arrives, send an arrive event via / message.
3. Audio Handling:
  - Listen on /audio for the corresponding audio bytes from the AOT Provider.
4. Control Operations:
  - Utilize /control for any additional commands or configurations as needed.
