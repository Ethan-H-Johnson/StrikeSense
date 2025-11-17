# StrikeSense

Real-time MMA strike analysis application using video processing, YOLO pose estimation, and SMPL 3D pose recovery. Provides actionable feedback for perfecting jabs and crosses.

## Architecture

- **Client**: Next.js web application (mobile-optimized)
- **Server**: FastAPI backend with WebSocket support
- **Pose Estimation**: YOLOv11-Pose for 2D keypoints → SMPL for 3D pose
- **Communication**: WebSocket for real-time, low-latency feedback

## Features

- Real-time video analysis from mobile camera
- Multi-person tracking with filtering (handles crowded scenes)
- Metrics calculation: punch angle, speed, hip rotation
- Biomechanics-based scoring for jabs and crosses
- LLM-powered actionable feedback

## Setup

### Server Setup

1. Navigate to server directory:
```bash
cd server
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure YOLO model file exists:
   - Place `yolo11n-pose.pt` in the project root or server directory
   - Or download it (YOLO will auto-download on first run)

5. Set environment variables (create `.env` file):
```env
LLM_API_KEY=your_openai_api_key_here
HOST=0.0.0.0
PORT=8000
DEBUG=True
```

6. Run server:
```bash
python main.py
# Or: uvicorn main:app --reload
```

### Client Setup

1. Navigate to client directory:
```bash
cd client
```

2. Install dependencies:
```bash
npm install
```

3. Set environment variables (create `.env.local`):
```env
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

4. Run development server:
```bash
npm run dev
```

5. Open browser to `http://localhost:3000`

## Usage

1. Start the server (see Server Setup)
2. Start the client (see Client Setup)
3. Open the web app in your browser
4. Click "Start Camera" to enable camera access
5. Click "Start Analysis" to begin real-time strike analysis
6. Perform jabs and crosses - receive real-time feedback!

## Project Structure

```
StrikeSense/
├── server/                 # FastAPI backend
│   ├── api/                # API endpoints
│   │   ├── websocket.py    # WebSocket handler
│   │   └── health.py       # Health check
│   ├── services/           # Business logic
│   │   ├── pose_estimator.py
│   │   ├── person_tracker.py
│   │   ├── metrics_calculator.py
│   │   ├── scorer.py
│   │   └── llm_feedback.py
│   ├── models/             # Data models
│   ├── config.py           # Configuration
│   └── main.py             # Entry point
│
├── client/                 # Next.js frontend
│   ├── app/                # Next.js app directory
│   ├── components/         # React components
│   └── lib/                # Utilities
│
└── data/                   # Video data files
```

## Configuration

### Server Configuration

Edit `server/config.py` or set environment variables:

- `YOLO_MODEL_PATH`: Path to YOLO model file
- `SMPL_MODEL_PATH`: Path to SMPL model files (for future use)
- `LLM_API_KEY`: OpenAI API key for feedback generation
- `MAX_CONCURRENT_SESSIONS`: Maximum concurrent analysis sessions
- `FRAME_SKIP_RATE`: Process every Nth frame (performance tuning)

### Biomechanics Settings

Target values (from research):
- `TARGET_ELBOW_ANGLE`: 175.0° (perfect extension)
- `PERFECT_RANGE`: ±5° for 10/10 score
- `EXTENSION_ANGLE`: 160° (minimum for punch detection)

## Development

### Adding SMPL Support

1. Install SMPL library:
```bash
pip install smplx
```

2. Download SMPL model files
3. Update `server/services/pose_estimator.py` to use SMPL
4. Update `server/services/smpl_service.py` with implementation

### Testing

Server health check:
```bash
curl http://localhost:8000/api/health
```

## Deployment

### AWS Deployment

1. Create EC2 instance
2. Install dependencies
3. Set up environment variables
4. Run with production WSGI server:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Client Deployment

Deploy to Vercel, Netlify, or similar:
```bash
cd client
npm run build
```

## Notes

- SMPL integration is currently a placeholder - implement when ready
- Person tracking uses heuristics to filter out audience/refs
- LLM feedback falls back to rule-based if API key not set
- Frame skipping can be adjusted for performance vs. accuracy tradeoff

## License

MIT
