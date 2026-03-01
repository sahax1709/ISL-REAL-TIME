# 🤟 ISLDetect — Real-Time Indian Sign Language Recognition

Real-time ISL detection using **MediaPipe** hand tracking with bilingual **English + Hindi (हिंदी)** output.

## Features
- Real-time webcam detection via WebSocket
- 25 ISL signs (alphabets, numbers, common words)
- English + Hindi output side by side
- Hand landmark visualization on camera feed
- Finger state indicator & detection history
- Docker + Render deploy-ready (free tier)

## 🧪 Signs You Can Test

### Alphabets
| Sign | Hindi | How to Form |
|------|-------|-------------|
| A | अ | Closed fist, thumb on side |
| B | ब | Flat hand, 4 fingers up, thumb folded |
| C | स | Curved hand like a cup |
| D | ड | Index up, rest curled |
| I | इ | Only pinky extended |
| L | ल | L-shape: thumb + index at 90 degrees |
| V | व | Peace/victory sign |
| W | व | Three fingers up |
| Y | य | Thumb + pinky extended |

### Numbers
| Sign | Hindi | How to Form |
|------|-------|-------------|
| One | एक | Index finger up |
| Two | दो | Index + middle up |
| Three | तीन | Index + middle + ring up |
| Four | चार | Four fingers up, thumb folded |
| Five | पाँच | All fingers spread |

### Words
| Sign | Hindi | How to Form |
|------|-------|-------------|
| Hello | नमस्ते | Open palm, fingers spread wide |
| Yes | हाँ | Thumbs up |
| Bad | बुरा | Thumbs down |
| OK | ठीक है | Thumb + index circle |
| Stop | रुको | Palm forward, fingers together |
| I Love You | मैं तुमसे प्यार... | Thumb + index + pinky extended |
| Rock | रॉक | Index + pinky extended |

## Local Setup
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open http://localhost:8000

## Docker
```bash
docker build -t isl-detector .
docker run -p 8000:10000 isl-detector
```

## Deploy to Render (Free)
1. Push to GitHub
2. Render.com -> New -> Web Service -> connect repo
3. Auto-detects Dockerfile. Port: 10000, Plan: free
4. Deploy!

## Tips for Best Detection
- Good even lighting, plain background
- Hand 30-60cm from camera
- Hold signs steady for 1-2 seconds
- Palm facing camera for most signs
