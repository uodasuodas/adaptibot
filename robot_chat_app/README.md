# Robot control chat

A dockerized web application for controlling a robot hand through natural language commands. The system uses a chat interface where users can see detected objects and give commands like "throw out all red balls". The backend uses LangChain/LangGraph with OpenAI's GPT to process commands and control the robot.

## Architecture

- **Frontend**: Simple HTML/CSS/JS chat interface with object visualization
- **Backend**: FastAPI with WebSocket for real-time communication
- **AI Agent**: LangGraph-based agent using GPT-4 with custom tools
- **Robot Server**: TCP server that communicates with the robot controller (tasker)

## Protocol

The system uses JSON messages over TCP:

### Commands to tasker
- `{"cmd":"give_objects"}` - Request current object list
- `{"cmd_list":["GRAB x y z", "DROP x y z", ...]}` - Execute grab/drop commands

### Commands from AI server
- `{"cmd":"start_listen"}` - Enable user input processing
- `{"cmd":"stop_listen"}` - Disable user input during execution
- `{"cmd":"clear_drop_points"}` - Clear all drop points
- `{"cmd":"set_drop_point","coordinates":"x y z","id":0}` - Set drop point

### Object format
Objects are sent as: `"id color_id x y z"`

**Object classes** (id):
- 0: unknown, 1: can, 2: duck, 3: cup, 4: sponge, 5: ball, 6: vegetable

**Colors** (id):
- 0: unknown, 1: red, 2: blue, 3: green, 4: yellow, 5: black, 6: grey, 7: white, 8: violet, 9: orange, 10: turquoise

## Setup

### Prerequisites
- Docker and Docker Compose
- OpenAI API key

### Configuration

1. Create a `.env` file from the example:
```bash
cp env.example .env
```

2. Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-your-actual-key-here
ROBOT_SERVER_HOST=0.0.0.0
ROBOT_SERVER_PORT=8000
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8080
```

### Running with Docker

1. Build and start the application:
```bash
docker-compose up --build
```

2. Open your browser and navigate to:
```
http://localhost:8080
```

3. In a separate terminal, run the tasker simulator to test:
```bash
python tests/tasker_simulator.py
```

### Running without Docker

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export OPENAI_API_KEY=sk-your-key-here
```

3. Run the backend:
```bash
python app.py
```

4. Open `frontend/index.html` in your browser or serve it with a simple HTTP server:
```bash
python -m http.server 8080
```

5. Run the tasker simulator:
```bash
python tests/tasker_simulator.py
```

## Usage

1. **Connect**: The system automatically connects when you open the web interface
2. **View objects**: Detected objects appear in the left panel with their colors and positions
3. **Give commands**: Type natural language commands or use voice input:
   - Type in the text box, or
   - Click the microphone button and speak your command
   - Example commands:
     - "Show me what objects you can see"
     - "Throw out all red balls"
     - "Pick up the blue cup and move it"
     - "Remove all yellow objects"
4. **Monitor execution**: The AI will create a plan and execute robot commands, reporting progress

### Voice input

The application uses the Web Speech API for voice recognition:
- Click the microphone button to start recording
- Speak your command clearly
- The button turns red while listening
- Click again to stop, or it will stop automatically after you finish speaking
- Supported browsers: Chrome, Edge, Safari (not Firefox)
- Requires microphone permission from your browser

## Example commands

The AI agent will:
1. First call `get_detected_objects()` to see available objects
2. Parse your request to identify target objects (e.g., "all red balls")
3. Generate GRAB/DROP command pairs for matching objects
4. Send commands to the robot via `execute_robot_commands()`
5. Report status back to you in the chat

## Development

### Project structure
```
robot_chat_app/
├── backend/
│   ├── app.py              # FastAPI application
│   ├── robot_server.py     # TCP server for robot
│   ├── llm_agent.py        # LangGraph agent
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── tests/
│   └── tasker_simulator.py # Robot simulator for testing
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Customization

**Add new object classes**: Edit `get_class_name()` in `robot_server.py`

**Add new colors**: Edit `get_color_name()` in `robot_server.py`

**Modify drop points**: Edit the `drop_points` list in `RobotServer.__init__()`

**Change AI model**: Edit `ChatOpenAI` initialization in `llm_agent.py`

## Troubleshooting

**Robot not connecting**: Ensure the tasker simulator is running and connecting to the correct port (8000)

**AI not responding**: Check that your OpenAI API key is valid and set in the `.env` file

**WebSocket errors**: Check browser console for connection issues, ensure backend is running on port 8080

## Notes

- The LLM generates command pairs (GRAB + DROP) for each object
- GRAB coordinates come from detected objects
- DROP coordinates must be from predefined drop points
- The system validates all commands before execution
- User input is blocked during command execution for safety

