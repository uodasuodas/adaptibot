# Quick start guide

Get the robot control chat running in 3 steps:

## Step 1: Configure

Create a `.env` file with your OpenAI API key:

```bash
cp env.example .env
nano .env  # or use your preferred editor
```

Add your actual API key:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
```

## Step 2: Start the application

Run the startup script:

```bash
./start.sh
```

Or manually with Docker:
```bash
docker-compose up --build
```

The application will start on:
- **Web interface**: http://localhost:8080
- **Robot TCP server**: localhost:8000

## Step 3: Test with simulator

In a new terminal, run the tasker simulator:

```bash
./run_tasker_simulator.sh
```

Or manually:
```bash
cd tests
python tasker_simulator.py
```

## Using the chat

Open http://localhost:8080 in your browser and try commands like:

- "What objects do you see?"
- "Pick up all red balls"
- "Remove all yellow objects"
- "Throw out the blue cup"

You can either:
- **Type** your command in the text box
- **Speak** by clicking the microphone button (works in Chrome, Edge, Safari)

The AI will:
1. Query the detected objects
2. Plan the actions needed
3. Execute grab/drop commands
4. Report back to you

## Example flow

```
You: Show me what you can see
AI: [Calls get_detected_objects tool]
    I can see:
    - red ball at (200, 80, 100)
    - blue cup at (271, 131, 100)
    - yellow sponge at (390, 57, 100)
    ...

You: Throw out all red balls
AI: [Creates plan, calls execute_robot_commands]
    I'll pick up the red ball and move it to the disposal point.
    [Executes GRAB and DROP commands]
    Successfully removed 1 red ball.
```

## Troubleshooting

**"Connection refused"**: Make sure the main application is running first (step 2)

**"OPENAI_API_KEY not set"**: Check your `.env` file has the correct key

**Objects not updating**: Click the "Refresh" button in the left panel

**Robot shows disconnected**: Start the tasker simulator (step 3)

**Voice button disabled**: Use Chrome, Edge, or Safari browser (Firefox doesn't support Web Speech API)

**Microphone not working**: Allow microphone access when browser prompts, or check browser settings

