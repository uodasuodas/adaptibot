"""
FastAPI backend with WebSocket support for robot control chat
"""
import os
import asyncio
import json
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import logging

from robot_server import robot_server
from llm_agent import create_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Robot Control Chat")

# WebSocket connections
active_connections: Set[WebSocket] = set()

# Initialize agent
agent = None


@app.on_event("startup")
async def startup_event():
    """Start robot server and initialize agent"""
    global agent
    
    # Start robot TCP server in background
    asyncio.create_task(robot_server.start())
    
    # Wait a bit for server to start
    await asyncio.sleep(1)
    
    # Create LLM agent
    try:
        agent = create_agent()
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise


@app.get("/")
async def read_root():
    """Serve frontend"""
    return FileResponse("../frontend/index.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "robot_connected": robot_server.connected,
        "agent_ready": agent is not None
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat"""
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "content": "Connected to robot control system",
            "robot_status": "connected" if robot_server.connected else "disconnected"
        })
        
        # If robot connected, initialize and send current objects
        if robot_server.connected:
            try:
                await robot_server.initialize_session()
                objects = await robot_server.get_objects()
                await websocket.send_json({
                    "type": "objects",
                    "objects": objects
                })
            except Exception as e:
                logger.error(f"Failed to initialize session: {e}")
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type", "message")
            content = data.get("content", "")
            
            if message_type == "message" and content:
                # Echo user message
                await websocket.send_json({
                    "type": "user",
                    "content": content
                })
                
                # Process with agent
                try:
                    # Send typing indicator
                    await websocket.send_json({
                        "type": "typing",
                        "content": "Processing..."
                    })
                    
                    response = await agent.process_message(content)
                    
                    # Send AI response
                    await websocket.send_json({
                        "type": "assistant",
                        "content": response
                    })
                    
                    # Update objects display
                    if robot_server.connected:
                        objects = robot_server.current_objects
                        await websocket.send_json({
                            "type": "objects",
                            "objects": objects
                        })
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "content": f"Error: {str(e)}"
                    })
            
            elif message_type == "refresh_objects":
                # Refresh object list
                if robot_server.connected:
                    try:
                        objects = await robot_server.get_objects()
                        await websocket.send_json({
                            "type": "objects",
                            "objects": objects
                        })
                    except Exception as e:
                        logger.error(f"Error refreshing objects: {e}")
    
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_connections.discard(websocket)


# Mount frontend static files
app.mount("/static", StaticFiles(directory="../frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8080"))
    
    uvicorn.run(app, host=host, port=port)

