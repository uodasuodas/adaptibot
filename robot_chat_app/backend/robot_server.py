"""
TCP server for robot hand communication
Handles connections from the tasker (robot controller)
"""
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port
        self.client_writer: Optional[asyncio.StreamWriter] = None
        self.client_reader: Optional[asyncio.StreamReader] = None
        self.connected = False
        self.current_objects: List[Dict] = []
        self.drop_points: List[str] = [
            "50 60 14",
            "550 160 15", 
            "850 180 16"
        ]
        self.response_queue: asyncio.Queue = asyncio.Queue()
        
    async def start(self):
        """Start the TCP server"""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        addr = server.sockets[0].getsockname()
        logger.info(f"Robot server listening on {addr}")
        async with server:
            await server.serve_forever()
    
    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming client connection"""
        addr = writer.get_extra_info('peername')
        logger.info(f"Client connected from {addr}")
        
        self.client_reader = reader
        self.client_writer = writer
        self.connected = True
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    logger.info("Client disconnected")
                    break
                
                message = data.decode('utf-8').strip()
                logger.info(f"Received from tasker: {message}")
                
                await self.response_queue.put(message)
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.connected = False
            self.client_writer = None
            self.client_reader = None
            writer.close()
            await writer.wait_closed()
    
    async def send_command(self, command: Dict[str, Any]) -> None:
        """Send command to tasker"""
        if not self.connected or not self.client_writer:
            raise ConnectionError("Tasker not connected")
        
        message = json.dumps(command)
        logger.info(f"Sending to tasker: {message}")
        self.client_writer.write(message.encode('utf-8'))
        await self.client_writer.drain()
        await asyncio.sleep(0.5)
    
    async def wait_for_response(self, timeout: float = 30.0) -> Optional[Dict]:
        """Wait for response from tasker"""
        try:
            response = await asyncio.wait_for(
                self.response_queue.get(), 
                timeout=timeout
            )
            return json.loads(response)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for response")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
            return None
    
    async def initialize_session(self) -> None:
        """Initialize robot session with default drop points"""
        if not self.connected:
            raise ConnectionError("Tasker not connected")
        
        await self.send_command({"cmd": "clear_drop_points"})
        await asyncio.sleep(1)
        
        for idx, coords in enumerate(self.drop_points):
            await self.send_command({
                "cmd": "set_drop_point",
                "coordinates": coords,
                "id": idx
            })
            await asyncio.sleep(1)
        
        await self.send_command({"cmd": "start_listen"})
        logger.info("Session initialized")
    
    async def get_objects(self) -> List[Dict]:
        """Request current object list from tasker"""
        await self.send_command({"cmd": "give_objects"})
        response = await self.wait_for_response()
        
        if response and "objects" in response:
            objects = self.parse_objects(response["objects"])
            self.current_objects = objects
            return objects
        return []
    
    def parse_objects(self, obj_list: List[str]) -> List[Dict]:
        """Parse object list from tasker format"""
        objects = []
        for obj_str in obj_list:
            parts = obj_str.split()
            if len(parts) == 5:
                objects.append({
                    "id": int(parts[0]),
                    "class_id": int(parts[0]),
                    "color_id": int(parts[1]),
                    "x": int(parts[2]),
                    "y": int(parts[3]),
                    "z": int(parts[4]),
                    "class_name": self.get_class_name(int(parts[0])),
                    "color_name": self.get_color_name(int(parts[1]))
                })
        return objects
    
    def get_class_name(self, class_id: int) -> str:
        """Get object class name from ID"""
        classes = {
            0: "unknown", 1: "can", 2: "duck", 
            3: "cup", 4: "sponge", 5: "ball", 6: "vegetable"
        }
        return classes.get(class_id, "unknown")
    
    def get_color_name(self, color_id: int) -> str:
        """Get color name from ID"""
        colors = {
            0: "unknown", 1: "red", 2: "blue", 3: "green",
            4: "yellow", 5: "black", 6: "grey", 7: "white",
            8: "violet", 9: "orange", 10: "turquoise"
        }
        return colors.get(color_id, "unknown")
    
    async def execute_commands(self, cmd_list: List[str]) -> bool:
        """Send command list to tasker for execution"""
        await self.send_command({"cmd_list": cmd_list})
        
        # Wait for stop_listen acknowledgment
        response = await self.wait_for_response()
        if not response:
            return False
        
        # Wait for start_listen when execution completes
        response = await self.wait_for_response(timeout=120.0)
        return response is not None


# Global instance
robot_server = RobotServer()

