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
    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self.client_writer: Optional[asyncio.StreamWriter] = None
        self.client_reader: Optional[asyncio.StreamReader] = None
        self.connected = False
        self.current_objects: List[Dict] = []
        # Drop points: default 3 boxes, robot can add more via set_drop_point
        self.drop_points: List[Dict] = [
            {"id": 0, "label": "BOX1", "coordinates": "50 60 14"},
            {"id": 1, "label": "BOX2", "coordinates": "550 160 15"},
            {"id": 2, "label": "BOX3", "coordinates": "850 180 16"}
        ]
        self.response_queue: asyncio.Queue = asyncio.Queue()
        self.broadcast_callback = None  # Will be set by app.py
        # Track objects in each box: {box_id: [objects]}
        self.box_contents: Dict[int, List[Dict]] = {0: [], 1: [], 2: []}
        
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
        
        # Broadcast robot connection status to UI
        if self.broadcast_callback:
            await self.broadcast_callback({
                "type": "system",
                "content": f"Robot connected from {addr[0]}",
                "robot_status": "connected"
            })
            # Send drop points and current state
            await self.broadcast_callback({
                "type": "drop_points",
                "drop_points": self.drop_points
            })
        
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    logger.info("Client disconnected")
                    break
                
                message = data.decode('utf-8').strip()
                logger.info(f"Received from tasker: {message}")
                
                # Check if this is a command from robot (not a response)
                try:
                    msg_data = json.loads(message)
                    
                    # Handle set_drop_point command from robot
                    if msg_data.get("cmd") == "set_drop_point":
                        await self.handle_set_drop_point(msg_data)
                        continue
                    
                    # Handle clear_drop_points command from robot
                    if msg_data.get("cmd") == "clear_drop_points":
                        await self.handle_clear_drop_points()
                        continue
                        
                except json.JSONDecodeError:
                    pass
                
                # Queue all messages for wait_for_response()
                await self.response_queue.put(message)
                
        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            self.connected = False
            self.client_writer = None
            self.client_reader = None
            self.current_objects = []
            # Reset to default 3 boxes
            self.drop_points = [
                {"id": 0, "label": "BOX1", "coordinates": "50 60 14"},
                {"id": 1, "label": "BOX2", "coordinates": "550 160 15"},
                {"id": 2, "label": "BOX3", "coordinates": "850 180 16"}
            ]
            self.reset_boxes()
            logger.info("Robot disconnected - cleared cache and reset to default boxes")
            
            # Broadcast robot disconnection to UI
            if self.broadcast_callback:
                await self.broadcast_callback({
                    "type": "system",
                    "content": "Robot disconnected",
                    "robot_status": "disconnected"
                })
                # Send reset drop points
                await self.broadcast_callback({
                    "type": "drop_points",
                    "drop_points": self.drop_points
                })
            
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
    
    async def handle_set_drop_point(self, msg_data: Dict) -> None:
        """Handle set_drop_point command from robot"""
        coords = msg_data.get("coordinates", "")
        box_id = msg_data.get("id", len(self.drop_points))
        label = msg_data.get("label", f"BOX{box_id + 1}")
        
        # Check if box with this ID already exists
        existing_idx = None
        for idx, box in enumerate(self.drop_points):
            if box["id"] == box_id:
                existing_idx = idx
                break
        
        box_data = {
            "id": box_id,
            "label": label,
            "coordinates": coords
        }
        
        if existing_idx is not None:
            self.drop_points[existing_idx] = box_data
            logger.info(f"Updated drop point: {label} ({coords})")
        else:
            self.drop_points.append(box_data)
            logger.info(f"Added drop point: {label} ({coords})")
        
        # Ensure box_contents has entry for this box
        if box_id not in self.box_contents:
            self.box_contents[box_id] = []
        
        # Broadcast to UI
        if self.broadcast_callback:
            await self.broadcast_callback({
                "type": "drop_points",
                "drop_points": self.drop_points
            })
    
    async def handle_clear_drop_points(self) -> None:
        """Handle clear_drop_points command from robot"""
        self.drop_points = []
        self.box_contents = {}
        logger.info("Cleared all drop points")
        
        # Broadcast to UI
        if self.broadcast_callback:
            await self.broadcast_callback({
                "type": "drop_points",
                "drop_points": []
            })
    
    async def initialize_session(self) -> None:
        """Initialize robot session - just start listening"""
        if not self.connected:
            raise ConnectionError("Tasker not connected")
        
        await self.send_command({"cmd": "start_listen"})
        logger.info("Session initialized")
    
    async def get_objects(self) -> List[Dict]:
        """Request current object list from tasker"""
        await self.send_command({"cmd": "give_objects"})
        response = await self.wait_for_response()
        
        if response and "objects" in response:
            objects = self.parse_objects(response["objects"])
            self.current_objects = objects
            logger.info(f" Parsed {len(objects)} objects from tasker")
            for obj in objects:
                logger.info(f"  - {obj['color_name']} {obj['class_name']} at ({obj['x']}, {obj['y']}, {obj['z']})")
            return objects
        logger.warning(" No objects in response from tasker")
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
        """Get color name from ID - matches real robot mapping"""
        colors = {
            0: "black", 1: "white", 2: "grey", 3: "red",
            4: "blue", 5: "green", 6: "cyan", 7: "magenta", 8: "yellow"
        }
        return colors.get(color_id, "unknown")
    
    def get_box_id(self, drop_coords: str) -> Optional[int]:
        """Get box ID from drop coordinates"""
        for box in self.drop_points:
            if box["coordinates"] == drop_coords:
                return box["id"]
        return None
    
    def add_to_box(self, box_index: int, obj: Dict) -> None:
        """Add object to a box"""
        if box_index in self.box_contents:
            self.box_contents[box_index].append({
                "color_name": obj.get("color_name"),
                "class_name": obj.get("class_name")
            })
    
    def reset_boxes(self) -> None:
        """Clear all box contents"""
        self.box_contents = {0: [], 1: [], 2: []}
    
    async def execute_commands(self, cmd_list: List[str]) -> bool:
        """Send command list to tasker for execution"""
        # Broadcast each command to UI
        if self.broadcast_callback:
            for i, cmd in enumerate(cmd_list):
                await self.broadcast_callback({
                    "type": "command",
                    "content": cmd,
                    "index": i + 1,
                    "total": len(cmd_list)
                })
        
        await self.send_command({"cmd_list": cmd_list})
        
        # Wait for stop_listen acknowledgment
        response = await self.wait_for_response()
        if not response:
            return False
        
        # Wait for start_listen when execution completes
        response = await self.wait_for_response(timeout=120.0)
        
        # Broadcast completion and refresh objects
        if response and self.broadcast_callback:
            # Get fresh object list
            objects = await self.get_objects()
            await self.broadcast_callback({
                "type": "objects",
                "objects": objects
            })
            await self.broadcast_callback({
                "type": "system",
                "content": f" Completed {len(cmd_list)//2} operations"
            })
        
        return response is not None
    
    async def execute_commands_sequential(self, grab_coords: List[str], drop_coords: List[str]) -> bool:
        """Execute GRAB & DROP commands one by one, updating UI after each pair"""
        total_pairs = len(grab_coords)
        
        # Get fresh object list before starting
        await self.get_objects()
        logger.info(f"🎯 Starting sequential execution with {len(self.current_objects)} objects available")
        
        for i, (grab, drop) in enumerate(zip(grab_coords, drop_coords)):
            pair_num = i + 1
            
            # Find which object is being grabbed (must do this BEFORE execution)
            grabbed_obj = None
            for obj in self.current_objects:
                obj_coords = f"{obj['x']} {obj['y']} {obj['z']}"
                logger.info(f"  Comparing: obj_coords='{obj_coords}' with grab='{grab}'")
                if obj_coords == grab:
                    grabbed_obj = obj
                    logger.info(f"   Found matching object: {obj}")
                    break
            
            if not grabbed_obj:
                logger.warning(f" Could not find object at coordinates: {grab}")
            
            # Broadcast which pair we're executing
            if self.broadcast_callback:
                await self.broadcast_callback({
                    "type": "command",
                    "content": f"GRAB {grab}",
                    "index": pair_num * 2 - 1,
                    "total": total_pairs * 2
                })
                await self.broadcast_callback({
                    "type": "command", 
                    "content": f"DROP {drop}",
                    "index": pair_num * 2,
                    "total": total_pairs * 2
                })
            
            # Execute this pair
            cmd_list = [f"GRAB {grab}", f"DROP {drop}"]
            await self.send_command({"cmd_list": cmd_list})
            
            # Wait for stop_listen acknowledgment
            response = await self.wait_for_response()
            if not response:
                logger.error(f" Failed to get stop_listen for pair {pair_num}")
                return False
            
            # Wait for start_listen when execution completes
            response = await self.wait_for_response(timeout=120.0)
            if not response:
                logger.error(f" Failed to get start_listen for pair {pair_num}")
                return False
            
            # Track which box the object was dropped into
            if grabbed_obj:
                box_id = self.get_box_id(drop)
                logger.info(f" Grabbed object: {grabbed_obj}, drop coords: {drop}, box_id: {box_id}")
                if box_id is not None:
                    self.add_to_box(box_id, grabbed_obj)
                    # Find the label for logging
                    box_label = next((box["label"] for box in self.drop_points if box["id"] == box_id), f"BOX{box_id}")
                    logger.info(f" Added {grabbed_obj['color_name']} {grabbed_obj['class_name']} to {box_label}")
                    logger.info(f" Box contents now: {self.box_contents}")
                else:
                    logger.warning(f" Could not find box for drop coords: {drop}")
            
            # Update objects list after each operation
            if self.broadcast_callback:
                objects = await self.get_objects()
                await self.broadcast_callback({
                    "type": "objects",
                    "objects": objects
                })
                # Send updated box contents
                await self.broadcast_callback({
                    "type": "box_contents",
                    "boxes": self.box_contents
                })
                await self.broadcast_callback({
                    "type": "system",
                    "content": f" Completed operation {pair_num}/{total_pairs}"
                })
            
            # Small delay for better UI experience
            await asyncio.sleep(0.5)
        
        return True


# Global instance
robot_server = RobotServer()

