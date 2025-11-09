"""
LangGraph agent for robot control
Uses OpenAI GPT with tools to query objects and execute robot commands
"""
import os
import json
from typing import List, Dict, Any, Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage

from robot_server import robot_server


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    objects: List[Dict]
    plan: str
    status: str


# Tool definitions
@tool
async def get_system_status() -> str:
    """
    Get current system status including robot connection, box contents, and available resources.
    Use this to check system state, connection status, and what's in each box.
    
    Returns comprehensive information about:
    - Robot connection status
    - Number of detected objects available
    - Number of drop boxes configured with their coordinates
    - Contents of each box (what objects have been placed in them)
    - Session command history count
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        status_parts = []
        
        # Robot connection status
        if robot_server.connected:
            status_parts.append("✓ Robot is CONNECTED and ready")
        else:
            status_parts.append("✗ Robot is NOT connected")
        
        # Object count
        obj_count = len(robot_server.current_objects)
        status_parts.append(f"- {obj_count} objects currently detected in scene")
        
        # Drop boxes and their contents
        box_count = len(robot_server.drop_points)
        status_parts.append(f"- {box_count} drop boxes configured:")
        
        for box in robot_server.drop_points:
            box_id = box['id']
            box_label = box['label']
            box_coords = box['coordinates']
            contents = robot_server.box_contents.get(box_id, [])
            
            if contents:
                items_desc = ", ".join([f"{item['color_name']} {item['class_name']}" for item in contents])
                status_parts.append(f"  • {box_label} at ({box_coords}): {len(contents)} item(s) - {items_desc}")
            else:
                status_parts.append(f"  • {box_label} at ({box_coords}): empty (0 items)")
        
        result = "\n".join(status_parts)
        logger.info(f"System status: connected={robot_server.connected}, objects={obj_count}, boxes={box_count}")
        return result
        
    except Exception as e:
        logger.error(f"get_system_status error: {e}")
        return f"Error getting system status: {str(e)}"


@tool
async def get_detected_objects() -> str:
    """
    Get list of currently detected objects from the robot camera.
    Returns objects with their ID, class, color, and coordinates.
    
    Use this to check what objects are available before executing any commands.
    If the user's request doesn't clearly match any objects, ask for clarification instead of guessing.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        objects = await robot_server.get_objects(retry_count=2)
        
        if not objects:
            logger.warning("get_detected_objects: No objects found after retries")
            return "No objects detected. The robot camera may not see any objects in the scene, or there may be a communication issue."
        
        result = "Detected objects:\n"
        for obj in objects:
            result += f"- {obj['color_name']} {obj['class_name']} at ({obj['x']}, {obj['y']}, {obj['z']})\n"
        
        logger.info(f"get_detected_objects: Found {len(objects)} objects")
        return result
    except Exception as e:
        logger.error(f"get_detected_objects error: {e}")
        return f"Error getting objects: {str(e)}"


@tool
async def execute_robot_commands(
    grab_coords: List[str],
    drop_coords: List[str]
) -> str:
    """
    Execute GRAB and DROP commands on the robot.
    
    ONLY call this function when you have a CLEAR MATCH between user's request and detected objects.
    If uncertain or ambiguous, ask the user for clarification instead.
    
    Args:
        grab_coords: List of coordinates to grab from, format: ["x y z", ...]
        drop_coords: List of coordinates to drop to, format: ["x y z", ...]
        
    Each grab must be followed by a drop. The lists must have the same length.
    IMPORTANT: Use get_system_status() first to see the current box coordinates.
    The robot can dynamically configure drop boxes, so always check their current coordinates.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if len(grab_coords) != len(drop_coords):
        logger.error(f"Coordinate mismatch: {len(grab_coords)} grabs, {len(drop_coords)} drops")
        return "Error: Number of grab and drop coordinates must match"
    
    logger.info(f"🤖 Executing {len(grab_coords)} grab-drop pairs sequentially")
    
    try:
        # This will wait for robot to actually complete all movements
        success = await robot_server.execute_commands_sequential(grab_coords, drop_coords)
        
        if success:
            logger.info(f"✅ Robot completed all {len(grab_coords)} operations")
            # Use cached state (no camera query)
            remaining = len(robot_server.current_objects)
            return f"Successfully completed all {len(grab_coords)} operations. {remaining} objects remaining in scene."
        else:
            logger.error("Command execution failed or robot did not respond")
            return "Failed to execute commands - robot may not have responded properly"
    except Exception as e:
        logger.error(f"execute_robot_commands error: {e}")
        return f"Error executing commands: {str(e)}"


# Create tools list
tools = [get_system_status, get_detected_objects, execute_robot_commands]


class RobotAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=api_key,
            temperature=0
        )
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create LangGraph workflow"""
        
        def should_continue(state: AgentState) -> str:
            """Determine if we should continue or end"""
            messages = state["messages"]
            last_message = messages[-1]
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            return "end"
        
        def call_model(state: AgentState) -> AgentState:
            """Call the LLM"""
            import logging
            logger = logging.getLogger(__name__)
            
            messages = state["messages"]
            
            # Add system message with context
            system_msg = SystemMessage(content="""You are a helpful robot control assistant. 

You have access to:
1. get_system_status() - check robot connection, see box coordinates and contents
2. get_detected_objects() - see what objects the robot camera detects
3. execute_robot_commands() - send grab/drop commands to the robot

When user asks about system state (connection, status, boxes, etc):
- ALWAYS call get_system_status() to get current state
- You can answer questions about box contents, objects, connection status even if robot is not connected
- The system maintains state - boxes can have objects in them from previous operations
- Give clear, direct answers based on the actual state data

When user asks to manipulate objects:
1. FIRST call get_system_status() to see current box coordinates (robot can update them dynamically)
2. Call get_detected_objects() to see what's available
3. Check if the user's request CLEARLY matches available objects
4. If UNCERTAIN or NO CLEAR MATCH:
   - DO NOT attempt to execute commands
   - Ask the user to clarify which specific object(s) they mean
   - List the available objects that might be relevant
   - Ask them to be more specific (e.g., specify color, type, or location)
5. If CLEAR MATCH found:
   - Create a plan based on the user's request
   - Extract coordinates of matching objects
   - Match user's drop location request (e.g., BOX1, BOX2) to the box coordinates from get_system_status()
   - Call execute_robot_commands() with grab coordinates and the CURRENT box coordinates
   - Report progress to the user

Drop boxes: The robot can dynamically configure drop boxes (BOX1, BOX2, BOX3, etc).
ALWAYS get the current box coordinates from get_system_status() before executing commands.
When user says "put in BOX1" or "drop to BOX2", use the box coordinates shown in get_system_status().

Object classes: can, duck, cup, sponge, ball, vegetable
Colors: black, white, grey, red, blue, green, cyan, magenta, yellow

IMPORTANT: 
- Never guess or assume which objects the user wants. Always ask for clarification if there's any ambiguity. Safety first!
- Box coordinates can change - ALWAYS check get_system_status() before executing drop commands
- You have access to full conversation history - you remember what commands were executed before
- The system maintains state across the session - box contents persist even if robot disconnects

Be concise and clear in your responses.""")
            
            messages_with_system = [system_msg] + messages
            response = self.llm_with_tools.invoke(messages_with_system)
            
            # Log what the LLM decided to do
            if hasattr(response, 'tool_calls') and response.tool_calls:
                logger.info(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            else:
                logger.info(f"LLM response (no tools): {response.content[:200] if response.content else 'empty'}")
            
            return {"messages": [response]}
        
        async def call_tools(state: AgentState) -> AgentState:
            """Execute tool calls"""
            import logging
            import asyncio
            logger = logging.getLogger(__name__)
            
            messages = state["messages"]
            last_message = messages[-1]
            
            tool_messages = []
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                logger.info(f"🔧 Calling tool: {tool_name} with args: {tool_args}")
                
                # Find and execute the tool
                tool_func = None
                for t in tools:
                    if t.name == tool_name:
                        tool_func = t
                        break
                
                if tool_func:
                    # Call the tool function
                    try:
                        import inspect
                        
                        # Find the actual coroutine function
                        # Try different possible attributes where the async function might be stored
                        actual_func = None
                        for attr in ['coroutine', 'afunc', 'func', '_run_func', 'run_func']:
                            if hasattr(tool_func, attr):
                                f = getattr(tool_func, attr)
                                if f is not None:
                                    actual_func = f
                                    logger.info(f"🔍 Found function at .{attr}: {type(f)}")
                                    break
                        
                        # Check if we have an async function
                        if actual_func and inspect.iscoroutinefunction(actual_func):
                            logger.info(f"Calling async function directly")
                            result = await actual_func(**tool_args)
                        else:
                            # Use coroutine-based invocation
                            logger.info(f"Using coroutine invocation")
                            result = await tool_func.arun(**tool_args)
                    except Exception as e:
                        logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
                        result = f"Error: {str(e)}"
                    
                    logger.info(f"Tool {tool_name} returned: {result[:200] if len(str(result)) > 200 else result}")
                    tool_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                else:
                    logger.error(f"Tool {tool_name} not found!")
            
            return {"messages": tool_messages}
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()
    
    async def process_message(self, message: str) -> str:
        """Process user message and return response"""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Processing message: {message}")
        
        state = {
            "messages": [HumanMessage(content=message)],
            "objects": [],
            "plan": "",
            "status": "processing"
        }
        
        result = await self.graph.ainvoke(state)
        
        logger.info(f"Graph execution complete. Message count: {len(result['messages'])}")
        
        # Extract final AI response
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                logger.info(f"Final response: {msg.content[:200]}")
                return msg.content
        
        return "I processed your request."


def create_agent(api_key: str = None) -> RobotAgent:
    """Create robot control agent"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    return RobotAgent(api_key)

