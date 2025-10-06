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
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from robot_server import robot_server


# Define agent state
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    objects: List[Dict]
    plan: str
    status: str


# Tool definitions
@tool
def get_detected_objects() -> str:
    """
    Get list of currently detected objects from the robot camera.
    Returns objects with their ID, class, color, and coordinates.
    """
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    objects = loop.run_until_complete(robot_server.get_objects())
    loop.close()
    
    if not objects:
        return "No objects detected"
    
    result = "Detected objects:\n"
    for obj in objects:
        result += f"- {obj['color_name']} {obj['class_name']} at ({obj['x']}, {obj['y']}, {obj['z']})\n"
    
    return result


@tool
def execute_robot_commands(
    grab_coords: List[str],
    drop_coords: List[str]
) -> str:
    """
    Execute GRAB and DROP commands on the robot.
    
    Args:
        grab_coords: List of coordinates to grab from, format: ["x y z", ...]
        drop_coords: List of coordinates to drop to, format: ["x y z", ...]
        
    Each grab must be followed by a drop. The lists must have the same length.
    Available drop points: "50 60 14", "550 160 15", "850 180 16"
    """
    if len(grab_coords) != len(drop_coords):
        return "Error: Number of grab and drop coordinates must match"
    
    cmd_list = []
    for grab, drop in zip(grab_coords, drop_coords):
        cmd_list.append(f"GRAB {grab}")
        cmd_list.append(f"DROP {drop}")
    
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(robot_server.execute_commands(cmd_list))
    loop.close()
    
    if success:
        return f"Successfully executed {len(grab_coords)} grab-drop pairs"
    else:
        return "Failed to execute commands"


# Create tools list
tools = [get_detected_objects, execute_robot_commands]


class RobotAgent:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(
            model="gpt-4",
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
            messages = state["messages"]
            
            # Add system message with context
            system_msg = SystemMessage(content="""You are a helpful robot control assistant. 

You have access to:
1. get_detected_objects() - to see what objects the robot camera detects
2. execute_robot_commands() - to send grab/drop commands to the robot

When user asks to manipulate objects:
1. First call get_detected_objects() to see what's available
2. Create a plan based on the user's request (e.g., "throw out all red balls")
3. Extract coordinates of matching objects
4. Call execute_robot_commands() with grab coordinates and drop point coordinates
5. Report progress to the user

Available drop points: "50 60 14", "550 160 15", "850 180 16"

Object classes: can, duck, cup, sponge, ball, vegetable
Colors: red, blue, green, yellow, black, grey, white, violet, orange, turquoise

Be concise and clear in your responses.""")
            
            messages_with_system = [system_msg] + messages
            response = self.llm_with_tools.invoke(messages_with_system)
            
            return {"messages": [response]}
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))
        
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
        state = {
            "messages": [HumanMessage(content=message)],
            "objects": [],
            "plan": "",
            "status": "processing"
        }
        
        result = await self.graph.ainvoke(state)
        
        # Extract final AI response
        messages = result["messages"]
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                return msg.content
        
        return "I processed your request."


def create_agent(api_key: str = None) -> RobotAgent:
    """Create robot control agent"""
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    return RobotAgent(api_key)

