"""
Tasker simulator - simulates the robot controller client
Based on the provided example script
"""
import socket
import signal
import sys
import time
import json

HOST = '127.0.0.1' 
PORT = 8000        

run_program = True

def handler(signum, frame):
    print("Terminate called")
    global run_program
    run_program = False

signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGTERM, handler)


object_list_counter = 0

# Sample object lists with different scenarios
object_list_cmd_1 = '{"objects": ["2 1 271 95 100", "3 4 271 131 100", "4 3 -42 113 100", "4 4 151 59 100", "4 1 390 57 100", "3 6 201 14 100", "2 1 595 -16 100", "1 3 94 53 100", "1 1 127 71 100", "1 1 298 30 100", "5 1 200 80 100"]}'
object_list_cmd_2 = '{"objects": ["2 1 564 149 100", "3 1 397 118 100", "4 2 308 65 100", "4 2 474 88 100", "4 0 506 -28 100", "3 1 418 64 100", "2 0 430 84 100", "1 4 515 114 100", "1 1 390 4 100", "1 1 272 -1 100", "5 1 242 56 100"]}'
object_list_cmd = [object_list_cmd_1, object_list_cmd_2]

drop_point_to_test = {}
drop_point_to_test["drop_points"] = []

def object_cmd_to_string_list(cmd):
    """Convert object list to coordinate strings"""
    data = json.loads(cmd)
    result = []
    for i in data["objects"]:
        parts = i.split(" ")
        test_string = parts[2] + " " + parts[3] + " " + parts[4]
        result.append(test_string)
    return result


while run_program:
    print("Starting tasker simulator...")
    print(f"Connecting to {HOST}:{PORT}")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:    
            client_socket.connect((HOST, PORT))
            print("Connected to the server")
            client_socket.settimeout(1.0)

            object_list_counter_prv = 0

            while run_program:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        print("Server disconnected")
                        break
                    
                    decoded = data.decode().strip()
                    print(f"\nReceived from server: {decoded}")
                    
                    payload = json.loads(decoded)

                    # Handle different commands
                    if "cmd" in payload:
                        cmd = payload["cmd"]
                        
                        if cmd == "clear_drop_points":
                            drop_point_to_test["drop_points"].clear()
                            print("Drop points cleared")
                            
                        elif cmd == "set_drop_point":
                            coords = payload.get("coordinates")
                            drop_point_to_test["drop_points"].append(coords)
                            print(f"Drop point added: {coords}")
                            
                        elif cmd == "start_listen":
                            print("Started listening mode")
                            
                        elif cmd == "stop_listen":
                            print("Stopped listening mode")
                            
                        elif cmd == "give_objects":
                            # Send object list
                            response = object_list_cmd[object_list_counter]
                            print(f"Sending objects: {response}")
                            client_socket.send(response.encode('utf-8'))
                            time.sleep(0.5)
                            object_list_counter_prv = object_list_counter
                            object_list_counter = (object_list_counter + 1) % 2
                    
                    elif "cmd_list" in payload:
                        print("\n=== Executing command list ===")
                        
                        # Send stop_listen acknowledgment
                        stop_msg = '{"cmd":"stop_listen"}'
                        client_socket.send(stop_msg.encode("utf-8"))
                        time.sleep(0.5)
                        
                        # Process commands
                        tasker_cmd_list = payload["cmd_list"]
                        objects_to_test = object_cmd_to_string_list(
                            object_list_cmd[object_list_counter_prv]
                        )
                        
                        prv_cmd = 1
                        errors = []
                        
                        for task_string in tasker_cmd_list:
                            parts = task_string.split(" ")
                            
                            if parts[0] == "GRAB":
                                if prv_cmd == 1:
                                    prv_cmd = 0
                                    target_coord = f"{parts[1]} {parts[2]} {parts[3]}"
                                    
                                    if target_coord in objects_to_test:
                                        print(f"✓ GRAB {target_coord}")
                                    else:
                                        error_msg = f"ERROR: Grab coords not found: {target_coord}"
                                        print(error_msg)
                                        errors.append(error_msg)
                                else:
                                    error_msg = "ERROR: Bad command pairs (GRAB after GRAB)"
                                    print(error_msg)
                                    errors.append(error_msg)
                            
                            elif parts[0] == "DROP":
                                if prv_cmd == 0:
                                    prv_cmd = 1
                                    target_coord = f"{parts[1]} {parts[2]} {parts[3]}"
                                    
                                    if target_coord in drop_point_to_test["drop_points"]:
                                        print(f"✓ DROP {target_coord}")
                                    else:
                                        error_msg = f"ERROR: Drop point not found: {target_coord}"
                                        print(error_msg)
                                        errors.append(error_msg)
                                else:
                                    error_msg = "ERROR: Bad command pairs (DROP after DROP)"
                                    print(error_msg)
                                    errors.append(error_msg)
                        
                        if errors:
                            print(f"\n⚠ Completed with {len(errors)} error(s)")
                        else:
                            print("\n✓ All commands executed successfully")
                        
                        print("=== Task execution complete ===\n")
                        
                        # Send start_listen to indicate completion
                        start_msg = '{"cmd":"start_listen"}'
                        client_socket.send(start_msg.encode('utf-8'))
                        time.sleep(0.5)

                except socket.timeout:
                    pass
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error: {e}")
                    break

    except ConnectionRefusedError:
        print("Connection refused. Retrying in 5 seconds...")
        time.sleep(5)
    except Exception as e:
        print(f"Connection error: {e}")
        time.sleep(5)

print("Exiting tasker simulator")

