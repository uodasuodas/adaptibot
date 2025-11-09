import socket
import signal
import sys
import select
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



object_list_counter = 1
object_list_counter_prv = 1

start_listening_cmd = '{"cmd":"start_listen"}'
stop_listening_cmd = '{"cmd":"stop_listen"}'
clear_drop_points_cmd = '{"cmd":"clear_drop_points"}'
set_drop_point_cmd_1 = '{"cmd":"set_drop_point","coordinates":"50 60 14"}'
set_drop_point_cmd_2 = '{"cmd":"set_drop_point","coordinates":"550 160 15"}'
set_drop_point_cmd_3 = '{"cmd":"set_drop_point","coordinates":"850 180 16"}'

object_list_cmd_1 = '{"objects": ["2 2 271 95 100", "3 4 271 131 100", "4 3 -42 113 100", "4 4 151 59 100", "4 2 390 57 100", "3 6 201 14 100", "2 1 595 -16 100", "1 3 94 53 100", "1 2 127 71 100", "1 2 298 30 100", "2 3 200 80 100"]}'
object_list_cmd_2 = '{"objects": ["2 1 564 149 100", "3 1 397 118 100", "4 2 308 65 100", "4 2 474 88 100", "4 0 506 -28 100", "3 1 418 64 100", "2 0 430 84 100", "1 4 515 114 100", "1 1 390 4 100", "1 1 272 -1 100", "2 1 242 56 100"]}'
object_list_cmd = [object_list_cmd_1,object_list_cmd_2]

#jau gautos komandos, processinimui
tasker_cmd_list = []

drop_point_to_test = {}
drop_point_to_test["drop_points"] = []
drop_point_to_test["drop_points"].append((json.loads(set_drop_point_cmd_1))["coordinates"])
drop_point_to_test["drop_points"].append((json.loads(set_drop_point_cmd_2))["coordinates"])
drop_point_to_test["drop_points"].append((json.loads(set_drop_point_cmd_3))["coordinates"])

def object_cmd_to_string_list( cmd ):
    data = json.loads(cmd)
    result = []
    for i in data["objects"]:
        tstSp = i.split(" ")
        test_string = tstSp[2]+" " +tstSp[3]+" "+ tstSp[4]
        result.append(test_string)
    return result


while run_program:
    print("Starting")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:    
        if client_socket.connect((HOST, PORT)) == None:
            print("Connected to the server!")
            client_socket.setblocking(False)

            # Sessijos prep, viska nudefaultina ir sudeda pagrindinius pointus
            client_socket.send(clear_drop_points_cmd.encode('utf-8'))
            time.sleep(1) # Kad nesuklijuotu i ta pati viena bufferio readinima, bus paprasciau parsinti
            client_socket.send(set_drop_point_cmd_1.encode('utf-8'))
            time.sleep(1)
            client_socket.send(set_drop_point_cmd_2.encode('utf-8'))
            time.sleep(1)
            client_socket.send(set_drop_point_cmd_3.encode('utf-8'))
            time.sleep(1)
            client_socket.send(start_listening_cmd.encode('utf-8'))
            time.sleep(1)

            while run_program:
                try:
                    data = client_socket.recv(1024)
                    if not data:  # empty means connection closed
                        print("Server disconnected")
                        break
                    decoded = data.decode()
                    payload = json.loads(decoded)

                    #Komando brenchas no error handling'o kad specialiai kreshintu
                    if "cmd" in payload:
                        payload["cmd"] == "give_objects"
                        client_socket.send(object_list_cmd[object_list_counter].encode('utf-8'))
                        time.sleep(1)
                        object_list_counter_prv = object_list_counter
                        object_list_counter = 1 #(object_list_counter+1)%2
                    if "cmd_list" in payload:
                        #uzblokuojamas LLM klausymas, nuo cia LLM nebeturetu klausyti zmogaus inputu, iki kol bus atidarytas vel
                        client_socket.send(stop_listening_cmd.encode("utf-8"))
                        time.sleep(1)
                        print("Gautos komandos :")
                        tasker_cmd_list.clear()
                        for i in payload["cmd_list"]:
                            print(i)
                            tasker_cmd_list.append(i)
                        
                        #############
                        # Prasideda gautu komandu testavimas ir grojimas
                        #############

                        objects_to_test = object_cmd_to_string_list(object_list_cmd[object_list_counter_prv])

                        prvCmd = 1;
                        for taskString in tasker_cmd_list:
                            cmdSpl = taskString.split(" ")
                            tast = {}
                            if cmdSpl[0] == "GRAB":
                                if prvCmd == 1:
                                    prvCmd = 0

                                    # Testuojama ar objeto koordinates egzituoja is siustu objekto koordinaciu
                                    target_coord_string = cmdSpl[1]+" "+ cmdSpl[2]+" "+ cmdSpl[3]
                                    
                                    matchFound = False
                                    for t in objects_to_test:
                                        if t == target_coord_string:
                                            matchFound = True
                                            break
                                    if not matchFound:
                                        print("ERROR: Grab object coords do not exist" , target_coord_string)
                                else:
                                    print("ERROR: Bad cmd pairs")

                            if cmdSpl[0] == "DROP":
                                if prvCmd == 0:
                                    prvCmd = 1
                                    target_coord_string = cmdSpl[1]+" "+ cmdSpl[2]+" "+ cmdSpl[3]
                                    if not target_coord_string in drop_point_to_test["drop_points"]:
                                        print("ERROR: Drop point coords do not exist" , target_coord_string)

                                else:
                                    print("ERROR: Bad cmd pairs")

                        print("Tasker run done")

                        # Robotas pabaiges visus taskus atidaro klausymo rezima
                        client_socket.send(start_listening_cmd.encode('utf-8'))
                        time.sleep(1)

                except BlockingIOError:
                    pass
                time.sleep(1)

    
print("Exiting")
