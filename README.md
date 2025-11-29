<p align="center">
  <img width="620" height="564" alt="image" src="https://github.com/user-attachments/assets/eb536db2-e89b-40af-b277-e00b445e054f" />
</p>

## Game Overview

Rivers and Stones is a two-player deterministic, turn-based board game played on three possible grid sizes (13×12, 15×14, 17×16). Each player controls identical pieces, each having two faces:

- Stone — scoring side

- River — mobility side (horizontal or vertical flow)

## Goal

Fill the player's scoring area (SA) — located near the opponent’s starting side — with the required number of Stones (4/5/6 depending on board size). A player wins immediately upon filling their SA or if the opponent runs out of time.

## Allowed Actions

- On each turn, a player may perform exactly one of the following:
- Move a Stone or River by one step (with restrictions)

- Push an adjacent piece (rules differ for Stone-push and River-push)

- Flip a piece (Stone ↔ River, choose orientation if flipped to River)

- Rotate a River by 90°

- Movement on Rivers allows multi-step traversal along flow direction, possibly chaining through multiple Rivers until blocked.
<p align="cente">
<img width="1215" height="9117" alt="image" src="https://github.com/user-attachments/assets/ef1f743b-dae4-4dce-90d5-a0c4d5855c2b" />
</p>

## Additional Rules

- No move may enter or cross the opponent’s scoring area.

- Only Stone faces can score.

- Pieces in scoring areas remain active and may still be moved, flipped, or pushed.

## Dependencies
- Python 3.9
- Pygame
- Numpy 
- Scipy
- flask
- python-socketio


## Setting up the Environment
```sh
pip install -r requirements.txt
```

## Main Files
- `gameEngine.py`: It is an instance of the game. It can be run locally on your environment. You can run in GUI or CLI mode.
- `agent.py`: It consists of the implementations of the Random Agent. 
- `student_agent.py` : The Hueristic agent is here.
- `web_server.py` : Use it to start the webserver.
- `bot_client.py`: It calls student_agent to get the moves while interacting with the web server. 

## Run Instructions
### Human vs Human
```sh
python gameEngine.py --mode hvh
```
### Human vs AI

```sh
python gameEngine.py --mode hvai --circle random
```
### AI vs AI

```sh
python gameEngine.py --mode aivai --circle random --square student
```

### No GUI
```sh
python gameEngine.py --mode aivai --circle random --square student --nogui
```

### Create server
```sh
bash start_server.sh 8080
```
Ensure that conda env is set before you start the server. Once server starts, you can navigate to http://localhost:8080" or whatever port you choose. Once to go to webpage, select boardsize and then click start game. Then we move to starting bots using following commands.
#### Starting first bot
```sh
python bot_client.py circle 8080 --strategy student
```
#### Running Second bot
```sh
python bot_client.py square 8080 --strategy student --server 10.10.10.10
```
--server will require the IP of system on which server is running. You can use ipconfig/ifconfig to identify the IP address. Provide that IP and it should connect. 

##  C++ Sample Files

It contains following files. You are allowed to create the files of your own. But player file must be named at student_agent.py

- student_agent_cpp.py - Serves as a wrapper between python and c++.
- student_agent.cpp - Can be used to write your c++ code.
- CMakeLists.txt - CMake file

### Dependencies
pybind11

### Installation
```sh
pip install pybind11
```

### Setting up the C++ Agent

```sh
mkdir build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
```
Run:

```sh
make
cd ..
```

## Running for a C++ program.

You need to use student_cpp rather than student for this case.

```sh
python gameEngine.py --mode aivai --circle random --square student_cpp
```

