#!/bin/bash

# Run the tasker simulator

echo "Starting tasker simulator..."
echo "This simulates the robot controller (tasker client)"
echo ""
echo "Connecting to TCP server at 127.0.0.1:8000"
echo "Press Ctrl+C to stop"
echo ""

cd tests
python tasker_simulator.py

