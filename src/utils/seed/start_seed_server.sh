#!/bin/bash
# Copyright 2025 POLAR Team and/or its affiliates
#
# SEED Score Server Startup Script (Multi-Process Mode)
#
# Usage:
#   ./start_seed_server.sh                    # Default: 8 workers, port 30001
#   ./start_seed_server.sh -w 16 -p 30002     # Custom workers and port
#   ./start_seed_server.sh --help             # Show help
#
# To run in background:
#   nohup ./start_seed_server.sh > seed_server.log 2>&1 &

set -e

# Default configuration
HOST="0.0.0.0"
PORT=30001
MAX_WORKERS=8
TIMEOUT=60.0
MAX_TASKS_PER_CHILD=100

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -m|--max-tasks-per-child)
            MAX_TASKS_PER_CHILD="$2"
            shift 2
            ;;
        --help)
            echo "SEED Score Server Startup Script (Multi-Process Mode)"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -h, --host HOST                 Host address to bind (default: 0.0.0.0)"
            echo "  -p, --port PORT                 Port number (default: 30001)"
            echo "  -w, --workers N                 Max worker processes (default: 8)"
            echo "  -t, --timeout SECS              Computation timeout (default: 60.0)"
            echo "  -m, --max-tasks-per-child N     Restart worker after N tasks (default: 100)"
            echo "  --help                          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Use defaults"
            echo "  $0 -w 32 -p 30030               # 32 workers on port 30030"
            echo "  $0 -m 50                        # Restart workers more frequently"
            echo "  $0 -w 32 -m 200                 # 32 workers, restart every 200 tasks"
            echo ""
            echo "Memory Management:"
            echo "  Workers auto-restart after max-tasks-per-child to prevent memory leak."
            echo "  GET /stats shows memory usage of main process and all children."
            echo ""
            echo "To run in background:"
            echo "  nohup $0 > seed_server.log 2>&1 &"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Add project to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "============================================"
echo "   SEED Score Server (Multi-Process Mode)"
echo "============================================"
echo "Host:                ${HOST}"
echo "Port:                ${PORT}"
echo "Workers:             ${MAX_WORKERS} processes"
echo "Timeout:             ${TIMEOUT}s"
echo "Max Tasks Per Child: ${MAX_TASKS_PER_CHILD} (auto-restart)"
echo "Project:             ${PROJECT_ROOT}"
echo "============================================"
echo ""
echo "Starting server..."
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python "${SCRIPT_DIR}/seed_server.py" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-workers "${MAX_WORKERS}" \
    --timeout "${TIMEOUT}" \
    --max-tasks-per-child "${MAX_TASKS_PER_CHILD}"
