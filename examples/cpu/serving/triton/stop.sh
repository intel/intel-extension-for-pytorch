#!/bin/bash

# Copyright (c) 2023 Intel Corporation
# SPDX-License-Identifier: Apache 2.0

source "$(pwd)"/config.properties

print_help() {
    echo ""
    echo "Usage: $0 [arg]"
    echo ""
    echo "   This script stops Docker container for host, client or localhost scenario."
    echo "   If no arguments are specified script will stop Docker containers for localhost."
    echo "   To perform full cleanup script should be run with sudo privileges."
    echo ""
    echo "   Available arguments:"
    echo "   client              - stops client container."
    echo "   host                - stops host container."
    echo "   localhost (default) - stops client and host containers on the same instance."
    echo ""
    exit 2
}

stop_host() { 
    check_state=$(docker ps | grep -c "${image_name}_host")
    [ "$check_state" -ne 0 ] && ( echo "Stopped: " && docker rm -f "${image_name}_host" ) || echo "No running host server to stop"
    [ -d "$(pwd)"/backend ] && echo "Cleanup: removing $(pwd)/backend" && rm -fr "$(pwd)"/backend
}

stop_client() { 
    check_state=$(docker ps | grep -c "${image_name}_client")
    [ "$check_state" -ne 0 ] && echo "Stopped: " && docker rm -f "${image_name}_client" || echo "No running client server to stop"
}

declare -l argument=${1:-"localhost"}

echo ""
case "$argument" in
    "client")
        echo ""
        echo "Trying to stop client server"
        echo ""
        stop_client
        ;;
    "host")
        echo ""
        echo "Trying to stop host server"
        echo ""
        stop_host
        ;;
    "localhost")
        echo ""
        echo "Trying to stop both client and host server for localhost deployment"
        echo ""
        stop_client
        stop_host
        ;;
    *)
	    print_help
        ;;
esac
echo ""