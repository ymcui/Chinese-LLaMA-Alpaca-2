#!/bin/bash

# NOTE: start the server first before running this script.
# usage: ./server_curl_example.sh your-instruction

SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
# SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.
INSTRUCTION=$1
ALL_PROMPT="[INST] <<SYS>>\n$SYSTEM_PROMPT\n<</SYS>>\n\n$INSTRUCTION [/INST]"
CURL_DATA="{\"prompt\": \"$ALL_PROMPT\",\"n_predict\": 128}"

curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data "$CURL_DATA"
