#!/bin/bash
echo "Starting Quantum-Flow..."
fuser -k 7474/tcp 2>/dev/null
sleep 1
cd /home/prp/prubbs/buffer-zone
Z1_CTX_SIZE=2048 ./target/release/buffer_zone &
BZ_PID=$!
sleep 2
curl -s -X POST http://127.0.0.1:7474/load_model \
  -H "Content-Type: application/json" \
  -d '{"path":"/home/prp/Z3-Quantum-Flow/ai_playground/Qwen2.5-Coder-3B-Instruct-abliterated-Q4_K_M.gguf"}' > /dev/null
echo "Ready — open http://127.0.0.1:7474"
wait $BZ_PID
