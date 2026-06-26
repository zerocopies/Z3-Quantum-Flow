#!/bin/bash

QFLOW=/home/prp/Z3-Quantum-Flow/target/release/qflow
MODELS_DIR=/home/prp/Z3-Quantum-Flow/ai_playground

echo "==================================="
echo "  Z3 Quantum-Flow Model Selector"
echo "==================================="
echo "1) Qwen2.5-Coder 3B  (fast,  2.4 tok/s, code)"
echo "2) Phi-3-mini         (small, 0.3 tok/s, general)"
echo "3) Llama 3.1 8B       (large, 0.7 tok/s, general)"
echo "==================================="
read -p "Select model [1-3]: " choice

case $choice in
    1) MODEL="$MODELS_DIR/qwen2.5-coder-3b-q4_K_M.gguf" ;;
    2) MODEL="$MODELS_DIR/phi-3-mini-q4_K_M.gguf" ;;
    3) MODEL="$MODELS_DIR/llama-3.1-8b-instruct-q4_k_m_lmu.gguf" ;;
    *) echo "Invalid choice"; exit 1 ;;
esac

RUST_LOG=info Z1_CTX_SIZE=2048 $QFLOW "$MODEL"
