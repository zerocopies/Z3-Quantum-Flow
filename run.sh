#!/bin/bash
QFLOW=/home/prp/Z3-Quantum-Flow/target/release/qflow
MODELS_DIR=/home/prp/Z3-Quantum-Flow/ai_playground
echo "==================================="
echo "  Z3 Quantum-Flow Model Selector"
echo "==================================="
echo "1) Qwen2.5-Coder 3B  (2.4 tok/s)"
echo "2) Qwen2.5-Coder 1.5B (fastest)"
echo "3) Qwen3.5 4B Aggressive"
echo "4) Phi-3-mini"
echo "5) Llama 3.1 8B"
echo "==================================="
read -p "Select [1-5]: " choice
case $choice in
    1) MODEL="$MODELS_DIR/qwen2.5-coder-3b-q4_K_M.gguf" ;;
    2) MODEL="$MODELS_DIR/qwen2.5-coder-1.5b-q4_K_M.gguf" ;;
    3) MODEL="$MODELS_DIR/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf" ;;
    4) MODEL="$MODELS_DIR/phi-3-mini-q4_K_M.gguf" ;;
    5) MODEL="$MODELS_DIR/llama-3.1-8b-instruct-q4_k_m_lmu.gguf" ;;
    *) echo "Invalid"; exit 1 ;;
esac
Z1_CTX_SIZE=2048 $QFLOW "$MODEL"
