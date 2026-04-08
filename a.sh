for task in easy medium hard; do
  export COMPILER_TASK=$task
  echo "=== Task: $task ==="
  python3 inference.py
  echo ""
done