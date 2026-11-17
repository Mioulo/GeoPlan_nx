#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$here/yopo_trt_transfer.py" ]]; then
    runtime="$here"
else
    runtime="$here/../../YOPO"
fi
cd "$runtime"
exec "${PYTHON:-python3}" yopo_trt_transfer.py \
    --weight saved/YOPO_116/epoch160.pth \
    --training_config saved/YOPO_116/ablation_config.json \
    --split_encoder --opset 16 --fp16 1 --benchmark 100 \
    --encoder_onnx proposed116_encoder.onnx \
    --planner_onnx proposed116_planner.onnx \
    --encoder_engine proposed116_encoder.engine \
    --planner_engine proposed116_planner.engine "$@"
