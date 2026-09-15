#!/bin/bash
#SBATCH --partition=eic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=benchmarking/runs/yolo_visual_%j.out
#SBATCH --error=benchmarking/runs/yolo_visual_%j.err
#SBATCH --job-name=yolo_visual

set -euo pipefail
cd /home/superhans/Master_Thesis/DNAO-Analysis-Tool

# Visual inspection: evaluate key models with plots enabled
declare -A VISUAL_MODELS=(
  # Best performing model
  ["single_bowtie"]="single_bowtie.seed42.v1"
  # Worst performing model  
  ["leave_out_triangle"]="leave_out_triangle.seed42.v1"
  # Average performer
  ["pooled_shape"]="pooled_shape.seed42.v1"
)

declare -A MODEL_WEIGHTS=(
  ["single_bowtie"]="benchmarking/runs/segment/single_bowtie.seed42.v1_yolo4/weights/best.pt"
  ["leave_out_triangle"]="benchmarking/runs/segment/leave_out_triangle.seed42.v1_20250812_203924/weights/best.pt"
  ["pooled_shape"]="benchmarking/runs/segment/pooled_shape.seed42.v1_20250819_100449/weights/best.pt"
)

OUT_ROOT="benchmarking/results/yolo_visual_inspection"
mkdir -p "$OUT_ROOT"

echo "==========================================="
echo "YOLO Visual Inspection (plots enabled)"
echo "Models: ${#VISUAL_MODELS[@]}"
echo "==========================================="

for model_name in "${!VISUAL_MODELS[@]}"; do
  weight_path="${MODEL_WEIGHTS[$model_name]}"
  split="${VISUAL_MODELS[$model_name]}"
  
  echo ""
  echo "=== Visual Inspection: $model_name ==="
  echo "Weight: $weight_path"
  echo "Test split: $split"
  
  # Check if split exists
  split_dir="benchmarking/data/splits/$split"
  test_txt="$split_dir/test.txt"
  
  if [ ! -f "$test_txt" ]; then
    echo "ERROR: Test file not found: $test_txt"
    continue
  fi
  
  # Create output directory
  out_dir="$OUT_ROOT/${model_name}"
  mkdir -p "$out_dir"
  
  # Create temporary YAML for this evaluation
  temp_yaml=$(mktemp --suffix=.yaml)
  cat > "$temp_yaml" << EOF
path: /home/superhans/Master_Thesis/DNAO-Analysis-Tool
train: $test_txt
val: $test_txt
test: $test_txt
nc: 1
names: ['origami']
EOF
  
  echo "Running YOLO evaluation with plots..."
  
  # Run YOLO validation with PLOTS ENABLED for visual inspection
  yolo segment val model="$weight_path" data="$temp_yaml" split=test \
    save_json=False save_txt=False plots=True verbose=True \
    project="$out_dir" name="visual" exist_ok=True
  
  # Clean up temp file
  rm "$temp_yaml"
  
  echo "-> Visualizations saved to $out_dir/visual/"
  echo "=== Completed: $model_name ==="
done

echo "==========================================="
echo "Visual inspection completed!"
echo "Check results in: $OUT_ROOT/"
echo "===========================================" 