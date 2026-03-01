GPU_IDS=(0 1 2 3)

design_names=(
    # "ariane133"
    # "ariane136"
    # "bp"
    # "bp_fe"
    # "bp_be"
    # "swerv_wrapper"
    "superblue1"
    "superblue3"
    "superblue4"
    "superblue5"
    # "superblue7"
    # "superblue10"
    # "superblue16"
    # "superblue18"
)

num_gpus=${#GPU_IDS[@]}
num_designs=${#design_names[@]}
idx=0

while [ $idx -lt $num_designs ]; do
    batch_end=$(( idx + num_gpus ))
    [ $batch_end -gt $num_designs ] && batch_end=$num_designs
    echo "=== Batch $(( idx / num_gpus + 1 )) (designs $(( idx + 1 ))-$batch_end of $num_designs) ==="
    for (( i=idx; i<batch_end; i++ )); do
        design=${design_names[i]}
        gpu_idx=$(( (i - idx) % num_gpus ))
        gpu_id=${GPU_IDS[$gpu_idx]}
        echo "Starting $design on GPU $gpu_id"
        python -m src.main --benchmark "$design" --seed 3 --config iccad --gpu "$gpu_id" &
    done
    wait
    idx=$batch_end
done
echo "All designs finished."

# python -m src.main --benchmark "superblue1" --seed 3 --config iccad --gpu 0 --debug