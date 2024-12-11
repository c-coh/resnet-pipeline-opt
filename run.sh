#!/bin/bash

# define the files to test
files=("baseline_nopipe.py" "baseline_lighting.py" "pipeline.py")

#define batch sizes and epochs to test
gpus=4
batch_sizes=(4 16 32)
epochs=(1)

#output file for storing timings
output_file="execution_times.txt"
> "$output_file" #clear the output file if it exists

#iterate over each file
for file in "${files[@]}"; do
    echo "Testing $file" >> "$output_file"

    #iterate over each batch size
    for batch_size in "${batch_sizes[@]}"; do

        #iterate over each epoch
        for epoch in "${epochs[@]}"; do
            echo "Testing $file with gpus: $gpus batch size: $batch_size epochs: $epoch"

            #command to execute
            cmd="mpiexec -n $gpus $file -b $batch_size -e $epoch"

            #measure time and log the result
            {
                echo "GPUs: $gpus   Batch Size: $batch_size  Epochs: $epoch"
                /usr/bin/time -f "Real: %e seconds\nUser: %U seconds\nSys: %S seconds" $cmd
            } >> "$output_file" 2>&1

            echo "-----------------------------------" >> "$output_file"
        done
    done
done

echo "Timings saved to $output_file"
