#!/bin/bash
#SBATCH --job-name=retriever-7B
#SBATCH --output=tongsearch_7B.out
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --partition=h200
cd /local/scratch/kjian54/Diver
source venv/bin/activate
cd Retriever
export HF_HOME=/local/scratch/kjian54/hf_cache

dataset_source="../data/BRIGHT"
# BRIGHT datasets
tasks=(biology earth_science economics psychology robotics stackoverflow sustainable_living pony aops theoremqa_questions theoremqa_theorems leetcode)
# tasks=(Bioinformatics Biology IIYi-Clinical Medical-Sciences MedQA-Diag MedXpertQA-Exam PMC-Clinical PMC-Treatment)
models=(qwen3-4b)  # retriever name infly/inf-retriever-v1
REASONING=TongSearch-QR-7B  # query expansion method diver-qexpand original
BS=-1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
for MODEL in ${models[@]}; do
    for TASK in ${tasks[@]}; do
        echo "Running task: $TASK"
        # # Using the original query achieves 28.9 NDCG@10, as reported in Table 3 of our paper: https://arxiv.org/pdf/2508.07995
        # CUDA_VISIBLE_DEVICES=3 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir ${dataset_source}/cache/cache_${MODEL} --encode_batch_size $BS 

        # Using expanded query achieves 33.9 NDCG@10, as reported in Table 3 of our paper
        CUDA_VISIBLE_DEVICES=0 python run.py --task $TASK --model $MODEL --dataset_source ${dataset_source} --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir ${dataset_source}/cache/cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS 
        
        # # Using the expanded query and rechunk module achieves 37.5 NDCG@10 on average across the 7 general datasets in BRIGHT, as shown in Table 5 of our paper
        # CUDA_VISIBLE_DEVICES=0 python run.py --task $TASK --model $MODEL --output_dir output/${MODEL}_${REASONING}_reasoning --cache_dir ${dataset_source}/cache/cache_${MODEL} --reasoning $REASONING --encode_batch_size $BS --document_expansion rechunk
    done
done    