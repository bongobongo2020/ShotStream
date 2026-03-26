CUDA_VISIBLE_DEVICES=0 \
python Inference_Causal.py \
    --config_path ckpts/shotstream.yaml \
    --output_folder demo/infer/ \
    --resume_ckpt ckpts/shotstream_merged.pt \
    --multi_caption True \
    --data_path demo/testdata/testset.csv