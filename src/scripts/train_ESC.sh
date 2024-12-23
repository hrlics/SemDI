python src/train.py \
    --dataset src/dataset/ESC/ESC_dataset \
    --dataset_name ESC \
    --num_folds 5 \
    --num_epochs 100 \
    --train_batchsize 20 \
    --test_batchsize 20 \
    --learning_rate 1e-5 \
    --bert_path FacebookAI/roberta-large \
    --d_model 1024 \
    --num_heads 16 \
    --dropout_rate 0.5 \
    --SEED 3407
