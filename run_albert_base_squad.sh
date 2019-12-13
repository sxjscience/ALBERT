set -x

MODEL_DIR=albert_base_v2
SQUAD_DIR=squad_data
OUT_DIR=${MODEL_DIR}_squad_1.1_finetune

python3 run_squad_sp.py \
    --do_train \
    --do_predict \
    --albert_config_file ${MODEL_DIR}/assets/albert_config.json \
    --vocab_file ${MODEL_DIR}/assets/30k-clean.vocab \
    --spm_model_file ${MODEL_DIR}/assets/30k-clean.model \
    --output_dir ${OUT_DIR} \
    --init_checkpoint ${MODEL_DIR}/variables/variables \
    --train_file ${SQUAD_DIR}/train-v1.1.json \
    --predict_file ${SQUAD_DIR}/dev-v1.1.json \
    --train_feature_file ${OUT_DIR}/train_feature_file.fea
