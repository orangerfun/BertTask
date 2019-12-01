#!/usr/bin/env bash
export PYTHONPATH="/home/tmxmall/rd/fangcheng/clstask/BertTask:$PYTHONPATH"
DATA="../data"
MODELDIR="../pre_model/chinese_L-12_H-768_A-12"
OUTPUT="../outputs"
BASEDIR="/home/tmxmall/rd/fangcheng/clstask/BertTask"

cd $BASEDIR
mkdir outputs

cd $BASEDIR/bert_master
python run_classifier.py \
    --data_dir=$DATA \                                   # 数据存放的位置
    --task_name=mypro \                                  # 自定义的任务
    --vocab_file=$MODELDIR/vocab.txt \                   # 下载下来的模型中的词表
    --bert_config_file=$MODELDIR/bert_config.json \      # bert模型配置文件，在下载下来的模型文件夹中
    --output_dir=$OUTPUT \                               # 输出位置
    --do_train=true \
    --do_eval=true \
    --init_checkpoint=$MODELDIR/bert_model.ckpt \         # bert 模型
    --max_sequence_length=200 \
    --train_batch_size=20 \
    --num_train_epoch=100

