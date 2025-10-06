

# Weibo,Weibo21,Pheme
python3 main.py --model MMLNet --weight_decay 0.005 --train_batch_size 16 --dev_batch_size 16 --learning_rate 1e-4 --clip_learning_rate 3e-6 --num_train_epochs 20 --layers 5 --max_grad_norm 6 --dropout_rate 0.3 --optimizer_name adam --text_size 768 --image_size 1024 --warmup_proportion 0.2 --device 0




