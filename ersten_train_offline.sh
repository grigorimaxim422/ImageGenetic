
# python3 -m validate.offline_trainer --epochs 600 --validate_epochs 50 --learning_rate 0.025
#python3 -m validate.ersten_offline_trainer --epochs 1 --validate_epochs 3 --learning_rate 0.01 --net_name ersten
python3 -m validate.ersten_offline_trainer --epochs 1 --validate_epochs 3 --learning_rate 0.01 --net_name mobilenetv2 --model_path saved_model/mobilenetv2.pt

