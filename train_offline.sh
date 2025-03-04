
python3 -m validate.offline_trainer --epochs 2 --validate_epochs 5 --learning_rate 0.025 --net_name squeezenet --model_path "saved_model/squeezenet-origin.pt"
#python3 -m validate.offline_trainer --epochs 5 --validate_epochs 5 --learning_rate 0.025 --net_name efficient01 --model_path "saved_model/efficient01-for-small.pt"
python3 -m validate.offline_trainer --epochs 200 --validate_epochs 50 --learning_rate 0.025 --net_name efficient01A --model_path "saved_model/efficient01A.pt"
#python3 -m validate.offline_trainer --epochs 1 --validate_epochs 3 --learning_rate 0.01

python3 -m validate.offline_trainer --epochs 200 --validate_epochs 50 --learning_rate 0.025 --net_name efficient01A --model_path "saved_model/efficient01A-prune.pt"