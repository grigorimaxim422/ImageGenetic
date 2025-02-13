python3 -m validate.offline_vali_trainer --validate_epochs 50  --model_path  "saved_model/efficient01A.pt.best" --learning_rate 0.025

# python3 -m validate.offline_vali_trainer --epochs 50 --model_path  "saved_model/model.pt" --learning_rate 0.025
# python3 -m validate.offline_vali_trainer --epochs 50 --model_path  "saved_model/first.pt" --learning_rate 0.025

##Testing
#python3 -m validate.offline_vali_trainer --validate_epochs 5  --model_path  "saved_model/mobilenetv2.pt" --learning_rate 0.025
#python3 -m validate.offline_vali_trainer --validate_epochs 50  --model_path  "saved_model/efficient01.pt.best" --learning_rate 0.025 #61%, 59%,

#python3 -m validate.offline_vali_trainer --validate_epochs 50  --model_path  "saved_model/nsgac100.pt" --learning_rate 0.025

