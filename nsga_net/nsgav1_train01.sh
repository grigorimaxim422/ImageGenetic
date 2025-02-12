# # architecture found from macro search space
#python validation/train.py --net_type macro --cutout --batch_size 128 --epochs 350 
# python3 -m validation.trainc100 --net_type macro --cutout --batch_size 128 --epochs 350 
# python3 -m validation.trainc100 --net_type macro --cutout --batch_size 128 --epochs 350 --check_only

# architecture found from micro search space
# python3 -m validation.train --net_type micro --arch NSGANet --layers 20 --init_channels 34 --filter_increment 4  --cutout --auxiliary --batch_size 96 --droprate 0.2 --SE --epochs 5
# python3 -m validation.trainc100 --net_type micro --arch NSGANet --layers 3 --init_channels 32 --filter_increment 4  --cutout --auxiliary --batch_size 64 --droprate 0.01 --SE --epochs 1 --model_name weights-micro-done-500k-39M-2MB.pt #--check_only

# python3 -m validation.trainc100  --net_type macro --cutout --batch_size 64 --epochs 5 

python3 -m validation.trainc100 --net_type micro --arch NSGANet --layers 3 --init_channels 32 --filter_increment 4  --cutout --auxiliary --batch_size 64 --droprate 0.0 --SE --epochs 1 --model_name weights-nano-done-500k-39M-2MB.pt #--check_only