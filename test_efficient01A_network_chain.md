1. Install

```
#apt install screen
#screen -S task

git clone https://github.com/grigorimaxim422/ImageGenetic
git checkout chain
cd ImageGenetic
python3 -m venv venv
source venv/bin/activate


pip install -r requirements.txt
```

2.Train

```
./train_offline.sh
```

3. Validate

```
./validate_offline.sh
```
