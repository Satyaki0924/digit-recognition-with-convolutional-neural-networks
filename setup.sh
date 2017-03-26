mkdir graphs points check_points dataset
touch points/accuracy.txt points/loss.txt points/time.txt
sudo apt-get install python-virtualenv
virtualenv -p python3 venv
source venv/bin/activate
sudo pip3 install -r requirements.txt