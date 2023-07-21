# UNet
UNet Archicture for images Segmentation on medical images(Mitocondria)


#Dataset 
1. "Download Dataset of Download training sub-volume"
2. "Download groundtruth training sub-volum"
   To download dataset use link  -- https://www.epfl.ch/labs/cvlab/data/data-em/

Prerequisites:
Pip version 23.2
Python version 3.10.6

To Install Requirmemnts file 
cd /UNet-test
pip install -r requirments.txt

#Create virtual environment
python3 -m venv .venv
source .venv/bin/activate


#Run over docker
sudo apt install docker.io

#Build docker image
docker build --tag python-unet-docker .

#Run Docker image
docker tag python-unet-docker:latest python-unet-docker:v1.0.0

#Diagnosis:
Check docker images: docker images
Check containers list: docker container ps
