# Cameroon License Plate Recognition Project  

## Description of the project  

Read the .pdf Report - Cameroon License Plate Recognition.pdf  



## Installation guide  

1. Install python3.7 (if it is not installed ) :
   * Activate internet connection
   * open the command-line (Ctrl-Alt- T)
   * enter  sudo apt update
   * enter  sudo apt install software-properties-common
   * enter  sudo add-apt-repository ppa:deadsnakes/ppa
   * enter  sudo apt update
   * enter  sudo apt install python3.7
   * close the command-line
2. Download **model_final.pth** from https://drive.google.com/file/d/11-eQ3fbYC2DtFFknSPPXGA4Dn6sA-Eb8/view?usp=sharing and put it in the repository
3. Install the dependancies :  
   * open the command-line (Ctrl-Alt-T)  
   * enter pip3 install -r requirements.txt  
   * Now we need to install detectron2  
   * enter pip3 install -U torch==1.4+cu100 torchvision==0.5+cu100 -f https://download.pytorch.org/whl/torch_stable.html  
   * enter pip3 install cython pyyaml==5.1  
   * enter pip3 install -U  'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' 
   * enter pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html  

4. Run the program:  
   * Take a photo or download a photo or choose a photo in the directory test_images (there are some
     images of cars)  
   * enter python predict_plate.py "/path/to/image". This means that if the image car.jpg is in the
     directory .images, we should write python predict_plate.py "./images/car.jpg  
   * It will output the prediction of the license plate  

