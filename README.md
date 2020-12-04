# Detection and Analysis of paintings and people in an art gallery
Project for the University course "Vision and Cognitive Systems" (Computer Engineering UniMoRe), implementing a pipeline for detection and analysis of paintings and people in videos taken in an art gallery.

Paper with description of the pipeline is available in this folder (report.pdf)

# Installation
The code is written in Python3 using PyTorch, tested on Ubuntu 18.04 system.

## Install system dependencies
Install dependencies for using python/pip/virtualenv:

```
$ sudo apt-get install build-essential curl libssl-dev libffi-dev python-dev
$ sudo apt install python3-pip
$ sudo apt install -y python3-venv
```

Create a new virtual environment and activate it:

```
$ python3 -m venv env
$ source env/bin/activate
```

Install all the required pip packages inside the virtual environment:

```
$ pip3 install -r requirements.txt --no-cache-dir
```

## Download software dependencies
You now need to download the cfg folder for the YOLO neural network (approx. 430 MB including pre-trained models)
and data folder required by the pipeline including painting database, and map files (approx. 60 MB)

You can download and extract all in the code directory simply using the provided script

```
$ chmod +x download_dependencies.sh
$ ./download_dependencies.sh
```

or manually download the [cfg](https://drive.google.com/file/d/1AetqzDa30b8GqApzl8g24LFAEaNxTQuT/view) archive, the [data](https://drive.google.com/file/d/1UvJbTCHc2qx-33AM9QEGxcFkphDP0Cee/view) archive and extract them in the code directory

## (Optional) Download videos for testing
You can use the following script to download a small video subset for testing (approx 350 MB):

```
$ chmod +x download_smallvideos.sh
$ ./download_smallvideos.sh
```

Now you should be ready!

# Usage
Inside your virtual environment simply launch the pipeline by running:

```
$ python3 main.py --video path/to/videofile
```

You can also skip frames between each iteration using the --skip-frames option:

```
$ python3 main.py --video path/to/videofile --skip-frames N
```

