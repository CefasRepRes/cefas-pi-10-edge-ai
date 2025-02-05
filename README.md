#+TITLE: Plankton Imager Edge-AI

* Introduction

This directory contains the Python code for the Plankton Imager Edge AI system, supported by the Turing Institute. It operates on an nvidia jetson by listening to the image stream from a plankton imager "PI10" model. 
The AI model training and validation code is in a separate repository, see https://github.com/alan-turing-institute/ViT-LASNet


Features include

- UDP listener that receives images from the PI10 instrument.

- Two classifiers - a ResNet50 and a ResNet18 that can discriminate
  copepod, non-copepod and detritus.

- Ability to store images in bin files (faster than writing individual
  images).

- Sending of summary data to Azure storage for use by a digital
  dashboard.

- Display of labelled images

- Parsing of GPS data

- Extraction of morphological measurements





* Installation: Linux

** Install Python environment, this has been tested on 3.8.10 (linux) and 3.11.0 (windows)

Linux tested using this version of python:
wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tar.xz


To install required packages. Linux:

#+begin_src bash
  cd ~/git/rapid-plankton/edge-ai
  python3 -m venv env
  source env/bin/activate
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements-jetson.txt
  
  It is likely at this point that you get an error, as the version of jetpack you are using has a specific version of torch in the requirements-jetson.txt, I would suggest we move towards using jetson containers for the correct version https://github.com/dusty-nv/jetson-containers/tree/master
  
  A possibly unrelated error after updating the jetson... you may also have to sudo apt install libopenblas-dev -y
  
#+end_src

If using azure command line interface tools (easy to install on jetson):

#+begin_src bash
  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#+end_src


To activate in Linux:

Subsequent activation (required every time). Linux:

#+begin_src bash
  cd ~/git/rapid-plankton/edge-ai
  source env/bin/activate
#+end_src


Once in (env), on Linux test with:

#+begin_src bash
  ./edge-ai.py --help
  ./classifier.py --folder ../data/examples --output /home/joe/Downloads/classifications.csv -m 2 -b 100
  ./extractor.py --folder ../data/examples --output /home/joe/Downloads/sizes.csv
  ./gps.py --folder ../data/examples --output /home/joe/Downloads/gps.csv
  python 
  import torch
  torch.cuda.is_available()
#+end_src



* Installation: Windows

** Install Python environment, tested on python 3.11.0 for windows

Install python 3.11.0 as tested on my Windows PC using MINGW64 bash on VS:
wget https://www.python.org/ftp/python/3.11.0/python-3.11.0.exe

Install dependencies on Windows:

#+begin_src bash
  cd ~/git/rapid-plankton/edge-ai
  python -m venv env
  source env/scripts/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#+end_src

To activate in Windows:

#+begin_src bash
  cd ~/git/rapid-plankton/edge-ai
  env/scripts/activate
#+end_src

Once in (env), on Windows test on the command line with:

#+begin_src bash
  python edge-ai.py --help
  python classifier.py --folder ../data/examples --output C:/Users/JR13/Downloads/classifications.csv -m 2 -b 100
  python extractor.py --folder ../data/examples --output C:/Users/JR13/Downloads/sizes.csv
  python gps.py --folder ../data/examples --output C:/Users/JR13/Downloads/gps.csv
  python 
  import torch
  torch.cuda.is_available()
#+end_src








* Usage
** Help

Features are switched on or off and configured by means of command
line arguments. For more informations see:

#+begin_src bash
  ./edge-ai.py --help
#+end_src

If you want to see progress, then use =--verbose= to log to the
terminal.

If things are going badly and you need more debug information, then
set =--loglevel DEBUG=.

In general, the more things you switch on, the lower the throughput.

Stopping the app is currently bug - try pressing =q=. If that doesn't
work, type =CTRL-C=.



* Here are some typical use cases

** Fast storage of images

Plug in your external hard drive and change directory to where you
want to store your data. Run this command:

#+begin_src
  ./edge-ai.py --store
#+end_src

Images are stored in =bin= files. these can be split up later into
tiff file using the supplied =split_bin.py= script.

** Local classification

Run the classifier. Use background correction.

#+begin_src
  ./edge-ai.py --classify --verbose  --background-correction
#+end_src

Counts of copepod, non-copepod and detritus per minute are logged to a file.

** Real time reporting to the NOC digital dashboard

Use the ResNet18 algorithm =--gray== developed by NOC and send results
to their digital dashboard.

#+begin_src
  ./edge-ai.py --classify --verbose --send --gray --gps --background-correction
#+end_src



* Running processing of .tif folders in parallel, using 12 cpu cores
** In bash, define a function to run per-core. Done here for individual folders at a time. This seems to achieve close to 100% cpu utilisation on the jetson, so to make it faster, we need to make extractor.py simpler or run on more processors.
#+begin_src
  process_folder() {
      folder="$1"
      bn=$(basename "$folder")
      echo $bn
      python extractor.py --folder "$folder" --output /home/joe/Downloads/deleteme/sizes.csv    
  }
  export -f process_folder
#+end_src
*** Pass the folder names with pipe to be executed with xargs across the 12 cores.
#+begin_src
  find /home/joe/Downloads/deleteme/dups/duplicates/duplicated/ -type d | xargs -P 12 -I {} bash -c 'process_folder "$@"' _ {}
#+end_src






* Running processing of .tif folders in parallel, using 12 cpu cores
** In bash, define a function to run per-core. Done here for individual folders at a time. This seems to achieve close to 100% cpu utilisation on the jetson, so to make it faster, we need to make extractor.py simpler or run on more processors. It took 19 minutes to run 1 million files in a folder called deleteme, nearly 1000/second.
#+begin_src
  apply_size_folder() {
      folder="$1"
      bn=$(basename "$folder")
      echo $bn
      python extractor.py --folder "$folder" --output /home/joe/Downloads/deleteme/sizes.csv    
  }
  export -f apply_size_folder
#+end_src
*** Pass the folder names with pipe to be executed with xargs across the 12 cores.
#+begin_src
  find /home/joe/Downloads/deleteme/ -type d | xargs -P 12 -I {} bash -c 'apply_size_folder "$@"' _ {}
#+end_src



** Running in parallel for gps.py. Again using nearly 100% cpu this took 3 or 4 minutes for 1m files.
#+begin_src
  apply_gps_folder() {
      folder="$1"
      bn=$(basename "$folder")
      echo $bn
      python gps.py --folder "$folder" --output /home/joe/Downloads/deleteme/gps.csv    
  }
  export -f apply_gps_folder
#+end_src
*** Pass the folder names with pipe to be executed with xargs across the 12 cores.
#+begin_src
  find /home/joe/Downloads/deleteme/ -type d | xargs -P 12 -I {} bash -c 'apply_gps_folder "$@"' _ {}
#+end_src




** Running in parallel for classify.py. This runs on the GPU but still seems to use 100% of the avilable cpu. Batch size is important here, I have set to 500 as the jetson crashed when I chose 1000. 100k files took less than 40 mins, possibly less.

classify.py should be used differently to get the most out of it.
Within a single folder, classify.py could be applied like so:
#+begin_src
/classifier.py --folder ../data/examples --output /home/joe/Downloads/classifications.csv -m 1 -b 100
#+end_src

You might get a speed of ~ 500 images per second if batching several thousand images at once.

If we parallelise across subfolders using xargs, this output seems to drop dramatically because loading the process is limiting the speed. Running 1000 images in 10 directories over 12 cores, it takes 10x as long!

Therefore in addition to using batching (-b) I have added a command line option of -r 1 to batch multiple folders of images together across subfolders recursively before passing to the GPU. I am not sure what the optimum batch size is but something around 100 seems fine, 1000 not always fine.
#+begin_src
./classifier.py --folder /home/joe/Downloads/deleteme/dups/duplicates/ --output /home/joe/Downloads/classifications.csv -m 1 -b 100 -r 1
#+end_src



* Notes on NVIDIA Jetson
Our target hardware is NVIDIA Jetson AGX Orin.
** Please be aware
Before reinstalling the OS in the case of a jetson that seemingly won't boot, firstly aways question whether the DP-HDMI adapter is working. Plugging in directly with DP and NO ADAPTER may be the fix you need if it seems that the jetson is bricked.
It is normal to get an unresponsive green nvidia symbol when SSH'ing in with X or VNC if it is not already plugged into a screen! This also does not mean it is bricked.
Lastly in this process, do not use cefas guest wifi for anything, the various firmware downloads will constantly fail, instead use a hotspot off your laptop after plugging it in to the building ethernet.

You flash using sdkmanager on a host linux machine, the one supplied by IT that is labelled "plankton project", via a USB-c plugged into the otg port (the lone usb c port beside the 2 regular USB ports, i.e. on the other side to the ethernet port and all the other ports).

sdkmanager can guide you through the right version of jetpack to install for our 32gb jetson orion developer kits. First you need to put the jetson into recovery mode by pressing the left and middle buttons (guidance on how to do this online varies, this seems to be the button combination for our jetsons).

After flashing the OS, reboot the jetson and set it up before installing any more of the packages suggested by sdkmanager. Again plug in with displayport not hdmi adapter and you should now have an OS with GUI.

At this point, you can follow the section titled "Installation: Linux" to install the rapid plankton repository as standard on linux (using requirements-jetson).

Jetpack is a pre-assembled package of packages. It should allow you to install an up-to-date version of cuda on your machine, allowing you to have up-to-date version of torch and torchvision libraries with CUDA's GPU acceleration.
Rob Blackwell did not use sdkmanager and jetpack to set up the Jetson. Unfortunately his installation was built around a version of torchvision that was taken down, so I was forced to find a way to reinstall on updated torchvision and therefore CUDA libraries. For his process, notes are available in the file jetson.org. 



** To auto run the jetson
I recommend you search and select "startup applications preferences" and add the following commands:
#+begin_src
	gnome-terminal -- sh git/rapid-plankton/edge-ai/startup1.sh
	teamviewer
#+end_src
By doing this, the jetson will be available for communication with teamviewer remote access directly, which may be preferable to mirroring the PI pc and then ssh'ing across the local network. One of the quirks of the jetson is that it must be plugged into a screen when booted if this is to work, else the GUI will not be present. It does not rule out the option of sshing across. 




