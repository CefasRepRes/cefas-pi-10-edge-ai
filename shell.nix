# Nix shell script to build a reproducible development
# environment for Python + OpenCV4
#
# <robert.blackwell@cefas.gov.uk>
#
# Packages are pinned to nixos-22.11 stable
#
# References
# https://nixos.wiki/wiki/Python
# https://churchman.nl/2019/01/22/using-nix-to-create-python-virtual-environments/
# https://stackoverflow.com/questions/40667313/how-to-get-opencv-to-work-in-nix

with import (fetchTarball "https://github.com/NixOS/nixpkgs/archive/60c0f762658916a4a5b5a36b3e06486f8301daf4.tar.gz")
  {
    config = {
      allowUnfree = true;
    };
  };
let

  my-python-packages = python-packages: with python-packages; [
    pip
    numpy
    pytorch-bin
    torchvision-bin
    (opencv4.override {enableGtk2 = true;})
    tifffile
    matplotlib
  ];

  python-with-my-packages = python3.withPackages my-python-packages;
in
  pkgs.mkShell {
    buildInputs = [
      python-with-my-packages
      linuxPackages.nvidia_x11
      black
    ];
    shellHook = ''
      export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib
      export PIP_PREFIX="$(pwd)/_build/pip_packages"
      export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.10/site-packages:$PYTHONPATH"
      unset SOURCE_DATE_EPOCH
    '';
  }
