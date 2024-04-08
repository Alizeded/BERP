#!/bin/bash

# URL of the checkpoint file (URE-Gammatone model)
url="https://jstorage.box.com/v/URE-sep-volume-module-gamma"

# Destination path
dest="assets/sep_volume_module/volume_module_Gammatone.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

