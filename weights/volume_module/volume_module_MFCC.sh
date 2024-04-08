#!/bin/bash

# URL of the checkpoint file (BERP volume module - MFCC)
url="https://jstorage.box.com/v/BERP-volume-module-mfcc"

# Destination path
dest="assets/sep_volume_module/volume_MFCC.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

