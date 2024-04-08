#!/bin/bash

# URL of the checkpoint file (BERP occupancy module - MFCC)
url="https://jstorage.box.com/v/BERP-occupancy-module-mfcc"

# Destination path
dest="assets/occupancy_module/occupancy_MFCC.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

