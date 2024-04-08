#!/bin/bash

# URL of the checkpoint file (BERP distance module - MFCC)
url="https://jstorage.box.com/v/BERP-distance-module-mfcc"

# Destination path
dest="weights/distance_module/distance_MFCC.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

