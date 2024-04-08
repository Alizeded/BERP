#!/bin/bash

# URL of the checkpoint file (URE-Gammatone model)
url="https://jstorage.box.com/v/URE-occupancy-module-mel"

# Destination path
dest="assets/occupancy_module/occupancy_module_Mel.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

