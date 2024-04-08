#!/bin/bash

# URL of the checkpoint file (BERP unified model - gammatonegram)
url="https://jstorage.box.com/v/BERP-unified-module-gamma"

# Destination path
dest="assets/unified_module/unified_gammatone.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

