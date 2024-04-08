#!/bin/bash

# URL of the checkpoint file (BERP unified model - melspectrogram)
url="https://jstorage.box.com/v/BERP-unified-module-mel"

# Destination path
dest="weights/unified_module/unified_mel.ckpt"

# Use wget to download the file
wget "$url" -P "$dest"

# Check if the download was successful
if [ $? -eq 0 ]; then
  echo "Download successful"
else
  echo "Download failed"
fi
```

