#!/bin/bash

# Replace this with your OneDrive shared folder link
SHARE_URL="https://drive.google.com/drive/folders/1572kjDuKqx0BxawbrQ_VFDFI1lIdYymo?usp=sharing"
#!/bin/bash
gdown --folder "$SHARE_URL"

gdown --folder "https://drive.google.com/drive/folders/1572kjDuKqx0BxawbrQ_VFDFI1lIdYymo?usp=sharing"

# Step 1: Extract the unique token from the URL
ENCODED_URL=$(python3 -c "import urllib.parse; print(urllib.parse.quote('''$SHARE_URL'''))")

API_URL="https://api.onedrive.com/v1.0/shares/u!${ENCODED_URL}/root/content"

# Step 3: Download the shared folder or file
wget -O onedrive_download.zip "$API_URL"

echo "✅ Download completed: onedrive_download.zip"
echo