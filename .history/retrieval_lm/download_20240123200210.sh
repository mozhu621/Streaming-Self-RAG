#!

# cd scratch place
cd data/

# Download zip dataset from Google Drive
filename='OfficeHomeDataset_10072016.zip'
fileid='0B81rNlvomiwed0V1YUxQdC1uOTg'
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip -q ${filename}
rm ${filename}
cd