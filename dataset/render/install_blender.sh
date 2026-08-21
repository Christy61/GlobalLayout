wget https://ftp.halifax.rwth-aachen.de/blender/release/Blender4.4/blender-4.4.3-linux-x64.tar.xz
tar -xf blender-4.4.3-linux-x64.tar.xz
rm blender-4.4.3-linux-x64.tar.xz
chmod +x blender-4.4.3-linux-x64/blender
blender-4.4.3-linux-x64/4.4/python/bin/python3.11 -m ensurepip
blender-4.4.3-linux-x64/4.4/python/bin/python3.11 -m pip install PyYAML
blender-4.4.3-linux-x64/4.4/python/bin/python3.11 -m pip install tqdm