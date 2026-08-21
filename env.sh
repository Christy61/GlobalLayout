pip install -r requirements.txt

# Genesis v0.2.1. Pin the commit so the rendering API stays reproducible.
GENESIS_COMMIT="1b2b7154fbaf068dbd81a197f3c27aff31fc2ee1"
GENESIS_DIR="${GENESIS_DIR:-./Genesis}"
if [ ! -d "${GENESIS_DIR}/.git" ]; then
    git clone https://github.com/Genesis-Embodied-AI/Genesis.git "${GENESIS_DIR}"
fi
git -C "${GENESIS_DIR}" fetch origin "${GENESIS_COMMIT}"
git -C "${GENESIS_DIR}" checkout --detach "${GENESIS_COMMIT}"
git -C "${GENESIS_DIR}" submodule update --init --recursive
pip install -e "${GENESIS_DIR}"

pip install -U flash-attn --no-build-isolation