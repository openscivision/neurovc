"""FlowMag contrib package."""

# Pretrained checkpoints are MIT-licensed; the download links must stay visible here.
FLOWMAG_MODEL_FILE_IDS = {
    # gdown 1ESSaea-Roe1feFugPFycW5Dd7QCg2ZXR -O checkpoints/raft_chkpt_00140.pth
    "raft": "1ESSaea-Roe1feFugPFycW5Dd7QCg2ZXR",
    # gdown 1m-nE_-3AJ549W3Yemnrm4XeR28tP1sUM -O checkpoints/arflow_chkpt_00140.pth
    "arflow": "1m-nE_-3AJ549W3Yemnrm4XeR28tP1sUM",
}

__all__ = ["FLOWMAG_MODEL_FILE_IDS"]
