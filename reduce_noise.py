import os
from tscn.TSCN import TSCN

PATH = "audio"

tscn_model = TSCN(
    weight_pth='./models/tscn',
    transfer=True,
    device="cuda",
)


def reduce_noise(filename) -> bool:
    try:
        noise_wav_path = os.path.join(PATH, "raw_audio", filename)
        denoise_wav_path = os.path.join(PATH, "denoised_audio", filename)
        tscn_model.inference(noise_wav_path, denoise_wav_path)
        return True
    except:
        return False
