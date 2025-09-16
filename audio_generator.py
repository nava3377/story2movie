# File: audio_generator.py
import numpy as np
from scipy.io import wavfile
from speechbrain.pretrained import Tacotron2, HIFIGAN

def generate_audio(narration_text, spec_gen, vocoder, file_path):
    """Generates a spoken audio file from text using pre-loaded models."""
    print(f"🎙️ Generating audio for: '{narration_text[:40]}...'")
    mel_output, _, _ = spec_gen.encode_text(narration_text)
    waveforms = vocoder.decode_batch(mel_output)
    
    audio_data = (waveforms.squeeze().cpu().numpy() * 32767).astype(np.int16)
    wavfile.write(file_path, 22050, audio_data)
    print(f"✅ Audio saved to: {file_path}")
    return file_path