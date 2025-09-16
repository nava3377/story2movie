# File: video_assembler.py
import subprocess
import os

def combine_video_and_audio(video_path, audio_path, output_path):
    """Merges a silent video clip with an audio file using FFmpeg."""
    print(f"🎞️ Combining {os.path.basename(video_path)} and {os.path.basename(audio_path)}...")
    command = [
        'ffmpeg', '-y', '-i', video_path, '-i', audio_path,
        '-c:v', 'copy', '-c:a', 'aac', '-shortest', output_path
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output_path

def concatenate_clips(clip_paths, output_path):
    """Stitches multiple video clips together into a single film using FFmpeg."""
    print(f"🎥 Assembling {len(clip_paths)} clips into the final film...")
    with open("clips_list.txt", "w") as f:
        for clip in clip_paths:
            f.write(f"file '{clip}'\n")
            
    command = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', 'clips_list.txt', '-c', 'copy', output_path
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    os.remove("clips_list.txt") # Clean up the temporary file
    print(f"✅ Full film saved to: {output_path}")
    return output_path