# File: main.py
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from narrative_parser import decompose_story_into_scenes
from visuals_generator import generate_start_image, generate_video_from_image
from audio_generator import generate_audio
from video_assembler import combine_video_and_audio, concatenate_clips

def main():
    # --- The Story ---
    story = """
    The old astronomer adjusted the brass telescope, its lens pointed towards the shimmering nebula. 
    For years, he had watched this celestial cloud, but tonight was different. 
    A tiny, pulsating light, a newborn star, blinked into existence at the heart of the cosmic dust. 
    Tears welled in his eyes as he recorded the discovery in his journal, a new chapter in the universe's story.
    """
    
    # --- Step 1: Load All Models into Memory (once) ---
    print("--- Loading all AI models into memory. This is the main setup time. ---")
    llm_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", load_in_4bit=True, device_map="auto")
    image_pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
    video_pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
    video_pipe.enable_model_cpu_offload() # Good practice even with swapping
    spectrogram_generator = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tacotron")
    vocoder = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_hifigan")
    print("\n✅ All models loaded and ready!")

    # --- Step 2: Decompose Story into Scenes ---
    story_plan = decompose_story_into_scenes(story, llm_model, tokenizer)
    if not story_plan:
        print("Could not create a story plan. Exiting.")
        return

    # --- Step 3: Process Each Scene ---
    final_clip_paths = []
    # Create an 'output' directory to store the clips
    if not os.path.exists("output"):
        os.makedirs("output")

    for i, scene in enumerate(story_plan):
        scene_num = i + 1
        print(f"\n--- Processing Scene {scene_num}/{len(story_plan)} ---")
        
        # A. Generate the visual start frame
        start_image = generate_start_image(scene["scene_description"], image_pipe, output_path=f"output/scene_{scene_num}_start.png")
        
        # B. Animate the frame (with memory swapping for stability)
        print("Swapping models to free VRAM for video generation...")
        image_pipe.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        
        silent_video = generate_video_from_image(start_image, video_pipe, output_path=f"output/scene_{scene_num}_silent.mp4")
        
        print("Swapping models back for next scene...")
        image_pipe.to('cuda')

        # C. Generate the audio narration
        audio_file = generate_audio(scene["narration_text"], spectrogram_generator, vocoder, file_path=f"output/scene_{scene_num}_audio.wav")
        
        # D. Combine the video and audio for the scene
        final_clip = combine_video_and_audio(silent_video, audio_file, output_path=f"output/scene_{scene_num}_final.mp4")
        final_clip_paths.append(final_clip)

    # --- Step 4: Assemble the Final Film ---
    if final_clip_paths:
        concatenate_clips(final_clip_paths, "Final_Movie.mp4")
        print("\n🎉 Story2Cinema process complete! 🎉")
        print("You can find your final film in the main project folder as 'Final_Movie.mp4'")
    else:
        print("No clips were generated.")

if __name__ == "__main__":
    main()