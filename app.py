# %cd /content/Qwen3-TTS-Colab
from subtitle import subtitle_maker
from process_text import text_chunk
from qwen_tts import Qwen3TTSModel
import subprocess
import os
import json
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import split_on_silence
from huggingface_hub import snapshot_download
from hf_downloader import download_model
import gc 
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
  HF_TOKEN=None

# Global model holders
loaded_models = {}
MODEL_SIZES = ["0.6B", "1.7B"]

# Speaker and language choices
SPEAKERS = [
    "Aiden", "Dylan", "Eric", "Ono_anna", "Ryan", "Serena", "Sohee", "Uncle_fu", "Vivian"
]
LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian"]

# --- Helper Functions ---

def get_model_path(model_type: str, model_size: str) -> str:
    """Get model path based on type and size."""
    try:
      return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}")
    except Exception as e:
      return download_model(f"Qwen/Qwen3-TTS-12Hz-{model_size}-{model_type}", download_folder="./qwen_tts_model", redownload= False)

def clear_other_models(keep_key=None):
    """Delete all loaded models except the current one."""
    global loaded_models
    keys_to_delete = [k for k in loaded_models if k != keep_key]
    for k in keys_to_delete:
        try:
            del loaded_models[k]
        except Exception:
            pass
    for k in keys_to_delete:
        loaded_models.pop(k, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model(model_type: str, model_size: str, allow_fallback: bool = True):
    """Load model and clear others to avoid OOM in Colab."""
    global loaded_models
    key = (model_type, model_size)
    if key in loaded_models:
        print(f"✅ Using cached model: {model_type} {model_size}")
        return loaded_models[key]
    
    clear_other_models(keep_key=key)
    model_path = get_model_path(model_type, model_size)
    use_cuda = torch.cuda.is_available()
    device_map = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    if not use_cuda:
        print("⚠️ CUDA is not available. Falling back to CPU mode (this will be much slower).")

    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
        )
        loaded_models[key] = model
        print(f"✅ Using model: {model_type} {model_size}")
        return model
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                f"Failed to load {model_type} {model_size} and fallback is disabled. {e}"
            )

        # If 1.7B cannot be loaded (OOM or runtime limits), fall back to 0.6B.
        if model_size != "1.7B":
            raise

        print(f"⚠️ Failed to load {model_type} 1.7B ({e}). Falling back to 0.6B.")
        fallback_key = (model_type, "0.6B")
        if fallback_key in loaded_models:
            loaded_models[key] = loaded_models[fallback_key]
            print(f"✅ Using cached model: {model_type} 0.6B")
            return loaded_models[fallback_key]

        clear_other_models(keep_key=fallback_key)
        fallback_model_path = get_model_path(model_type, "0.6B")
        fallback_model = Qwen3TTSModel.from_pretrained(
            fallback_model_path,
            device_map=device_map,
            dtype=dtype,
        )
        loaded_models[fallback_key] = fallback_model
        loaded_models[key] = fallback_model
        print(f"✅ Using model: {model_type} 0.6B")
        return fallback_model

def _normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip:
        y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y

def _audio_to_tuple(audio):
    """Convert Gradio audio input to (wav, sr) tuple."""
    if audio is None: return None
    if isinstance(audio, str):
        try:
            wav, sr = sf.read(audio)
            wav = _normalize_audio(wav)
            return wav, int(sr)
        except Exception as e:
            print(f"Error reading audio file: {e}")
            return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = _normalize_audio(wav)
        return wav, int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = _normalize_audio(audio["data"])
        return wav, sr
    return None

def transcribe_reference(audio_path, mode_input, language="English"):
    """Uses subtitle_maker to extract text from the reference audio."""
    should_run = False
    if isinstance(mode_input, bool): should_run = mode_input
    elif isinstance(mode_input, str) and "High-Quality" in mode_input: should_run = True

    if not audio_path or not should_run: return None
    
    print(f"Starting transcription for: {audio_path}")
    src_lang = language if language != "Auto" else "English"
    try:
        results = subtitle_maker(audio_path, src_lang)
        transcript = results[7]
        return transcript if transcript else "Could not detect speech."
    except Exception as e:
        print(f"Transcription Error: {e}")
        return f"Error during transcription: {str(e)}"

# --- Audio Processing Utils (Disk Based) ---

def remove_silence_function(file_path, minimum_silence=100):
    """Removes silence from an audio file using Pydub."""
    try:
        base, ext = os.path.splitext(file_path)
        ext = ext.lower().lstrip(".")
        out_ext = ext if ext in ("wav", "mp3") else "wav"
        output_path = f"{base}_no_silence.{out_ext}"

        sound = AudioSegment.from_file(file_path)
        audio_chunks = split_on_silence(sound,
                                        min_silence_len=minimum_silence,
                                        silence_thresh=-45,
                                        keep_silence=50)
        combined = AudioSegment.empty()
        for chunk in audio_chunks:
            combined += chunk
        combined.export(output_path, format=out_ext)
        return output_path
    except Exception as e:
        print(f"Error removing silence: {e}")
        return file_path

def process_audio_output(audio_path, make_subtitle, remove_silence, language="Auto"):
    """Handles Silence Removal and Subtitle Generation."""
    # 1. Remove Silence
    final_audio_path = audio_path
    if remove_silence:
        final_audio_path = remove_silence_function(audio_path)
    
    # 2. Generate Subtitles
    default_srt, custom_srt, word_srt, shorts_srt = None, None, None, None
    if make_subtitle:
        try:
            results = subtitle_maker(final_audio_path, language)
            default_srt = results[0]
            custom_srt = results[1]
            word_srt = results[2]
            shorts_srt = results[3]
        except Exception as e:
            print(f"Subtitle generation error: {e}")

    return final_audio_path, default_srt, custom_srt, word_srt, shorts_srt

def stitch_chunk_files(chunk_files,output_filename):
    """
    Takes a list of file paths.
    Stitches them into one file.
    Deletes the temporary chunk files.
    """
    if not chunk_files:
        return None

    combined_audio = AudioSegment.empty()
    
    print(f"Stitching {len(chunk_files)} audio files...")
    for f in chunk_files:
        try:
            segment = AudioSegment.from_wav(f)
            combined_audio += segment
        except Exception as e:
            print(f"Error appending chunk {f}: {e}")

    out_ext = os.path.splitext(output_filename)[1].lower().lstrip(".")
    out_format = out_ext if out_ext in ("wav", "mp3") else "wav"
    combined_audio.export(output_filename, format=out_format)
    
    # Clean up temp files
    for f in chunk_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            print(f"Warning: Could not delete temp file {f}: {e}")
            
    return output_filename

# --- Generators (Memory Optimized) ---

def generate_voice_design(text, language, voice_description, remove_silence, make_subs):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None
    
    try:
        # 1. Chunk Text
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        print(f"Processing {len(text_chunks)} chunks...")
        
        chunk_files = []
        # Enforce true VoiceDesign 1.7B usage (no silent 0.6B fallback).
        tts = get_model("VoiceDesign", "1.7B", allow_fallback=False)

        # 2. Generate & Save Loop
        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_voice_design(
                text=chunk.strip(),
                language=language,
                instruct=voice_description.strip(),
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            
            # Save immediately to disk
            temp_filename = f"temp_chunk_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            
            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
        
        # 3. Stitch from disk
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        
        # 4. Post-Process
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        
        return final_audio, "Generation Success!", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None

def generate_custom_voice(text, language, speaker, instruct, model_size, remove_silence, make_subs):
    if not text or not text.strip(): return None, "Error: Text is required.", None, None, None, None
    
    try:
        text_chunks, tts_filename = text_chunk(text, language, char_limit=280)
        chunk_files = []
        tts = get_model("CustomVoice", model_size)
        formatted_speaker = speaker.lower().replace(" ", "_")

        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_custom_voice(
                text=chunk.strip(),
                language=language,
                speaker=formatted_speaker,
                instruct=instruct.strip() if instruct else None,
                non_streaming_mode=True,
                max_new_tokens=2048,
            )
            # Save immediately
            temp_filename = f"temp_custom_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)
            
            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()
            
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, "Generation Success!", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None


def _safe_name(value: str, fallback: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in (value or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def _generate_custom_voice_file(text, language, speaker, instruct, model_size, output_filename):
    """Generate one TTS file and save it directly to output_filename."""
    text_chunks, _ = text_chunk(text, language, char_limit=280)
    chunk_files = []
    tts = get_model("CustomVoice", model_size)
    formatted_speaker = speaker.lower().replace(" ", "_")

    for i, chunk in enumerate(text_chunks):
        wavs, sr = tts.generate_custom_voice(
            text=chunk.strip(),
            language=language,
            speaker=formatted_speaker,
            instruct=instruct.strip() if instruct else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        temp_filename = f"temp_batch_{i}_{os.getpid()}.wav"
        sf.write(temp_filename, wavs[0], sr)
        chunk_files.append(temp_filename)

        del wavs
        torch.cuda.empty_cache()
        gc.collect()

    return stitch_chunk_files(chunk_files, output_filename)


def _generate_voice_design_file(text, language, voice_description, output_filename):
    """Generate one VoiceDesign TTS file and save it directly to output_filename."""
    text_chunks, _ = text_chunk(text, language, char_limit=280)
    chunk_files = []
    tts = get_model("VoiceDesign", "1.7B", allow_fallback=False)

    for i, chunk in enumerate(text_chunks):
        wavs, sr = tts.generate_voice_design(
            text=chunk.strip(),
            language=language,
            instruct=voice_description.strip(),
            non_streaming_mode=True,
            max_new_tokens=2048,
        )
        temp_filename = f"temp_vd_batch_{i}_{os.getpid()}.wav"
        sf.write(temp_filename, wavs[0], sr)
        chunk_files.append(temp_filename)

        del wavs
        torch.cuda.empty_cache()
        gc.collect()

    return stitch_chunk_files(chunk_files, output_filename)


def _generate_voice_clone_file(
    text,
    language,
    ref_audio,
    ref_text,
    clone_mode,
    model_size,
    output_filename,
):
    """Generate one VoiceClone TTS file and save it directly to output_filename."""
    if not ref_audio or not str(ref_audio).strip():
        raise ValueError("ref_audio is required for voice_clone mode.")

    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Reference audio not found: {ref_audio}")

    mode = (clone_mode or "fast").strip().lower()
    if mode not in ("fast", "high_quality"):
        raise ValueError("clone_mode must be either 'fast' or 'high_quality'.")

    use_xvector_only = mode == "fast"
    final_ref_text = ref_text
    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        raise ValueError(f"Could not load reference audio: {ref_audio}")

    if not use_xvector_only and (not final_ref_text or not str(final_ref_text).strip()):
        print("Auto-transcribing reference for high_quality cloning...")
        final_ref_text = transcribe_reference(ref_audio, True, language)
        if not final_ref_text or "Error" in final_ref_text:
            raise RuntimeError(f"Transcription failed: {final_ref_text}")

    text_chunks, _ = text_chunk(text, language, char_limit=280)
    chunk_files = []
    tts = get_model("Base", model_size)

    for i, chunk in enumerate(text_chunks):
        wavs, sr = tts.generate_voice_clone(
            text=chunk.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=final_ref_text.strip() if final_ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        temp_filename = f"temp_clone_batch_{i}_{os.getpid()}.wav"
        sf.write(temp_filename, wavs[0], sr)
        chunk_files.append(temp_filename)

        del wavs
        torch.cuda.empty_cache()
        gc.collect()

    return stitch_chunk_files(chunk_files, output_filename)


def generate_from_json(
    audio_json_path,
    output_dir="./generated_audio_json",
    model_type="voice_design",
    default_language="Portuguese",
    default_speaker="",
    default_instruct="Fale em portugues com um tom divertido, brincalhao e bem humorado.",
    clone_ref_audio="",
    clone_ref_text="",
    clone_mode="fast",
    model_size="1.7B",
    output_format="mp3",
    remove_silence=False,
    make_subs=False,
):
    """
    Batch-generate TTS audio from a JSON file.

    Supported item formats:
    - "Some text"
    - {"text": "Some text", "filename": "name.mp3", "language": "Portuguese", "instruct": "..."}
    - For custom voice mode: include "speaker" (or use --speaker)
    - For voice clone mode: include "ref_audio", optionally "ref_text" and "clone_mode"
    """
    if not os.path.exists(audio_json_path):
        raise FileNotFoundError(f"JSON file not found: {audio_json_path}")

    with open(audio_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("audio.json must contain a JSON array.")

    output_format = (output_format or "mp3").lower().strip()
    if output_format not in ("wav", "mp3"):
        raise ValueError("output_format must be either 'wav' or 'mp3'.")

    model_type = (model_type or "voice_design").strip().lower()
    if model_type not in ("voice_design", "custom_voice", "voice_clone"):
        raise ValueError("model_type must be one of: 'voice_design', 'custom_voice', 'voice_clone'.")

    if model_type == "voice_design":
        model_size = "1.7B"
    elif model_type == "custom_voice" and (not default_speaker or not str(default_speaker).strip()):
        raise ValueError("default_speaker is required when model_type='custom_voice'.")

    if not torch.cuda.is_available() and model_size == "1.7B":
        print("⚠️ CPU runtime detected. Switching model size from 1.7B to 0.6B for better stability.")
        model_size = "0.6B"

    os.makedirs(output_dir, exist_ok=True)
    generated_files = []

    for idx, item in enumerate(payload, start=1):
        if isinstance(item, str):
            text = item.strip()
            language = default_language
            instruct = default_instruct
            filename = f"audio_{idx:03d}.{output_format}"
            ref_audio = clone_ref_audio
            ref_text = clone_ref_text
            item_clone_mode = clone_mode
        elif isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            language = item.get("language", default_language)
            instruct = item.get("voice_description", item.get("instruct", default_instruct))
            filename = item.get("filename") or f"audio_{idx:03d}.{output_format}"
            ref_audio = item.get("ref_audio", clone_ref_audio)
            ref_text = item.get("ref_text", clone_ref_text)
            item_clone_mode = item.get("clone_mode", clone_mode)
        else:
            print(f"[skip] Item {idx}: unsupported type {type(item).__name__}")
            continue

        if not text:
            print(f"[skip] Item {idx}: empty text")
            continue

        filename = str(filename)
        base_name, ext = os.path.splitext(filename)
        ext = ext.lower().lstrip(".")
        if ext not in ("wav", "mp3"):
            base_name = filename
            ext = output_format

        safe_base = _safe_name(base_name, f"audio_{idx:03d}")
        out_path = os.path.join(output_dir, f"{safe_base}.{ext}")

        print(f"[{idx}/{len(payload)}] Generating: {out_path}")
        if model_type == "voice_design":
            stitched_file = _generate_voice_design_file(
                text=text,
                language=language,
                voice_description=instruct,
                output_filename=out_path,
            )
        elif model_type == "custom_voice":
            speaker = item.get("speaker", default_speaker) if isinstance(item, dict) else default_speaker
            stitched_file = _generate_custom_voice_file(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
                model_size=model_size,
                output_filename=out_path,
            )
        else:
            stitched_file = _generate_voice_clone_file(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                clone_mode=item_clone_mode,
                model_size=model_size,
                output_filename=out_path,
            )

        final_audio, _, _, _, _ = process_audio_output(
            stitched_file,
            make_subtitle=make_subs,
            remove_silence=remove_silence,
            language=language,
        )
        generated_files.append(final_audio)

    return generated_files

def smart_generate_clone(ref_audio, ref_text, target_text, language, mode, model_size, remove_silence, make_subs):
    if not target_text or not target_text.strip(): return None, "Error: Target text is required.", None, None, None, None
    if not ref_audio: return None, "Error: Ref audio required.", None, None, None, None

    # 1. Mode & Transcript Logic
    use_xvector_only = ("Fast" in mode)
    final_ref_text = ref_text
    audio_tuple = _audio_to_tuple(ref_audio)

    if not use_xvector_only:
        if not final_ref_text or not final_ref_text.strip():
            print("Auto-transcribing reference...")
            try:
                final_ref_text = transcribe_reference(ref_audio, True, language)
                if not final_ref_text or "Error" in final_ref_text:
                     return None, f"Transcription failed: {final_ref_text}", None, None, None, None
            except Exception as e:
                return None, f"Transcribe Error: {e}", None, None, None, None
    else:
        final_ref_text = None

    try:
        # 2. Chunk Target Text
        text_chunks, tts_filename = text_chunk(target_text, language, char_limit=280)
        chunk_files = []
        tts = get_model("Base", model_size)

        # 3. Generate Loop
        for i, chunk in enumerate(text_chunks):
            wavs, sr = tts.generate_voice_clone(
                text=chunk.strip(),
                language=language,
                ref_audio=audio_tuple,
                ref_text=final_ref_text.strip() if final_ref_text else None,
                x_vector_only_mode=use_xvector_only,
                max_new_tokens=2048,
            )
            # Save immediately
            temp_filename = f"temp_clone_{i}_{os.getpid()}.wav"
            sf.write(temp_filename, wavs[0], sr)
            chunk_files.append(temp_filename)

            # Clear memory
            del wavs
            torch.cuda.empty_cache()
            gc.collect()

        # 4. Stitch & Process
        stitched_file = stitch_chunk_files(chunk_files,tts_filename)
        final_audio, srt1, srt2, srt3, srt4 = process_audio_output(stitched_file, make_subs, remove_silence, language)
        return final_audio, f"Success! Mode: {mode}", srt1, srt2, srt3, srt4

    except Exception as e:
        return None, f"Error: {e}", None, None, None, None


import click
@click.command()
@click.option("--audio-json", default=None, help="Path to JSON file for batch TTS generation.")
@click.option("--output-dir", default="./generated_audio", show_default=True, help="Folder for generated audio files.")
@click.option("--model-type", default="voice_design", type=click.Choice(["voice_design", "custom_voice", "voice_clone"]), show_default=True, help="Use 'voice_design' (no speaker), 'custom_voice' (speaker), or 'voice_clone' (reference audio).")
@click.option("--language", default="Portuguese", show_default=True, help="Default language when an item in audio.json does not provide one.")
@click.option("--speaker", default="", show_default=True, help="Default speaker for custom_voice mode when an item in audio.json does not provide one.")
@click.option(
    "--instruct",
    default="Fale em portugues com um tom divertido, brincalhao e bem humorado.",
    show_default=True,
    help="Default speaking style when an item in audio.json does not provide one.",
)
@click.option("--clone-ref-audio", default="", show_default=True, help="Default reference audio path for voice_clone mode.")
@click.option("--clone-ref-text", default="", show_default=True, help="Default transcript of the reference audio for voice_clone mode.")
@click.option("--clone-mode", default="fast", type=click.Choice(["fast", "high_quality"]), show_default=True, help="Voice clone mode: fast (x-vector only) or high_quality (uses reference transcript).")
@click.option("--model-size", default="1.7B", type=click.Choice(MODEL_SIZES), show_default=True, help="Model size for batch generation.")
@click.option("--output-format", default="mp3", type=click.Choice(["wav", "mp3"]), show_default=True, help="Default output format when filename has no supported extension.")
@click.option("--remove-silence", is_flag=True, default=False, help="Remove silence in output files.")
@click.option("--make-subs", is_flag=True, default=False, help="Generate subtitles for output files.")
def main(
    audio_json,
    output_dir,
    model_type,
    language,
    speaker,
    instruct,
    clone_ref_audio,
    clone_ref_text,
    clone_mode,
    model_size,
    output_format,
    remove_silence,
    make_subs,
):
    if not audio_json:
        raise click.UsageError("Please provide --audio-json <path> to generate audio from a JSON file.")

    generated = generate_from_json(
        audio_json_path=audio_json,
        output_dir=output_dir,
        model_type=model_type,
        default_language=language,
        default_speaker=speaker,
        default_instruct=instruct,
        clone_ref_audio=clone_ref_audio,
        clone_ref_text=clone_ref_text,
        clone_mode=clone_mode,
        model_size=model_size,
        output_format=output_format,
        remove_silence=remove_silence,
        make_subs=make_subs,
    )
    print(f"Done. Generated {len(generated)} audio file(s) in: {output_dir}")

if __name__ == "__main__":
    main()    
