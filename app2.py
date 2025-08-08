import tkinter as tk
from tkinter import messagebox, simpledialog
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import datetime
import os
import threading
import random
import simpleaudio as sa
from typing import List, Optional
from pathlib import Path
from pydub import AudioSegment

# --- Configuration & Constants ---
# Dark theme color palette
COLOR_PRIMARY = "#00B8D4"  # Vibrant Cyan
COLOR_SECONDARY = "#212121"  # Deep Charcoal
COLOR_ACCENT = "#4CAF50"  # Bright Green
COLOR_TEXT = "#E0E0E0"  # Light Gray
COLOR_ERROR = "#FF5252"  # Red
COLOR_BG = "#121212"  # Very Dark Gray
COLOR_CANVAS_BG = "#303030" # Dark gray for canvas
COLOR_WAVEFORM = "#00B8D4" # Curve color
COLOR_PLAYBACK_LINE = "#FFC107"  # Amber

FONT_TITLE = ("Arial", 20, "bold")
FONT_BUTTON = ("Helvetica", 12, "bold")
FONT_LABEL = ("Helvetica", 11, "italic")
FONT_QUOTE = ("Arial", 12, "bold")
FONT_TIMER = ("Courier", 14, "bold")

# --- Application Logic ---
class SoundboxRecorderApp:
    def __init__(self, root: tk.Tk):
        """Initializes the main application window and components."""
        self.root = root
        self.root.title("✨ Recorder ✨")
        self.root.geometry("600x600")
        self.root.configure(bg=COLOR_BG)

        # Global state variables
        self.recording = False
        self.playback_active = False
        self.record_frames = []
        self.fs = 44100
        self.channels = 2
        self.playback_audio = None
        self.playback_position = 0
        self.total_samples = 0
        self.start_time = None
        self.live_data_buffer = np.array([])
        
        self.base_dir = Path(__file__).parent
        self.audio_files_dir = self.base_dir / "audio_files"
        self.mp3_files: List[Path] = self._load_audio_files()

        self._setup_ui()
        
    def _load_audio_files(self) -> List[Path]:
        """Loads MP3 files from the audio_files directory."""
        if not self.audio_files_dir.exists():
            messagebox.showerror("Folder Not Found", "The 'audio_files' folder does not exist! Please create it and add some .mp3 files.")
            return []
        
        return list(self.audio_files_dir.glob("*.mp3"))

    def _setup_ui(self):
        """Builds all the widgets and packs them into the window."""
        title_lbl = tk.Label(self.root, text="Recorder", font=FONT_TITLE, fg=COLOR_PRIMARY, bg=COLOR_BG)
        title_lbl.pack(pady=20)

        button_frame = tk.Frame(self.root, bg=COLOR_BG)
        button_frame.pack(pady=10)

        self.record_btn = tk.Button(button_frame, text="⏺️ Start Recording", command=self.start_record,
                                    width=20, height=2, bg=COLOR_ACCENT, fg=COLOR_TEXT, font=FONT_BUTTON,
                                    relief="flat", borderwidth=0)
        self.record_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = tk.Button(button_frame, text="⏹️ Stop Recording", command=self.stop_record,
                                  width=20, height=2, bg=COLOR_ERROR, fg=COLOR_TEXT, font=FONT_BUTTON,
                                  relief="flat", borderwidth=0, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        self.play_btn = tk.Button(button_frame, text="▶️ Play Recording", command=self.play_and_visualize,
                                  width=20, height=2, bg=COLOR_PRIMARY, fg=COLOR_TEXT, font=FONT_BUTTON,
                                  relief="flat", borderwidth=0, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=10)

        self.status_lbl = tk.Label(self.root, text="Ready to record!", font=FONT_LABEL, fg=COLOR_TEXT, bg=COLOR_BG,
                                   wraplength=500, justify="center")
        self.status_lbl.pack(pady=10)
        
        self.time_lbl = tk.Label(self.root, text="00:00.00", font=FONT_TIMER, fg=COLOR_TEXT, bg=COLOR_BG)
        self.time_lbl.pack(pady=5)

        self.audio_display_lbl = tk.Label(self.root, text="Let's Record", font=FONT_QUOTE, fg=COLOR_WAVEFORM, bg=COLOR_BG,
                                             wraplength=500, justify="center")
        self.audio_display_lbl.pack(pady=10)
        
        self.canvas = tk.Canvas(self.root, bg=COLOR_CANVAS_BG, height=100)
        self.canvas.pack(pady=20, padx=10, fill=tk.BOTH, expand=True)

    def _draw_live_waveform(self, data):
        self.canvas.delete("all")
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        horizontal_scale = 5
        
        self.live_data_buffer = np.concatenate((self.live_data_buffer, data.flatten()))
        if len(self.live_data_buffer) * horizontal_scale > width:
            self.live_data_buffer = self.live_data_buffer[-(width // horizontal_scale):]
        
        amps = self.live_data_buffer
        mx = np.max(np.abs(amps)) if amps.size > 0 else 1
        
        points = []
        for i, a in enumerate(amps):
            y = height // 2 - (a / mx) * height // 2
            x = i * horizontal_scale
            points.append(x)
            points.append(y)

        if points:
            self.canvas.create_line(points, fill=COLOR_WAVEFORM, smooth=True, width=2)

    def _draw_full_waveform(self, audio_data):
        self.canvas.delete("all")
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        
        amps = np.abs(audio_data).flatten()
        
        if len(amps) > width:
            step = len(amps) / width
            scaled_amps = np.array([amps[int(i * step)] for i in range(width)])
        else:
            scaled_amps = amps
        
        mx = np.max(scaled_amps) if scaled_amps.size > 0 else 1
        
        points = []
        for i, a in enumerate(scaled_amps):
            y = height // 2 - (a / mx) * height // 2
            points.append(i)
            points.append(y)
        
        if points:
            self.canvas.create_line(points, fill=COLOR_WAVEFORM, smooth=True, width=2)

    def _reset_graph(self):
        self.live_data_buffer = np.array([])
        self.canvas.delete("all")

    def _play_random_audio(self):
        if not self.mp3_files:
            messagebox.showinfo("No Files", "No MP3 files found in the 'audio_files' folder.")
            return

        selected_mp3 = random.choice(self.mp3_files)
        self.audio_display_lbl.config(text=f"🎶 Playing: {selected_mp3.name}")

        def play_thread():
            try:
                audio = AudioSegment.from_mp3(selected_mp3)
                wav_data = audio.export(format="wav")
                wav_obj = sa.WaveObject.from_wave_file(wav_data)
                play_obj = wav_obj.play()
                play_obj.wait_done()
            except Exception as e:
                messagebox.showerror("Playback Error", f"Failed to play {selected_mp3.name}: {e}")
        
        threading.Thread(target=play_thread).start()


    def _play_sound_effect(self, filename: str):
        try:
            wave_obj = sa.WaveObject.from_wave_file(self.base_dir / filename)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception:
            pass

    def _audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.record_frames.append(indata.copy())
            self._draw_live_waveform(indata)

    def _record_thread(self):
        try:
            with sd.InputStream(samplerate=self.fs, channels=self.channels, callback=self._audio_callback):
                while self.recording:
                    sd.sleep(10)
        except Exception as e:
            messagebox.showerror("Recording Error", f"An error occurred during recording: {e}")
            self.stop_record()

    def start_record(self):
        if self.recording or self.playback_active:
            return
        
        self.start_time = datetime.datetime.now()
        self._reset_graph()
        
        self._play_sound_effect("click.wav")
        self.recording = True
        self.record_frames = []
        self.status_lbl.config(text="🎙️ Recording... Talk into your mic!", fg=COLOR_ACCENT)
        
        self.record_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.DISABLED)

        self.record_thread = threading.Thread(target=self._record_thread)
        self.record_thread.start()
        self._update_timer()

    def stop_record(self):
        if not self.recording:
            return

        self._play_sound_effect("click.wav")
        self.recording = False
        self.status_lbl.config(text="🌀 Processing...", fg=COLOR_PRIMARY)
        
        self.record_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

        if self.record_frames:
            audio = np.concatenate(self.record_frames, axis=0)
            self.playback_audio = audio
            self.total_samples = len(audio)

            self.play_btn.config(state=tk.NORMAL)
            self.status_lbl.config(text="✅ Recording finished!", fg=COLOR_TEXT)
            
            self._draw_full_waveform(self.playback_audio)
            
            filename = simpledialog.askstring("Save Recording", "Enter a filename (e.g., my_recording):")
            
            if filename:
                sanitized_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-')).rstrip()
                final_filename = f"{sanitized_filename}.wav"
                
                write(final_filename, self.fs, audio)
                messagebox.showinfo("✅ Saved!", f"Recording saved as {final_filename}")
            else:
                messagebox.showinfo("❌ Canceled", "Recording was not saved.")
        else:
            messagebox.showinfo("⚠️ No Audio", "No audio data was captured. Was your mic on?")
            self.status_lbl.config(text="Ready to record!", fg=COLOR_TEXT)
            self._reset_graph()
    
    def _update_timer(self):
        if self.recording or self.playback_active:
            elapsed_time = datetime.datetime.now() - self.start_time
            self.time_lbl.config(text=f"{int(elapsed_time.total_seconds() // 60):02d}:{int(elapsed_time.total_seconds() % 60):02d}.{int((elapsed_time.total_seconds() * 100) % 100):02d}")
            self.root.after(10, self._update_timer)
        else:
            self.time_lbl.config(text="00:00.00")

    def play_and_visualize(self):
        if self.playback_active or self.playback_audio is None:
            return

        self.playback_active = True
        self.playback_position = 0
        self.status_lbl.config(text="▶️ Playing back...", fg=COLOR_ACCENT)
        self.record_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        
        self.start_time = datetime.datetime.now()
        
        self.playback_thread = threading.Thread(target=self._playback_audio_thread)
        self.playback_thread.start()

        self._update_timer()

    def _playback_audio_thread(self):
        try:
            with sd.OutputStream(samplerate=self.fs, channels=self.channels) as stream:
                self._update_playback_visual() 
                stream.write(self.playback_audio)
        except Exception as e:
            messagebox.showerror("Playback Error", f"An error occurred during playback: {e}")
        finally:
            self.playback_active = False
            self.status_lbl.config(text="✅ Playback finished!", fg=COLOR_TEXT)
            self.record_btn.config(state=tk.NORMAL)
            self.play_btn.config(state=tk.NORMAL)
            self.time_lbl.config(text=f"{len(self.playback_audio)/self.fs:.2f}s")
            self._play_random_audio()
            self._draw_full_waveform(self.playback_audio)

    def _update_playback_visual(self):
        if not self.playback_active:
            return

        elapsed_time = (datetime.datetime.now() - self.start_time).total_seconds()
        self.playback_position = int(elapsed_time * self.fs)

        if self.playback_position < self.total_samples:
            self.canvas.delete("all")
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            horizontal_scale = 5 

            display_width_pixels = width
            display_samples = display_width_pixels // horizontal_scale
            
            start_index = max(0, self.playback_position - display_samples // 2)
            end_index = min(self.total_samples, start_index + display_samples)
            
            display_audio = self.playback_audio[start_index:end_index]
            
            amps = display_audio.flatten()
            mx = np.max(np.abs(amps)) if amps.size > 0 else 1
            
            points = []
            for i, a in enumerate(amps):
                y = height // 2 - (a / mx) * height // 2
                x = i * horizontal_scale
                points.append(x)
                points.append(y)
            
            if points:
                self.canvas.create_line(points, fill=COLOR_WAVEFORM, smooth=True, width=2)
            
            center_x = width // 2
            self.canvas.create_line(center_x, 0, center_x, height, fill=COLOR_PLAYBACK_LINE, width=2, tags="playback_line")

            self.root.after(10, self._update_playback_visual)
        else:
            self.canvas.delete("playback_line")

if __name__ == "__main__":
    root = tk.Tk()
    app = SoundboxRecorderApp(root)
    root.mainloop()