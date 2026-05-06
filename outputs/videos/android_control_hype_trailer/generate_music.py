#!/usr/bin/env python3
"""Generate a proper electronic track with kick, hat, bass, melody, and pad."""
import numpy as np
import struct
import wave

sr = 44100
duration = 30  # seconds
n_samples = int(sr * duration)
t = np.arange(n_samples) / sr

# BPM = 128
bpm = 128
beat_duration = 60 / bpm  # seconds per beat
beats = int(duration / beat_duration)

def envelope(t, attack=0.01, decay=0.1, sustain=0.3, release=0.3):
    """Simple ADSR envelope."""
    attack_s = int(attack * sr)
    decay_s = int(decay * sr)
    release_s = int(release * sr)
    env = np.zeros(len(t))
    for i in range(len(t)):
        if i < attack_s:
            env[i] = i / attack_s
        elif i < attack_s + decay_s:
            env[i] = 1.0 - (i - attack_s) / decay_s * (1.0 - sustain)
        elif i < len(t) - release_s:
            env[i] = sustain
        else:
            env[i] = sustain * (1.0 - (i - (len(t) - release_s)) / release_s)
    return np.maximum(env, 0)

# Kick drum
kick = np.zeros(n_samples)
for beat in range(beats):
    start = int(beat * beat_duration * sr)
    if start >= n_samples:
        break
    freq = 150 * np.exp(-40 * (t[start:start+int(0.1*sr)]))
    kick[start:start+int(0.1*sr)] = 0.3 * np.sin(2 * np.pi * freq * np.arange(min(int(0.1*sr), n_samples-start)) / sr) * envelope(np.arange(min(int(0.1*sr), n_samples-start))/sr, attack=0.001, decay=0.05, sustain=0, release=0.05)

# Hi-hat
hat = np.zeros(n_samples)
for beat in range(beats):
    start = int((beat + 0.5) * beat_duration * sr)
    if start >= n_samples:
        break
    noise = np.random.randn(min(int(0.05*sr), n_samples-start))
    hat[start:start+int(0.05*sr)] = 0.05 * noise * envelope(np.arange(min(int(0.05*sr), n_samples-start))/sr, attack=0.001, decay=0.02, sustain=0, release=0.02)

# Bass line
bass = np.zeros(n_samples)
bass_freqs = [55, 55, 65.41, 73.42, 55, 55, 65.41, 82.41]  # A2, C3, D3, E2
for beat in range(beats):
    start = int(beat * beat_duration * sr)
    if start >= n_samples:
        break
    freq = bass_freqs[beat % len(bass_freqs)]
    bass[start:start+int(beat_duration*sr)] = 0.15 * np.sin(2 * np.pi * freq * (t[start:start+int(beat_duration*sr)])) * envelope(t[start:start+int(beat_duration*sr)], attack=0.01, decay=0.1, sustain=0.7, release=0.1)

# Melody/arpeggio
melody = np.zeros(n_samples)
melody_notes = [440, 523.25, 659.25, 783.99, 659.25, 523.25, 440, 349.23]  # A4, C5, E5, G5
for beat in range(beats):
    start = int(beat * beat_duration * sr)
    if start >= n_samples:
        break
    freq = melody_notes[beat % len(melody_notes)]
    melody[start:start+int(beat_duration*0.8*sr)] = 0.08 * np.sin(2 * np.pi * freq * (t[start:start+int(beat_duration*0.8*sr)])) * envelope(t[start:start+int(beat_duration*0.8*sr)], attack=0.01, decay=0.1, sustain=0.5, release=0.1)

# Pad (ambient background)
pad = np.zeros(n_samples)
pad_freqs = [220, 330, 440]  # A3, E4, A4
for freq in pad_freqs:
    pad += 0.03 * np.sin(2 * np.pi * freq * t) * envelope(t, attack=0.5, decay=0.5, sustain=0.3, release=0.5)

# Mix all layers
mix = kick + hat + bass + melody + pad

# Normalize
max_val = np.max(np.abs(mix))
if max_val > 0:
    mix = mix / max_val * 0.9

# Fade in/out
fade_in = np.linspace(0, 1, int(2*sr))
fade_out = np.linspace(1, 0, int(2*sr))
mix[:len(fade_in)] *= fade_in
mix[-len(fade_out):] *= fade_out

# Save as WAV
with wave.open('assets/music.wav', 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    for sample in mix:
        wf.writeframes(struct.pack('<h', int(np.clip(sample * 32767, -32768, 32767))))

print(f"Generated {duration}s music at {sr}Hz")
print(f"File size: {np.ceil(duration * sr * 2 / 1024):.1f}KB")
