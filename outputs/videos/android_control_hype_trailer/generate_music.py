#!/usr/bin/env python3
"""Generate cinematic hype trailer music: layered electronic with build/drop."""
import numpy as np
import struct
import wave

sr = 44100
duration = 30
n = int(sr * duration)
t = np.arange(n) / sr

bpm = 132
beat_dur = 60 / bpm
beats = int(duration / beat_dur)


def adsr(length, attack=0.01, decay=0.05, sustain=0.7, release=0.1):
    """ADSR envelope as a vector of length samples."""
    a = max(1, int(attack * sr))
    d = max(1, int(decay * sr))
    r = max(1, int(release * sr))
    env = np.zeros(length)
    if length > a:
        env[:a] = np.linspace(0, 1, a)
    if length > a + d:
        env[a:a+d] = np.linspace(1, sustain, d)
        s_end = max(a + d, length - r)
        env[a+d:s_end] = sustain
        if length > s_end:
            env[s_end:] = np.linspace(sustain, 0, length - s_end)
    return env


def add_note(buf, start_sample, freq, dur_sec, gain=0.15, wave_type='sine',
             attack=0.01, decay=0.05, sustain=0.7, release=0.1, detune=0):
    """Mix a single note into buf at start_sample."""
    length = int(dur_sec * sr)
    if start_sample >= n or length <= 0:
        return
    end = min(start_sample + length, n)
    actual = end - start_sample
    tt = np.arange(actual) / sr
    if wave_type == 'sine':
        tone = np.sin(2 * np.pi * freq * tt)
        if detune:
            tone += 0.5 * np.sin(2 * np.pi * (freq * (1 + detune)) * tt)
            tone /= 1.5
    elif wave_type == 'saw':
        tone = 2 * (freq * tt - np.floor(0.5 + freq * tt))
        if detune:
            tone += 2 * (freq * (1+detune) * tt - np.floor(0.5 + freq * (1+detune) * tt))
            tone /= 2
    elif wave_type == 'square':
        tone = np.sign(np.sin(2 * np.pi * freq * tt))
    elif wave_type == 'tri':
        tone = 2 * np.abs(2 * (freq * tt - np.floor(0.5 + freq * tt))) - 1
    else:
        tone = np.sin(2 * np.pi * freq * tt)
    env = adsr(actual, attack, decay, sustain, release)
    buf[start_sample:end] += gain * tone * env


# === Note frequencies (A minor key) ===
A2, C3, D3, E3, F3, G3 = 110, 130.81, 146.83, 164.81, 174.61, 196.00
A3, C4, D4, E4, F4, G4 = 220, 261.63, 293.66, 329.63, 349.23, 392.00
A4, C5, D5, E5, F5, G5 = 440, 523.25, 587.33, 659.25, 698.46, 783.99
A5, C6, E6 = 880, 1046.50, 1318.51

# Section boundaries (seconds): intro→demo1→demo2→metrics→tasks→outro
# 0..3 intro / 3..9 demo1 / 9..15 demo2 / 15..21 metrics(build+drop@21) / 21..26 tasks / 26..30 outro

# === Buffers ===
kick = np.zeros(n)
snare = np.zeros(n)
hat = np.zeros(n)
bass = np.zeros(n)
pad = np.zeros(n)
arp = np.zeros(n)
lead = np.zeros(n)
sub = np.zeros(n)

rng = np.random.default_rng(7)

# === Drums ===
def kick_hit(start, gain=0.5):
    if start >= n:
        return
    L = int(0.18 * sr)
    L = min(L, n - start)
    tt = np.arange(L) / sr
    pitch = 110 * np.exp(-30 * tt) + 50
    body = np.sin(2 * np.pi * np.cumsum(pitch) / sr)
    click = rng.standard_normal(L) * np.exp(-180 * tt) * 0.3
    env = np.exp(-12 * tt)
    kick[start:start+L] += gain * (body + click) * env


def snare_hit(start, gain=0.3):
    if start >= n:
        return
    L = int(0.15 * sr)
    L = min(L, n - start)
    tt = np.arange(L) / sr
    body = np.sin(2 * np.pi * 200 * tt) * np.exp(-15 * tt)
    noise = rng.standard_normal(L) * np.exp(-12 * tt)
    snare[start:start+L] += gain * (0.4 * body + 0.6 * noise)


def hat_hit(start, gain=0.08, dur=0.04):
    if start >= n:
        return
    L = int(dur * sr)
    L = min(L, n - start)
    tt = np.arange(L) / sr
    # high-freq noise
    noise = rng.standard_normal(L)
    # crude high-pass: subtract running mean
    noise = noise - np.convolve(noise, np.ones(8)/8, mode='same')
    env = np.exp(-40 * tt)
    hat[start:start+L] += gain * noise * env


# === Pad chord progression (Am - F - C - G) — 4 bars per chord ===
# chord per beat (4 beats per bar)
chord_progression = [
    # bar 0-1: Am
    (A3, C4, E4),
    # bar 2-3: F
    (F3, A3, C4),
    # bar 4-5: C
    (C4, E4, G4),
    # bar 6-7: G
    (G3, B := 246.94, D4),
]

# === Build the track section by section ===

# Track sections by time:
# 0..3   intro (sparse pad)
# 3..15  groove (kick + bass + arp + pad)
# 15..21 build (rising filter, drum fills) → drop at 21
# 21..26 drop (full drums + lead)
# 26..30 outro (pad fades)

# --- Pad: continuous chord progression through whole track ---
chord_dur = 4 * beat_dur  # one chord per bar (4 beats)
total_chords = int(np.ceil(duration / chord_dur)) + 1
for ci in range(total_chords):
    start = int(ci * chord_dur * sr)
    chord = chord_progression[ci % len(chord_progression)]
    for f in chord:
        add_note(pad, start, f, chord_dur, gain=0.06, wave_type='sine',
                 attack=0.4, decay=0.3, sustain=0.6, release=0.5)
        # detuned octave for warmth
        add_note(pad, start, f * 2, chord_dur, gain=0.025, wave_type='sine',
                 attack=0.5, decay=0.4, sustain=0.5, release=0.5)

# --- Sub bass: root note of each chord ---
for ci in range(total_chords):
    start = int(ci * chord_dur * sr)
    if ci * chord_dur >= duration:
        break
    chord = chord_progression[ci % len(chord_progression)]
    root = chord[0] / 2  # one octave down for sub
    add_note(sub, start, root, chord_dur, gain=0.18, wave_type='sine',
             attack=0.05, decay=0.1, sustain=0.8, release=0.2)

# --- Drums: groove from 3s onwards, fuller from 9s, fills at 18s, drop at 21s ---
for beat in range(beats):
    bt = beat * beat_dur
    bs = int(bt * sr)
    if bs >= n:
        break

    # Kick on 1, 3
    if beat % 4 in (0, 2):
        if 3.0 <= bt < 21.0:
            kick_hit(bs, gain=0.45)
        elif 21.0 <= bt < 28.0:
            kick_hit(bs, gain=0.6)
    # Extra kick on offbeat in drop
    if 21.0 <= bt < 26.0 and beat % 4 == 3:
        kick_hit(bs + int(0.5 * beat_dur * sr), gain=0.35)

    # Snare on 2, 4
    if beat % 4 in (1, 3):
        if 6.0 <= bt < 21.0:
            snare_hit(bs, gain=0.22)
        elif 21.0 <= bt < 27.0:
            snare_hit(bs, gain=0.32)

    # Hi-hats: eighth notes from 5s onwards
    if 5.0 <= bt < 21.0:
        hat_hit(bs, gain=0.06)
        hat_hit(bs + int(0.5 * beat_dur * sr), gain=0.05)
    elif 21.0 <= bt < 27.0:
        # 16ths in drop
        for k in range(4):
            hat_hit(bs + int(k * 0.25 * beat_dur * sr), gain=0.05)

# --- Build-up snare roll & riser from 18s to 21s ---
build_start = 18.0
build_end = 21.0
roll_div = 16  # subdivisions across build
for k in range(roll_div):
    progress = k / roll_div
    # accelerate
    bt = build_start + (build_end - build_start) * (progress ** 1.6)
    bs = int(bt * sr)
    snare_hit(bs, gain=0.10 + 0.20 * progress)

# Riser: pitch sweep
riser_len = int((build_end - build_start) * sr)
riser_start = int(build_start * sr)
rt = np.arange(riser_len) / sr
sweep_freq = 200 + (rt / (build_end - build_start)) ** 2 * 1800
riser = 0.0 * np.zeros(riser_len)
phase = np.cumsum(sweep_freq) / sr
riser = np.sin(2 * np.pi * phase) * 0.10
# Add noise sweep
noise_riser = rng.standard_normal(riser_len) * 0.04
# fade in
fade = np.linspace(0, 1, riser_len) ** 1.5
end = min(riser_start + riser_len, n)
length = end - riser_start
snare[riser_start:end] += (riser[:length] + noise_riser[:length]) * fade[:length]

# --- Bass line: groove from 9s to 26s ---
# Following chord roots: Am F C G
bass_pattern_per_chord = [
    # 8 eighth notes per bar — pattern with octave jumps
    (1, 0, 1, 1, 0.5, 1, 0, 1)
]

for beat in range(beats):
    bt = beat * beat_dur
    if bt < 9.0 or bt >= 26.0:
        continue
    bs = int(bt * sr)
    if bs >= n:
        break
    chord_idx = (beat // 4) % len(chord_progression)
    root = chord_progression[chord_idx][0]
    # Eighth note pattern within the beat
    for k in range(2):
        sub_start = bs + int(k * 0.5 * beat_dur * sr)
        if sub_start >= n:
            break
        # Slight syncopation
        gain = 0.18 if (beat % 2 == 0 and k == 0) else 0.10
        add_note(bass, sub_start, root, 0.5 * beat_dur, gain=gain,
                 wave_type='saw', attack=0.005, decay=0.05, sustain=0.4,
                 release=0.05, detune=0.005)

# --- Arpeggio: from 5s onwards through metrics, drops out for build ---
arp_pattern = [0, 1, 2, 1]  # chord notes
for beat in range(beats):
    bt = beat * beat_dur
    if bt < 5.0:
        continue
    if 18.0 <= bt < 21.0:  # silence in build for tension
        continue
    if bt >= 26.0:
        continue
    bs = int(bt * sr)
    if bs >= n:
        break
    chord_idx = (beat // 4) % len(chord_progression)
    chord = chord_progression[chord_idx]
    # 4 16th notes per beat
    for k in range(4):
        sub_start = bs + int(k * 0.25 * beat_dur * sr)
        if sub_start >= n:
            break
        note_idx = arp_pattern[(beat * 4 + k) % len(arp_pattern)]
        f = chord[note_idx % 3] * 2  # octave up
        gain = 0.06 if 21.0 <= bt < 26.0 else 0.045
        add_note(arp, sub_start, f, 0.25 * beat_dur, gain=gain,
                 wave_type='tri', attack=0.005, decay=0.04, sustain=0.3,
                 release=0.04)

# --- Lead melody: prominent in drop (21..26) and outro ---
# Simple memorable motif over Am - F - C - G
lead_motif = [
    # (beat_offset, freq_index_in_chord, duration_beats)
    (0, 2, 1.0),    # E5
    (1, 1, 0.5),    # C5
    (1.5, 2, 0.5),  # E5
    (2, 0, 1.0),    # A4
    (3, 1, 1.0),    # C5
]

for bar in range(int((duration - 21) / (4 * beat_dur)) + 2):
    bar_start = 21.0 + bar * 4 * beat_dur
    if bar_start >= 28.5:
        break
    chord_idx = bar % len(chord_progression)
    chord = chord_progression[chord_idx]
    for offset, idx, dur in lead_motif:
        nt = bar_start + offset * beat_dur
        if nt >= duration:
            break
        ns = int(nt * sr)
        f = chord[idx % 3] * 2
        add_note(lead, ns, f, dur * beat_dur, gain=0.08,
                 wave_type='saw', attack=0.01, decay=0.08, sustain=0.5,
                 release=0.1, detune=0.003)

# Hook lead during intro (0-3s): single sustained note that resolves
add_note(lead, int(0.5 * sr), E5, 1.5, gain=0.06,
         wave_type='sine', attack=0.3, decay=0.4, sustain=0.5, release=0.5)
add_note(lead, int(2.0 * sr), A5, 1.0, gain=0.05,
         wave_type='sine', attack=0.2, decay=0.3, sustain=0.5, release=0.4)

# === Mix ===
mix = (
    1.00 * kick +
    0.90 * snare +
    0.80 * hat +
    0.85 * bass +
    0.55 * sub +
    0.70 * pad +
    0.65 * arp +
    0.75 * lead
)

# Light compression (soft tanh saturation)
mix = np.tanh(mix * 1.2) * 0.85

# Normalize to ~-2dBFS
peak = np.max(np.abs(mix))
if peak > 0:
    mix = mix / peak * 0.92

# Fade in/out
fade_in = np.linspace(0, 1, int(0.5 * sr)) ** 2
fade_out = np.linspace(1, 0, int(2.0 * sr)) ** 2
mix[:len(fade_in)] *= fade_in
mix[-len(fade_out):] *= fade_out

# Save WAV
with wave.open('assets/music.wav', 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    samples = np.clip(mix * 32767, -32768, 32767).astype(np.int16)
    wf.writeframes(samples.tobytes())

print(f"Generated {duration}s music at {sr}Hz, peak={peak:.3f}")
print(f"File size: {len(samples)*2/1024:.1f}KB")
