# Demo Video 
<https://youtu.be/cWd5Yz30CEI?si=Qf5_znDkPowZ3t5d>

# Data Preparation

## Dateset
1. LibriSpeech
2. MUSAN Dataset

## Steps
1. **Speech selection**
   - Take 1.5–2.5 s clean speech (pad short gaps with brief silence if needed).
   - Place it randomly within a 4s canvas (uniform start). 

2. **RIR setup**
   - Randomize azimuth.
   - T60 = 150 ms, distance = 0.5 m (fixed, will randomize if needed).
   - Optionally attenuate late reverb.
   - Convolve speech with RIR → **reverberant speech**

4. **Labeling**
   - Run Silero VAD on the **reverberant speech** to get:
     - `lab_frame` (label per-frame with 10 ms hop, e.g., 160 samples @16 kHz)

5. **Add noise**
   - Choose noise type (environmental/musical/white noise), SNR range −5 to 20 dB.
