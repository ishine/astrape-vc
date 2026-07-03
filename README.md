# astrape-vst

Strict-causal, zero-shot voice conversion VST — real-time on Apple Silicon (MPS).

| | | 
|---|---|
| cos768 | **0.935** (probe, 8L StridingAdapter) |
| Encoder params | 24.9M |
| Algorithmic latency | ~49ms E2E (0 look-ahead) |
| Platform | macOS / Apple Silicon (MPS) |

## Design

- **0 look-ahead** — strictly causal. No future frames, KV-cache streaming.
- **Content/speaker split** — *what* is said (768d content @25Hz) vs *who* says it (global embedding).
- **Teacher–student** — trained by distilling a frozen, bidirectional MioCodec teacher into a causal student.
- **Zero-shot VC** — GRL disentanglement + Q2D2 bottleneck. No parallel data needed.

## Quick Start

```bash
# Voicebank
python -m astrape.build_voicebank ref.wav -o speaker.astrape

# Convert
python -m astrape.evaluate --source src.wav --target speaker.astrape --output out.wav
```

## References

- **Q2D2** — Shuster & Nachmani, "Two-Dimensional Quantization for Geometry-Aware Audio Coding", ICML 2026, arXiv:2512.01537
- **Mamba** — Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023, arXiv:2312.00752
- **Hyena** — Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models", 2023, arXiv:2302.10866
- **MioCodec** — Aratako/MioCodec-25Hz-44.1kHz-v2 (teacher codec, HuggingFace)
- **WavLM** — Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", 2022
- **GRL** — Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation", ICML 2015
- **ConvNeXt** — Liu et al., "A ConvNet for the 2020s", CVPR 2022
- **WavTokenizer** — Ji et al., "WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer", 2024
- **Snake / BigVGAN** — Lee et al., "BigVGAN: A Universal Neural Vocoder", 2023, arXiv:2206.02944
- **iSTFT / Vocos** — Siuzdak et al., "Vocos: Closing the Gap Between Time-Domain and Fourier-Based Neural Vocoders", 2024, arXiv:2306.00819
- **Predictive Coding** — Oord et al., "Representation Learning with Contrastive Predictive Coding (CPC)", 2018
- **APCodec** — Ai et al., "APCodec: A Neural Audio Codec", IEEE/ACM TASLP 2024, arXiv:2402.10533
- FreeGAN — Is GAN Necessary for Mel Vocoder? (2508.07711)
- DLL-APNet — Distilled Low-Latency causal vocoder (2509.13667)
- APNet2 (2311.11545) · APNet (2305.07952)
- Low-Latency phase prediction, anti-wrapping (2403.17378)
- FreeV — pinv-mel magnitude prior (2406.08196)
- Wavehax — aliasing-free F0 harmonic prior (2411.06807) · MS-Wavehax / streaming analysis (2506.03554)
- Universal Harmonic Discriminator (2512.03486)
- Prosody-Guided Harmonic Attention (2601.14472)

