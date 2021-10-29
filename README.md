[![fxslava](https://circleci.com/gh/fxslava/tiny_mp2v_dec.svg?style=svg)](https://app.circleci.com/pipelines/github/fxslava/tiny_mp2v_dec?branch=master)

# _Draft_: tiny_mp2v_dec
The idea of the project is to create a super light mpeg2 cross-platform decoder for the possibility of its use in ARM microcontrollers with a rather low performance and a small amount of memory.
The goal of the project is to achieve the highest performance from a software decoder on the Raspberry Pi 4 platform, ultra low latency, support for spatial and temporal scalability features.

# How to build
1. Clone repository:
```console
git clone https://github.com/fxslava/tiny_mp2v_dec.git | cd ./tiny_mp2v_dec
```
2. Generate project with cmake:
```console
cmake -S ./ -B ./build-x64 | cd ./build-x64
```
3. Make it:
```console
make .
```


# Supported platforms
- [X] Linux x64/aarch64
- [X] Windows x64/aarch64

# Done features
- [X] Support SSE for x64 and NEON for aarch64
- [X] Multithreading:
  - [X] by pictures
  - [X] by slices
- [X] Supported chroma formats: 4:2:0, 4:2:2, 4:4:4

# TODO
- [ ] Interlaced streams support
- [ ] Field frames support
- [ ] Spatial scalability support
- [ ] Temporal scalability support
- [ ] Enhanced data support
- [ ] Dual prime motion compensation

# Status
Product is not ready for release yet, external API is incomplete. Also, the decoder is poorly tested on streams with support for current features.
Nevertheless, you can test the decoder on your platform, for this you can use the sample: [tiny_mp2v_dec/tiny_decoder/tiny_mp2v_dec.cpp](tiny_decoder/tiny_mp2v_dec.cpp).
Run with cmd:
```console
./tiny_mp2v_dec_sample -v 1080p_422.m2v -o output_yuv.yuv
```
**_NOTE:_**  For performance testing, you should turn off file output.
