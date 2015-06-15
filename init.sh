#!/bin/bash
curl -# --silent http://download.linuxaudio.org/lilypond/binaries/darwin-x86/lilypond-2.18.2-1.darwin-x86.tar.bz2 | tar xvz
curl -# --silent http://aubio.org/pub/aubio-0.4.1.tar.bz2 | tar xvz
python music_transcriber.py examples/twinkle_short.wav
open examples/twinkle_short.pdf

