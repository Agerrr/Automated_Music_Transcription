#!/bin/bash


#function to download lillypond based on OS 
function download { 
    #parameter should be url 
    url=$1
    curl -# --silent url | tar xvz
}
echo "getting OS information" 
OS=`uname -s`
echo "you are running on ${OS}"
echo "downloading lilypond" 

if [ "${OS}" == 'Darwin' ]
then
    download http://download.linuxaudio.org/lilypond/binaries/darwin-x86/lilypond-2.18.2-1.darwin-x86.tar.bz2
fi

if [ "${OS}" == 'Linux' ]
then
    download http://download.linuxaudio.org/lilypond/binaries/darwin-x86/lilypond-2.18.2-1.darwin-x86.tar.bz2
fi

if [ "${OS}" == 'Windows' ]
then
    download http://download.linuxaudio.org/lilypond/binaries/mingw/lilypond-2.18.2-1.mingw.exe
fi

#install and configure aubio
curl -# --silent http://aubio.org/pub/aubio-0.4.1.tar.bz2 | tar xvz
cd aubio* 
#instructions taken off of aubio website 
./waf configure build
sudo ./waf install 

#change back to next directory up
cd ..
python music_transcriber.py examples/twinkle_short.wav
open examples/twinkle_short.pdf

