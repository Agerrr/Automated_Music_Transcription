import sys, os, wave, math
import scipy, pylab, matplotlib
import scipy.io.wavfile as wav
import wave
from scipy import signal
from itertools import product
import numpy



''' Reads a sound wave from a standard input and finds its parameters. '''
def readWav():

    # Read the sound wave from the input.
    sound_wave = wave.open(sys.argv[1], "r")
    
    # Get parameters of the sound wave.
    nframes = sound_wave.getnframes()
    framerate = sound_wave.getframerate()
    params = sound_wave.getparams()
    duration = nframes / float(framerate)

    print "frame rate: %d " % (framerate,)
    print "nframes: %d" % (nframes,)
    print "duration: %f seconds" % (duration,)
    print scipy.array(sound_wave)

    return (sound_wave, nframes, framerate, duration, params)


''' Returns the duration of a given sound file. '''
def getDuration(sound_file):
    wr = wave.open(sound_file,'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname =  wr.getparams()
    return nframes / float(framerate)

''' Returns the frame rate of a given sound file. '''
def getFrameRate(sound_file):
    wr = wave.open(sound_file, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wr.getparams()
    return framerate

''' Returns number of channels of a given sound file. '''
def get_channels_no(sound_file):
    wr = wave.open(sound_file, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wr.getparams()
    return nchannels

''' Plots a given sound wave. '''
def plotSoundWave(rate, sample):
    t = scipy.linspace(0, 2, 2*rate, endpoint=False)
    pylab.figure('Sound wave')
    T = int(0.0001*rate)
    pylab.plot(t[:T], sample[:T],)
    pylab.show()


''' Calculates and plots the power spectrum of a given sound wave. '''
def plotPartials(binFrequencies, maxFreq, magnitudes):

    T = int(maxFreq)
    pylab.figure('Power spectrum')
    pylab.plot(binFrequencies[:T], magnitudes[:T],)
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Power spectrum (|X[k]|^2)')
    pylab.show()

''' Calculates and plots the power spectrum of a given sound wave. '''
def plotPowerSpectrum(FFT, binFrequencies, maxFreq):

    T = int(maxFreq)
    pylab.figure('Power spectrum')
    pylab.plot(binFrequencies[:T], scipy.absolute(FFT[:T])*scipy.absolute(FFT[:T]),)
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Power spectrum (|X[k]|^2)')
    pylab.show()


def get_frequencies_axis(framerate, fft_length):
    binResolution = float(framerate) / float(fft_length)
    binFreqs = []
    for k in range(fft_length):
        binFreq = k * binResolution
        binFreqs.append(binFreq)
    return binFreqs


''' Returns the closest number that is smaller than n that is a power of 2. '''
def get_next_power_2(n):
    power = 1
    while (power < n):
        power *= 2
    if power > 1:
        return power / 2
    else:
        return 1


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
'' Class for MIDI notes detection given a .wav file. ''
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
class MIDI_Detector(object):

    def __init__(self, wav_file):
        self.wav_file = wav_file
        self.minFreqConsidered = 20
        self.maxFreqConsidered = 5000
        self.low_f0s = [27.5, 29.135, 30.868, 32.703, 34.648, 37.708, 38.891, 41.203, 43.654, 46.249,
                        48.999, 51.913, 55.0, 58.27, 61.735, 65.406, 69.296, 73.416, 77.782, 82.407]


    ''' The algorithm for calculating midi notes from a given wav file. '''
    def detect_MIDI_notes(self):
        (framerate, sample) = wav.read(self.wav_file)
        if get_channels_no(self.wav_file) > 1:
            sample = sample.mean(axis=1)
        duration = getDuration(self.wav_file)
        midi_notes = []

        # Consider only files with a duration longer than 0.18 seconds.
        if duration > 0.18:   
            (FFT, filteredFreqs, maxFreq, magnitudes, significant_freq) = self.calculateFFT(duration, framerate, sample)
            #plotPowerSpectrum(FFT, filteredFreqs, 1000)
            clusters = self.clusterFrequencies(filteredFreqs)
            averagedClusters = self.getClustersMeans(clusters)
            f0_candidates = self.getF0Candidates(averagedClusters)
            midi_notes = self.matchWithMIDINotes(f0_candidates)

            '''
            OCTAVE CORRECTION METHOD
            '''
            '''

            # Include a note with a significant magnitude:
            # if its magnitude is higher than the sum of magnitudes of all other spectral peaks
            # include it in the list of detected notes and remove the note that's octave lower than this one
            # if it was also detected.
            if significant_freq > 0:
                significant_midi_notes = self.matchWithMIDINotes([significant_freq])
                significant_midi_note = significant_midi_notes[0]
                if significant_midi_note not in midi_notes:
                    midi_notes.append(significant_midi_note)
                    midi_notes = self.remove_lower_octave(significant_midi_note, midi_notes)
            '''
            
        return midi_notes


    def remove_lower_octave(self, upper_octave, midi_notes):
        lower_octave = upper_octave - 12
        if lower_octave in midi_notes:
            midi_notes.remove(lower_octave)
        return midi_notes


    def get_candidates_with_partials(self, frequencies, magnitudes):
        print frequencies
        partial_margin = 11.0                   # Hz
        candidates_freq = []                    # A list of frequencies of each candidate.
        candidates_magnitude = []               # A list of magnitudes of frequencies of each candidate.    
        for i in range(len(frequencies)):
            (partials, partial_magnitudes) = self.find_partials(frequencies[i:], frequencies[i], magnitudes[i:])
            candidates_freq.append(partials)
            candidates_magnitude.append(partial_magnitudes)
        return (candidates_freq, candidates_magnitude)



    ''' Calculates FFT for a given sound wave. 
        Considers only frequencies with the magnitudes higher than
        a given threshold. '''
    def calculateFFT(self, duration, framerate, sample):
        fft_length = int(duration * framerate)
        fft_length = get_next_power_2(fft_length)   # For the FFT to work much faster take the length that is a power of 2.
        FFT = numpy.fft.fft(sample, n=fft_length)


        ''' ADJUSTING THRESHOLD - HIGHEST SPECTRAL PEAK METHOD'''
        threshold = 0
        power_spectra = []
        frequency_bin_with_max_spectrum = 0
        for i in range(len(FFT)/2):
            power_spectrum = scipy.absolute(FFT[i])*scipy.absolute(FFT[i])
            if power_spectrum > threshold:
                threshold = power_spectrum
                frequency_bin_with_max_spectrum = i 
            power_spectra.append(power_spectrum)
        max_power_spectrum = threshold
        threshold *= 0.1
        

        binFrequencies = []
        magnitudes = []
        binResolution = float(framerate) / float(fft_length)
        sum_of_significant_spectra = 0
        # For each bin calculate the corresponding frequency.
        for k in range(len(FFT)):
            binFreq = k * binResolution
            
            # Truncating the FFT so we consider only hearable frequencies.
            if binFreq > self.maxFreqConsidered:
                FFT = FFT[:k]
                break
            elif binFreq > self.minFreqConsidered:
                # Consider only the frequencies with magnitudes higher than the threshold.
                power_spectrum = power_spectra[k]
                if power_spectrum > threshold:
                    magnitudes.append(power_spectrum)
                    binFrequencies.append(binFreq)
                    
                    # Sum all significant power spectra except the max power spectrum.
                    if power_spectrum != max_power_spectrum:
                        sum_of_significant_spectra += power_spectrum
            
        significant_freq = 0.0
        
        if max_power_spectrum > sum_of_significant_spectra:
            significant_freq = frequency_bin_with_max_spectrum * binResolution
        
        maxFreq = len(FFT) / duration  # Max. frequency considered after truncating.
                                       # maxFreq = rate without truncating.
        return (FFT, binFrequencies, maxFreq, magnitudes, significant_freq)


    ''' Computes STFT for a given sound wave using Hanning window. '''
    # Code for STFT taken from:
    # http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    def STFT(self, x, samplingFreq, framesz, hop):
        framesamp = int(framesz*samplingFreq)
        print 'FRAMESAMP: ' + str(framesamp)
        hopsamp = int(hop*samplingFreq)
        print 'HOP SAMP: ' + str(hopsamp)
        # Modification: using Hanning window instead of Hamming - by Pertusa
        w = signal.hann(framesamp)
        X = numpy.array([numpy.fft.fft(w*x[i:i+framesamp]) 
                         for i in range(0, len(x)-framesamp, hopsamp)])
        return X


    ''' Calculates and plots the magnitude spectrum of a given sound wave. '''
    def plotMagnitudeSpectrogram(self, rate, sample, framesz, hop):
        X = self.STFT(sample, rate, framesz, hop)

        # Plot the magnitude spectrogram.
        pylab.figure('Magnitude spectrogram')
        pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto',
                     interpolation='nearest')
        pylab.xlabel('Time')
        pylab.ylabel('Frequency')
        pylab.show()


    ''' Returns a list of frequencies with the magnitudes higher than a given threshold. '''
    def getFilteredFFT(self, FFT, duration, threshold):
        significantFreqs = []
        for i in range(len(FFT)):
            power_spectrum = scipy.absolute(FFT[i])*scipy.absolute(FFT[i])
            if power_spectrum > threshold:
                significantFreqs.append(i / duration)

        return significantFreqs


    ''' Clusters frequencies. '''
    def clusterFrequencies(self, freqs):
        if len(freqs) == 0:
            return {}
        clusteredFreqs = {}
        bin = 0
        clusteredFreqs[0] = [freqs[0]]
        for i in range(len(freqs)-1):
            dist = self.calcDistance(freqs[i], freqs[i+1])
            if dist < 2.0:
                clusteredFreqs[bin].append(freqs[i+1])
            else:
                bin += 1
                clusteredFreqs[bin] = [freqs[i+1]]

        return clusteredFreqs


    ''' Given clustered frequencies finds a mean of each cluster. '''
    def getClustersMeans(self, clusters):
        means = []
        for bin, freqs in clusters.iteritems():
            means.append(sum(freqs)/len(freqs))
        return means


    ''' Returns a list of distances between each frequency. '''
    def getDistances(self, freqs):
        distances =  { (freqs[i], freqs[j]): self.calcDistance(freqs[i], freqs[j])
                       for (i,j) in product(range(len(freqs)), repeat=2) }
        distances = { freq_pair: dist for freq_pair, dist in distances.iteritems() if dist < 2.0 }
        return distances


    ''' Calculates distance between frequencies taking into account that
        the frequencies of pitches increase logarithmically. '''
    def calcDistance(self, freq1, freq2):
        difference = abs(freq1 - freq2)
        log = math.log((freq1 + freq2) / 2)
        return difference / log


    ''' Given frequencies finds possible F0 candidates
        by discarding potential harmonic frequencies. '''
    def getF0Candidates(self, frequencies):
        f0_candidates = []
        
        '''
        MODIFICATION: CONSIDER ONLY MIDDLE RANGE FREQUENCIES
        '''
        '''

        if len(frequencies) > 0 and frequencies[0] < 83.0:
            low_freq_candidate = self.find_low_freq_candidate(frequencies)
            if low_freq_candidate > 0.0:
                f0_candidates.append(low_freq_candidate)
                #frequencies = self.filterOutHarmonics(frequencies, low_freq_candidate)
        '''
        
        while len(frequencies) > 0:
            f0_candidate = frequencies[0]
            f0_candidates.append(f0_candidate)
            frequencies.remove(f0_candidate)
            frequencies = self.filterOutHarmonics(frequencies, f0_candidate)
        return f0_candidates


    ''' Given frequencies and an f0 candidate remove
        all possible harmonics of this f0 candidate. '''
    def filterOutHarmonics(self, frequencies, f0_candidate):
        REMAINDER_THRESHOLD = 0.2   # If an integer frequency is a multiple of another frequency
                                    # then it is its harmonic. This constant was found empirically.

        def is_multiple(f, f0):
            return abs(round(f/f0) - f/f0) < REMAINDER_THRESHOLD

        return [ f for f in frequencies if not is_multiple(f, f0_candidate)]


    def find_low_freq_candidate(self, frequencies):
        REMAINDER_THRESHOLD = 0.05
        f0_candidates = []

        def is_multiple(f, f0):
            return abs(round(f/f0) - f/f0) < REMAINDER_THRESHOLD

        best_candidate = -1
        max_no_partials = 0
        for low_f0 in self.low_f0s:
            num_of_partials = 0
            for f in frequencies:
                if is_multiple(f, low_f0):
                    num_of_partials += 1
            if num_of_partials > max_no_partials:
                max_no_partials = num_of_partials
                best_candidate = low_f0
        return best_candidate


    ''' Given frequencies, frequency magnitudes and an f0 candidate
        return the partials and magnitudes of this f0 candidate. '''
    def find_partials(self, frequencies, f0_candidate, magnitudes):
        REMAINDER_THRESHOLD = 0.05

        def is_multiple(f, f0):
            return abs(round(f/f0) - f/f0) < REMAINDER_THRESHOLD

        partials = []
        partial_magnitudes = []
        for i in range(len(frequencies)):
            if is_multiple(frequencies[i], f0_candidate):
                partials.append(frequencies[i])
                partial_magnitudes.append(magnitudes[i])
        return (partials, partial_magnitudes)


    def matchWithMIDINotes(self, f0_candidates):
        midi_notes = []
        for freq in f0_candidates:
            midi_notes.append(int(round(69 + 12*math.log(freq/440)/math.log(2))))  # Formula for calculating MIDI note number.
        return midi_notes


if __name__ == '__main__':
    MIDI_detector = MIDI_Detector(sys.argv[1])
    midi_notes = MIDI_detector.detect_MIDI_notes()
    print midi_notes