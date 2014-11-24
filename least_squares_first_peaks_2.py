import sys
import math
import scipy
import pylab
import scipy.io.wavfile as wav
import wave
from scipy import signal
from itertools import product
import numpy


def readWav():
    """
        Reads a sound wave from a standard input and finds its parameters.
    """

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


def getDuration(sound_file):
    """
        Returns the duration of a given sound file.
    """

    wr = wave.open(sound_file,'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname =  wr.getparams()
    return nframes / float(framerate)


def getFrameRate(sound_file):
    wr = wave.open(sound_file, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wr.getparams()
    return framerate


def plotSoundWave(rate, sample):
    """
        Plots a given sound wave.
    """

    t = scipy.linspace(0, 2, 2 * rate, endpoint=False)
    pylab.figure('Sound wave')
    T = int(0.0001 * rate)
    pylab.plot(t[:T], sample[:T],)
    pylab.show()


def plotPartials(binFrequencies, maxFreq, magnitudes):
    """
        Calculates and plots the power spectrum of a given sound wave.
    """

    T = int(maxFreq)
    pylab.figure('Power spectrum')
    pylab.plot(binFrequencies[:T], magnitudes[:T],)
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Power spectrum (|X[k]|^2)')
    pylab.show()


def plotPowerSpectrum(FFT, binFrequencies, maxFreq):
    """
        Calculates and plots the power spectrum of a given sound wave.
    """

    T = int(maxFreq)
    pylab.figure('Power spectrum')
    pylab.plot(binFrequencies[:T], scipy.absolute(FFT[:T]) * scipy.absolute(FFT[:T]),)
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


def is_Prime(n):
    """
        Check if a number is prime.
    """

    # make sure n is a positive integer
    n = abs(int(n))
    # 0 and 1 are not primes
    if n < 2:
        return False
    # 2 is the only even prime number
    if n == 2:
        return True
    # all other even numbers are not primes
    if not n & 1:
        return False
    # range starts with 3 and only needs to go up the squareroot of n
    # for all odd numbers
    for x in range(3, int(n ** 0.5) + 1, 2):
        if n % x == 0:
            return False
    return True


def get_next_power_2(n):
    """
        Returns the closest number that is smaller than n that is a power of 2.
    """

    power = 1
    while (power < n):
        power *= 2
    if power > 1:
        return power / 2
    else:
        return 1


class MIDI_Detector_Least_Squares_2(object):
    """
        Class for MIDI notes detection given a .wav file.
    """

    # Constants:
    #THRESHOLD = 0.005e+13    # Empirically found magnitude threshold.
                             # Only frequencies whose magnitude is higher than
                             # this threshold are going to be considered.
    HAN_WINDOW = 0.093       # 93 ms Hanning window size by Pertusa.
    HOP_SIZE = 0.00928       # 9.28 ms hop size by Pertusa.
    
    AkPnBcht_noise      = 4.5883573833e+15 
    AkPnBcht_std        = 3.60397998387e+15
    AkPnBcht_noise_2    = 3.42198774903e+15
    AkPnBcht_std_2      = 2.65280448383e+15
    AkPnBcht_std_10     = 1.03972925552e+15     # From 10 samples
    AkPnBcht_std__10    = 1.69916357929e+14     # From 10 samples

    AkPnBsdf_noise      = 2.91911620721e+15
    AkPnBsdf_std        = 1.69388970617e+15
    AkPnBsdf_noise_2    = 1.41729560313e+13     # From 10 samples
    AkPnBsdf_std_2      = 6.54120706376e+14     # From 10 samples

    AkPnCGdD_noise      = 2.74224876671e+15
    AkPnCGdD_std        = 3.28471607186e+15
    AkPnCGdD_noise_2    = 9.07842836729e+14     # From 10 samples
    AkPnCGdD_std_2      = 2.10397850449e+14     # From 10 samples

    AkPnStgb_noise_10   = 1.20237557768e+15     # From 10 samples
    AkPnStgb_std_10     = 2.3459546709e+14      # From 10 samples

    ENSTDkAm_noise_10   = 1.2158350262e+16      # From 10 samples
    ENSTDkAm_std_10     = 8.43175297998e+15     # From 10 samples

    ENSTDkCl_noise_10   = 7.01687672799e+15     # From 10 samples
    ENSTDkCl_std_10     = 3.72640836838e+15     # From 10 samples

    SptkBGAm_noise_10   = 6.6553382974e+15
    SptkBGAm_std_10     = 4.38008203527e+15

    SptkBGCl_noise_10   = 2.99160007806e+15
    SptkBGCl_std_10     = 1.38230425716e+15

    StbgTGd2_noise_10   = 1.22960679454e+15
    StbgTGd2_std_10     = 8.05049808568e+14

    def __init__(self, wav_file):
        self.wav_file = wav_file
        self.THRESHOLD = 0.005e+13  # before: 0.005e+13  twinkle: 0.002e+14 scale: 0.005e+16
        self.HAN_WINDOW = 0.093
        self.HOP_SIZE = 0.00928
        self.minFreqConsidered = 20
        self.maxFreqConsidered = 5000
        self.low_f0s = [27.5, 29.135, 30.868, 32.703, 34.648, 37.708, 38.891,
                        41.203, 43.654, 46.249, 48.999, 51.913, 55.0, 58.27,
                        61.735, 65.406, 69.296, 73.416, 77.782, 82.407]
        #self.noise_threshold = 1.22960679454e+15
        #self.noise_std =  8.05049808568e+14
        self.avg_noise_power = 9.47594450397e+16
        self.std_noise_power = 856838153209.0

    def detect_MIDI_notes(self):
        """
            The algorithm for calculating midi notes from a given wav file.
        """

        (framerate, sample) = wav.read(self.wav_file)
        
        # We need to change the 2 channels into one because STFT works only
        # for 1 channel. We could also do STFT for each channel separately.
        #monoChannel = sample.mean(axis=1)
        
        duration = getDuration(self.wav_file)
        midi_notes = []

        # Consider only files with a duration longer than 0.2 seconds.
        if duration > 0.18:
            #print 'Calculating FFT...'

            (FFT, filteredFreqs, maxFreq, magnitudes, significant_freq) = self.calculateFFT(duration, framerate, sample)
            #plotPartials(filteredFreqs, maxFreq, magnitudes)
            #print FFT
            #print filteredFreqs
            #print filteredFreqs
            #plotPowerSpectrum(FFT, get_frequencies_axis(framerate, int(duration * framerate)), maxFreq)

            #plotPowerSpectrum(FFT, filteredFreqs, 1000)
            #print filteredFreqs
            #print 'Clustering frequencies...'
            clusters = self.clusterFrequencies(filteredFreqs)
            #print clusters
            #print 'Getting clusters means...'
            averagedClusters = self.getClustersMeans(clusters)
            #print averagedClusters
            #print averagedClusters
            #print 'Getting F0 candidates...'
            #print averagedClusters
            f0_candidates = self.getF0Candidates(averagedClusters)
            #print f0_candidates
            #print f0_candidates
            #print 'Matching with MIDI notes...'
            midi_notes = self.matchWithMIDINotes(f0_candidates)

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

            #print 'Successfully detected MIDI notes.'
            '''
            for i in range(len(midi_notes)):
                midi_notes[i] = midi_notes[i]+1
            '''
        return midi_notes

    def remove_lower_octave(self, upper_octave, midi_notes):
        lower_octave = upper_octave - 12
        if lower_octave in midi_notes:
            midi_notes.remove(lower_octave)
        return midi_notes

    def Pertusa_Inesta_Algorithm(self):
        (framerate, sample) = wav.read(self.wav_file)
        monoChannel = sample.mean(axis=1)
        stft = self.STFT(monoChannel, framerate, self.HAN_WINDOW, self.HOP_SIZE)
        (stft_bin_freqs, stft_magnitudes) = self.get_stft_bin_freqs(stft, framerate)
        print stft_bin_freqs
        #print self.get_candidates_with_partials(stft_bin_freqs[0], stft_magnitudes[0])

        '''
        midi_notes = []
        for freqs in stftBinFrequencies:
            midi_notes.append(self.matchWithMIDINotes(sorted(freqs.keys())))
        print midi_notes[0]
        return midi_notes
        '''

    ''' Given STFT returns a list of dictionaries of bin frequencies and corresponding amplitudes.
        Each list is a list of bin frequencies for a corresponding FFT in STFT.
        The frequencies are filtered based on their magnitude. '''
        # TODO: Consider zero padding to get more precise estimation for lower frequencies.
    def get_stft_bin_freqs(self, stft, framerate):
        fft_length = self.HAN_WINDOW * framerate
        binResolution = float(framerate) / float(fft_length)
        stft_binFrequencies = []
        stft_magnitudes = []
        for i in range(len(stft)):
            binFreqs = []
            magnitudes = []
            for k in range(len(stft[i])):
                binFreq = k * binResolution
                if binFreq > self.minFreqConsidered and binFreq < self.maxFreqConsidered:
                    power_spectrum = scipy.absolute(stft[i][k]) * scipy.absolute(stft[i][k])
                    if power_spectrum > self.THRESHOLD:
                        binFreqs.append(binFreq)
                        magnitudes.append(power_spectrum)
                    stft_binFrequencies.append(binFreqs)
                    stft_magnitudes.append(magnitudes)
        return (stft_binFrequencies, stft_magnitudes)

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

    def calculateFFT(self, duration, framerate, sample):
        """
            Calculates FFT for a given sound wave. 
            Considers only frequencies with the magnitudes higher than
            a given threshold.
        """

        fft_length = int(duration * framerate) # 216090
        #fft_length = int(2**17)
        
        '''
        # numpy.fft is very slow if fft length is a prime number.
        # TODO: Fix this problem.
        while is_Prime(fft_length):
            print 'PRIME!!!'
            fft_length -= 1
        FFT = numpy.fft.fft(sample, n=fft_length)
        '''
        fft_length = get_next_power_2(fft_length)
        FFT = numpy.fft.fft(sample, n=fft_length)

        ''' ADJUSTING THRESHOLD '''
        
        threshold = 0
        power_spectra = []
        frequency_bin_with_max_spectrum = 0
        for i in range(len(FFT) / 2):
            power_spectrum = scipy.absolute(FFT[i]) * scipy.absolute(FFT[i])
            if power_spectrum > threshold:
                threshold = power_spectrum
                frequency_bin_with_max_spectrum = i
            power_spectra.append(power_spectrum)
        max_power_spectrum = threshold
        threshold = float(self.avg_noise_power) - 3 * float(self.std_noise_power)

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
            
            #binFrequencies.append(binFreq)
        significant_freq = 0.0
        
        if max_power_spectrum > sum_of_significant_spectra:
            significant_freq = frequency_bin_with_max_spectrum * binResolution
        
        maxFreq = len(FFT) / duration  # Max. frequency considered after truncating.
                                       # maxFreq = rate without truncating.
        return (FFT, binFrequencies, maxFreq, magnitudes, significant_freq)

    # Code for STFT taken from:
    # http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    def STFT(self, x, samplingFreq, framesz, hop):
        """
            Computes STFT for a given sound wave using Hanning window.
        """

        framesamp = int(framesz * samplingFreq)
        print 'FRAMESAMP: ' + str(framesamp)
        hopsamp = int(hop * samplingFreq)
        print 'HOP SAMP: ' + str(hopsamp)
        # Modification: using Hanning window instead of Hamming - by Pertusa
        w = signal.hann(framesamp)
        X = numpy.array([numpy.fft.fft(w * x[i:i + framesamp])
                         for i in range(0, len(x) - framesamp, hopsamp)])
        return X

    def plotMagnitudeSpectrogram(self, rate, sample, framesz, hop):
        """
            Calculates and plots the magnitude spectrum of a given sound wave.
        """

        X = self.STFT(sample, rate, framesz, hop)

        # Plot the magnitude spectrogram.
        pylab.figure('Magnitude spectrogram')
        pylab.imshow(scipy.absolute(X.T), origin='lower', aspect='auto',
                     interpolation='nearest')
        pylab.xlabel('Time')
        pylab.ylabel('Frequency')
        pylab.show()

    def getFilteredFFT(self, FFT, duration, threshold):
        """
            Returns a list of frequencies with the magnitudes higher than a given threshold.
        """

        significantFreqs = []
        for i in range(len(FFT)):
            power_spectrum = scipy.absolute(FFT[i]) * scipy.absolute(FFT[i])
            if power_spectrum > threshold:
                significantFreqs.append(i / duration)

        return significantFreqs

    def clusterFrequencies(self, freqs):
        """
            Clusters frequencies.
        """

        if len(freqs) == 0:
            return {}
        clusteredFreqs = {}
        bin = 0
        clusteredFreqs[0] = [freqs[0]]
        for i in range(len(freqs) - 1):
            dist = self.calcDistance(freqs[i], freqs[i + 1])
            if dist < 2.0:
                clusteredFreqs[bin].append(freqs[i + 1])
            else:
                bin += 1
                clusteredFreqs[bin] = [freqs[i + 1]]

        return clusteredFreqs

    def getClustersMeans(self, clusters):
        """
            Given clustered frequencies finds a mean of each cluster.
        """

        means = []
        for bin, freqs in clusters.iteritems():
            means.append(sum(freqs)/len(freqs))
        return means

    def getDistances(self, freqs):
        """
            Returns a list of distances between each frequency.
        """

        distances =  {(freqs[i], freqs[j]): self.calcDistance(freqs[i], freqs[j])
                       for (i, j) in product(range(len(freqs)), repeat=2)}
        distances = {freq_pair: dist for freq_pair, dist in distances.iteritems() if dist < 2.0}
        return distances
    
    def calcDistance(self, freq1, freq2):
        """
            Calculates distance between frequencies taking into account that
            the frequencies of pitches increase logarithmically.
        """

        difference = abs(freq1 - freq2)
        log = math.log((freq1 + freq2) / 2)
        if log < 0.00001:               # TODO: Decide what to do if log is 0
            return freq1 + freq2
        return difference / log

    def getF0Candidates(self, frequencies):
        """
            Given frequencies finds possible F0 candidates
            by discarding potential harmonic frequencies.
        """

        f0_candidates = []
        
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

    def filterOutHarmonics(self, frequencies, f0_candidate):
        """
            Given frequencies and an f0 candidate remove
            all possible harmonics of this f0 candidate.
        """

        REMAINDER_THRESHOLD = 0.2   # If an integer frequency is a multiple of another frequency
                                    # then it is its harmonic. This constant was found empirically.
                                    # TODO: This constant may change for inharmonic frequencies!!!

        def is_multiple(f, f0):
            return abs(round(f / f0) - f / f0) < REMAINDER_THRESHOLD

        return [f for f in frequencies if not is_multiple(f, f0_candidate)]

    def find_low_freq_candidate(self, frequencies):
        REMAINDER_THRESHOLD = 0.05
        f0_candidates = []

        def is_multiple(f, f0):
            return abs(round(f / f0) - f / f0) < REMAINDER_THRESHOLD

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

    def find_partials(self, frequencies, f0_candidate, magnitudes):
        """
            Given frequencies, frequency magnitudes and an f0 candidate
            return the partials and magnitudes of this f0 candidate.
        """

        REMAINDER_THRESHOLD = 0.05

        def is_multiple(f, f0):
            return abs(round(f / f0) - f / f0) < REMAINDER_THRESHOLD

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
            #print 'FREQUENCY: ' + str(freq)
            midi_notes.append(int(round(69 + 12 * math.log(freq / 440) /math.log(2))))  # Formula for calculating MIDI note number.
        return midi_notes


if __name__ == '__main__':
    MIDI_detector = MIDI_Detector(sys.argv[1])
    midi_notes = MIDI_detector.detect_MIDI_notes()
    print midi_notes
    #pertusa_inesta_midi_notes = MIDI_detector.Pertusa_Inesta_Algorithm()
    #print pertusa_inesta_midi_notes

    #plotPowerSpectrum(FFT, binFreqs, maxFreq)

    #plotMagnitudeSpectrogram(rate, monoChannel, HAN_WINDOW, HOP_SIZE)
