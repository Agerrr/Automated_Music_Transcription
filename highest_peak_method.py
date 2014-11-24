import sys
import wave
import math
import scipy
import pylab
import scipy.io.wavfile as wav
import numpy


def getDuration(sound_file):
    """
        Returns the duration of a given sound file.
    """

    wr = wave.open(sound_file, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname =  wr.getparams()
    return nframes / float(framerate)


def getFrameRate(sound_file):
    wr = wave.open(sound_file, 'r')
    nchannels, sampwidth, framerate, nframes, comptype, compname = wr.getparams()
    return framerate


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


class Highest_Peaks_MIDI_Detector(object):
    """
        Class for MIDI notes detection given a .wav file.
    """

    def __init__(self, wav_file):
        self.wav_file = wav_file
        # before: 0.005e+13  twinkle: 0.002e+14 scale: 0.005e+16
        self.THRESHOLD = 0.005e+13
        self.HAN_WINDOW = 0.093
        self.HOP_SIZE = 0.00928
        self.minFreqConsidered = 27.0
        self.maxFreqConsidered = 2093

    def detect_MIDI_notes(self):
        """
            The algorithm for calculating midi notes from a given wav file.
        """

        (framerate, sample) = wav.read(self.wav_file)
        # We need to change the 2 channels into one because STFT works only
        # for 1 channel. We could also do STFT for each channel separately.
        monoChannel = sample.mean(axis=1)
        duration = getDuration(self.wav_file)
        midi_notes = []

        # Consider only files with a duration longer than 0.2 seconds.
        if duration > 0.18:
            frequency_power = self.calculateFFT(duration, framerate, monoChannel)
            filtered_frequencies = [f for (f, p) in frequency_power]
            #self.plot_power_spectrum(frequency_power)
            #self.plot_power_spectrum_dB(frequency_power)
            f0_candidates = self.get_pitch_candidates_remove_highest_peak(frequency_power)
            midi_notes = self.matchWithMIDINotes(f0_candidates)
        return midi_notes

    def get_pitch_candidates_remove_highest_peak(self, frequency_power):
        peak_frequencies = []
        while len(frequency_power) > 0:
            # sort the frequency_power by power (highest power first)
            sorted_frequency_power = sorted(frequency_power, key=lambda power: power[1], reverse=True)
            peak_frequency = sorted_frequency_power[0][0]
            peak_frequencies.append(peak_frequency)
            frequency_power = self.filterOutHarmonics(frequency_power, peak_frequency)
        return peak_frequencies

    def plot_power_spectrum(self, frequency_power):
        T = int(600)
        pylab.figure('Power spectrum')
        frequencies = [f[0] for f in frequency_power]
        powers = [p[1] for p in frequency_power]

        pylab.plot(frequencies[:T], powers[:T],)
        pylab.xlabel('Frequency [Hz]')
        pylab.ylabel('Power spectrum []')
        pylab.show()

    def plot_power_spectrum_dB(self, frequency_power):
        T = int(600)
        pylab.figure('Power spectrum')
        frequencies = [f[0] for f in frequency_power]
        powers = [p[1] for p in frequency_power]
        dBs = [10 * math.log10(power) if power > 0 else 0 for power in powers]

        pylab.plot(frequencies[:T], dBs[:T],)
        pylab.xlabel('Frequency [Hz]')
        pylab.ylabel('Power spectrum [dB]')
        pylab.show()

    def calculateFFT(self, duration, framerate, sample):
        """
            Calculates FFT for a given sound wave.
            Considers only frequencies with the magnitudes higher than
            a given threshold.
        """

        fft_length = int(duration * framerate)

        fft_length = get_next_power_2(fft_length)
        FFT = numpy.fft.fft(sample, n=fft_length)

        ''' ADJUSTING THRESHOLD '''
        threshold = 0
        power_spectra = []
        for i in range(len(FFT) / 2):
            power_spectrum = scipy.absolute(FFT[i]) * scipy.absolute(FFT[i])
            if power_spectrum > threshold:
                threshold = power_spectrum
            power_spectra.append(power_spectrum)
        threshold *= 0.1

        binResolution = float(framerate) / float(fft_length)
        frequency_power = []
        # For each bin calculate the corresponding frequency.
        for k in range(len(FFT) / 2):
            binFreq = k * binResolution

            if binFreq > self.minFreqConsidered and binFreq < self.maxFreqConsidered:
                power_spectrum = power_spectra[k]
                #dB = 10*math.log10(power_spectrum)
                if power_spectrum > threshold:
                    frequency_power.append((binFreq, power_spectrum))

        return frequency_power

    def filterOutHarmonics(self, frequency_power, f0_candidate):
        """
            Given frequency_power pairs and an f0 candidate remove
            all possible harmonics of this f0 candidate.
        """

        # If an integer frequency is a multiple of another frequency
        # then it is its harmonic. This constant was found empirically.
        # TODO: This constant may change for inharmonic frequencies!!!
        REMAINDER_THRESHOLD = 0.2

        def is_multiple(f, f0):
            return abs(round(f / f0) - f / f0) < REMAINDER_THRESHOLD

        return [(f, p) for (f, p) in frequency_power if not is_multiple(f, f0_candidate)]

    def matchWithMIDINotes(self, f0_candidates):
        midi_notes = []
        for freq in f0_candidates:
            #print 'FREQUENCY: ' + str(freq)
            midi_notes.append(int(round(69 + 12 * math.log(freq / 440) / math.log(2))))  # Formula for calculating MIDI note number.
        return midi_notes


if __name__ == '__main__':
    MIDI_detector = Highest_Peaks_MIDI_Detector(sys.argv[1])
    midi_notes = MIDI_detector.detect_MIDI_notes()
    print midi_notes
