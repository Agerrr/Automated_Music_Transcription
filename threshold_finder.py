from audio_reader import Audio_Reader
import sys
import numpy
import scipy
import pylab


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


class Threshold_Finder(object):

    def __init__(self, chord, first_single, second_single, third_single):
        self.audio_reader = Audio_Reader()
        self.chord = self.audio_reader.read_sample(chord)
        self.first_single = self.audio_reader.read_sample(first_single)
        self.second_single = self.audio_reader.read_sample(second_single)
        self.third_single = self.audio_reader.read_sample(third_single)

        self.chord_file = chord
        self.first_single_file = first_single
        self.second_single_file = second_single
        self.third_single_file = third_single

    def get_FFT(self, sample):
        # TODO: check if the fft_lengths are the same and if not decide what to do with it...
        fft_length = int(self.audio_reader.get_duration(self.chord_file) * self.audio_reader.get_framerate(self.chord_file))

        if is_Prime(fft_length):
            FFT = numpy.fft.fft(sample, n=fft_length-1)
        else:
            FFT = numpy.fft.fft(sample, n=fft_length)
        return FFT

    def find_least_squares(self):
        
        fft_chord = self.get_FFT(self.chord)
        fft_first = self.get_FFT(self.first_single)
        fft_second = self.get_FFT(self.second_single)
        fft_third = self.get_FFT(self.third_single)
        
        chord_array = []
        first_array = []
        second_array = []
        third_array = []
        for i in range(len(fft_chord)):
            chord_array.append(fft_chord[i].real)
            chord_array.append(fft_chord[i].imag)
            first_array.append(fft_first[i].real)
            first_array.append(fft_first[i].imag)
            second_array.append(fft_second[i].real)
            second_array.append(fft_second[i].imag)
            third_array.append(fft_third[i].real)
            third_array.append(fft_third[i].imag)

        A = numpy.array([first_array, second_array, third_array]).T
        b = chord_array
        
        # x: coefficients for each transform for single notes, rnorm: sum of squares of magnitudes of the noise.
        x, rnorm = scipy.optimize.nnls(A, b)
        average_noise_power = rnorm ** 2 / len(fft_chord)
        return (x, rnorm, average_noise_power)

    def get_FFT_of_noise(self, x, rnorm):
        sum_of_singles = x[0] * self.get_FFT(self.first_single) + x[1] * self.get_FFT(self.second_single) + x[2] * self.get_FFT(self.third_single)
        fft = scipy.absolute(self.get_FFT(self.chord) - sum_of_singles)
        return fft

    def get_sum_of_squares(self, fft):
        sum_of_squares = 0.0
        for i in range(len(fft)):
            sum_of_squares += scipy.absolute(fft[i]) * sscipy.absolute(fft[i])
        return sum_of_squares

    def plot_power_spectrum(self, fft):
        T = int(600)

        pylab.figure('Power spectrum')
        pylab.plot(scipy.absolute(fft[:T]) * scipy.absolute(fft[:T]),)
        pylab.xlabel('Frequency [Hz]')
        pylab.ylabel('Power spectrum []')
        pylab.show()


if __name__ == '__main__':
    threshold_finder = Threshold_Finder(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    x, rnorm, average_magnitude = threshold_finder.find_least_squares()
    print 'rnorm squared: ' + str(rnorm * rnorm)
    print 'rnorm: ' + str(rnorm)
    print 'average power: ' + str(average_magnitude ** 2)

    fft = threshold_finder.get_FFT_of_noise(x, rnorm)
    chord_fft = threshold_finder.get_FFT(threshold_finder.chord)
    first_fft = threshold_finder.get_FFT(threshold_finder.first_single)
    #threshold_finder.plot_power_spectrum(chord_fft)
    #threshold_finder.plot_power_spectrum(first_fft)
    #threshold_finder.plot_power_spectrum(fft)
    print threshold_finder.get_sum_of_squares(fft)
