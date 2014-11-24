import sys
import numpy
from threshold_finder import Threshold_Finder


class Average_Threshold_Finder(object):

    def get_average_noise_threshold(self, file_with_samples, no_of_samples):
        with open(file_with_samples) as f:
            samples = [line[:-1] for line in f]
        noise_spectra = []
        avg_noise_powers = []
        for i in range(0, int(no_of_samples), 4):
            chord = samples[i]
            first = samples[i + 1]
            second = samples[i + 2]
            third = samples[i + 3]
            t_finder = Threshold_Finder(chord, first, second, third)
            coefficients, residual, average_noise_power = t_finder.find_least_squares()
            noise = residual ** 2
            noise_spectra.append(noise)
            avg_noise_powers.append(average_noise_power)
        average_noise = numpy.mean(noise_spectra)
        sd_noise = numpy.std(noise_spectra)
        avg_power = numpy.mean(avg_noise_powers)
        sd_power = numpy.std(avg_noise_powers)
        return (average_noise, sd_noise, avg_power, sd_power)


if __name__ == '__main__':
    file_with_samples = sys.argv[1]
    no_of_samples = sys.argv[2]
    threshold_finder = Average_Threshold_Finder()
    average_noise, sd_noise, avg_power, sd_power = threshold_finder.get_average_noise_threshold(file_with_samples, no_of_samples)
    print average_noise
    print sd_noise
    print avg_power
    print sd_power
