import sys
import wave
import os


class OnsetFrameSplitter(object):
    """
        A class for splitting a file into onset frames.
    """

    def __init__(self, music_file, output_directory):
        self.music_file = music_file
        self.output_directory = output_directory
        self.verbose = False

    def onset_frames_split(self):
        """
            Splits a music file into onset frames.
        """
        print 'Just about to execute object frames split function'
        onsets_output_file = "onsets.txt"
        #OD_METHOD = 'mkl'

        # Executing aubioonset command to get the onsets.
        os.system('aubioonset -i ' + self.music_file + ' --onset complex > ' + onsets_output_file)
        onsets = [float(x) for x in open(onsets_output_file).read().splitlines()]
        if self.verbose:
            print 'onsets: '
            for o in onsets:
                print o
        print 'Executed aubioonset function to split the file into onsets'

        # Reading in the music wave and getting parameters.
        input_music_wave = wave.open(self.music_file, "rb")
        nframes = input_music_wave.getnframes()
        params = input_music_wave.getparams()
        framerate = input_music_wave.getframerate()
        duration = nframes / float(framerate)

        if self.verbose:
            print "nframes: %d" % (nframes,)
            print "frame rate: %d " % (framerate,)
            print "duration: %f seconds" % (duration,)

        onsets.append(duration)
        onsets[0] = 0.0

        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        print 'Just about to split the file into onset frames'
        # Splitting the music file into onset frames.
        for i in range(len(onsets) - 1):
            frame = int(framerate * (onsets[i + 1] - onsets[i]))
            sound = input_music_wave.readframes(frame)
            music_wave = wave.open(self.output_directory + "/note%d.wav" % (i, ), "wb")
            music_wave.setparams(params)
            music_wave.setnframes(frame)
            music_wave.writeframes(sound)
            music_wave.close()
        print 'Split the file into onset frames'


if __name__ == '__main__':
    music_file = sys.argv[1]
    directory = 'frames'
    splitter = OnsetFrameSplitter(music_file, directory)
    splitter.onset_frames_split()
