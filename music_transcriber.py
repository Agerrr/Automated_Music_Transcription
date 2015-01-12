import sys
from onset_frames_split import OnsetFrameSplitter
#from first_peaks_method import MIDI_Detector
from plotNotes import NotePlotter
#from highest_peak_method import Highest_Peaks_MIDI_Detector


class MusicTranscriber(object):
    """
        The class responsible for transcibing music stored in a .wav file
        to pdf sheet notes.
    """

    def __init__(self, music_file):
        self.music_file = music_file
        self.onset_frames_dir = 'frames'

    def transcribe(self):
        """
            Splits the music file to be transcribed into onset frames,
            detects the notes in each frame and plots them on the staff.
        """

        splitter = OnsetFrameSplitter(self.music_file, self.onset_frames_dir)
        print 'Created onset frame splitter object'
        splitter.onset_frames_split()
        print 'Splitted the file into frames'
        note_plotter = NotePlotter(self.music_file)
        print 'Created a note plotter object'
        note_plotter.plot_multiple_notes(self.onset_frames_dir)
        print 'Plotted multiple notes'


if __name__ == '__main__':
    # Provide the name of the music file (in wav. format) to be transcribed.
    music_file = sys.argv[1]
    print 'Read in a music file'
    transcriber = MusicTranscriber(music_file)
    print 'Created a transcriber object'
    transcriber.transcribe()
    print 'Transcribed the music piece'
