import sys, os
from onset_frames_split import OnsetFrameSplitter
#from first_peaks_method import MIDI_Detector
from plotNotes import NotePlotter
#from highest_peak_method import Highest_Peaks_MIDI_Detector


''' The class responsible for transcibing music stored in a wav. file to pdf sheet notes. '''
class MusicTranscriber(object):

	def __init__(self, music_file):
		self.music_file = music_file
		self.onset_frames_dir = 'frames'

	''' Splits the music file to be transcribed into onset frames, detects the notes in each frame
	    and plots them on the staff. '''
	def transcribe(self):
		splitter = OnsetFrameSplitter(self.music_file, self.onset_frames_dir)
		splitter.onset_frames_split()
		note_plotter = NotePlotter(self.music_file)
		note_plotter.plot_multiple_notes(self.onset_frames_dir)
			

if __name__ == '__main__':
	music_file = sys.argv[1]	# The name of the music file (in wav. format) to be transcribed.
	transcriber = MusicTranscriber(music_file)
	transcriber.transcribe()
