from first_peaks_method import MIDI_Detector
#from least_squares_method import Highest_Peaks_MIDI_Detector
import sys, os
from subprocess import call
#from least_squares_first_peaks_2 import MIDI_Detector_Least_Squares_2
#from least_squares_highest_peaks_2 import Highest_Peaks_Least_Squares


''' Class used for plotting sheet notes given MIDI note numbers. '''
class NotePlotter(object):
	def __init__(self, wav_file):
		self.wav_file = wav_file
		self.output_file = wav_file[:-3] + 'ly'
		self.number2note = {
			21: 'a,,,',
			22: 'ais,,,',
			23: 'b,,,',
			24: 'c,,',
			25: 'cis,,',
			26: 'd,,',
			27: 'dis,,',
			28: 'e,,',
			29: 'f,,',
			30: 'fis,,',
			31: 'g,,',
			32: 'gis,,',
			33: 'a,,',
			34: 'ais,,',
			35: 'b,,',
			36: 'c,',
			37: 'cis,',
			38: 'd,',
			39: 'dis,',
			40: 'e,',
			41: 'f,',
			42: 'fis,',
			43: 'g,',
			44: 'gis,',
			45: 'a,',
			46: 'ais,',
			47: 'b,',
			48: 'c',
			49: 'cis',
			50: 'd',
			51: 'dis',
			52: 'e',
			53: 'f',
			54: 'fis',
			55: 'g',
			56: 'gis',
			57: 'a',
			58: 'ais',
			59: 'b',
			60: 'c\'',
			61: 'cis\'',
			62: 'd\'',
			63: 'dis\'',
			64: 'e\'',
			65: 'f\'',
			66: 'fis\'',
			67: 'g\'',
			68: 'gis\'',
			69: 'a\'',
			70: 'ais\'',
			71: 'b\'',
			72: 'c\'\'',
			73: 'cis\'\'',
			74: 'd\'\'',
			75: 'dis\'\'',
			76: 'e\'\'',
			77: 'f\'\'',
			78: 'fis\'\'',
			79: 'g\'\'',
			80: 'gis\'\'',
			81: 'a\'\'',
			82: 'ais\'\'',
			83: 'b\'\'',
			84: 'c\'\'\'',
			85: 'cis\'\'\'',
			86: 'd\'\'\'',
			87: 'dis\'\'\'',
			88: 'e\'\'\'',
			89: 'f\'\'\'',
			90: 'fis\'\'\'',
			91: 'g\'\'\'',
			92: 'gis\'\'\'',
			93: 'a\'\'\'',
			94: 'ais\'\'\'',
			95: 'b\'\'\'',
			96: 'c\'\'\'\'',
			97: 'cis\'\'\'',
			98: 'd\'\'\'',
			99: 'dis\'\'\'',
			100: 'e\'\'\'',
			101: 'f\'\'\'',
			102: 'fis\'\'\'',
			103: 'g\'\'\'',
			104: 'gis\'\'\'',
			105: 'a\'\'\'',
			106: 'ais\'\'\'',
			107: 'b\'\'\'',
			108: 'c\'\'\'\'',
		}


	''' Given .wav file detect MIDI notes, convert them into corresponding character names.
	   	Afterwards plot and save into an output file. 
	    The class uses lilypond library for drawing sheet notes. '''
	def plot_notes_violin_stuff(self):
		#detector = MIDI_Detector(self.wav_file)
		detector = Highest_Peaks_MIDI_Detector(self.wav_file)
		midi_numbers = detector.detect_MIDI_notes()
		lilypond_text = '\\version \"2.14.2\" \n{ \n  \\clef treble \n'
		for n in midi_numbers:
			if n in self.number2note.keys():
				lilypond_text += self.number2note[n] + ' '
		lilypond_text += '\n}'
		with open(self.output_file, 'w') as f:
			f.write(lilypond_text)
		command = "lilypond "
		command += self.output_file
		print command
		os.system(command)


	def plot_notes(self):
		detector = MIDI_Detector(self.wav_file)
		#detector = Highest_Peaks_MIDI_Detector(self.wav_file)
		midi_numbers = detector.detect_MIDI_notes()
		lilypond_text = '\\version \"2.14.2\" \n'
		lilypond_text += '  \\new PianoStaff { \n'
		lilypond_text += '    \\autochange { \n <'
		for n in midi_numbers:
			if n in self.number2note.keys():
				lilypond_text += self.number2note[n] + ' '
		lilypond_text += '>    \n}  \n}'
		with open(self.output_file, 'w') as f:
			f.write(lilypond_text)
		command = "lilypond "
		command += self.output_file
		print command
		os.system(command)


	''' Plots notes using LilyPond library. The notes are on a left and right hand staff (piano)
		and may be plotted as chords (multiple notes played simultaneously). The generated sheet notes
		are named after the music file. '''
	def plot_multiple_notes(self, directory):
		lilypond_text = '\\version \"2.14.2\" \n'
		lilypond_text += '  \\new PianoStaff { \n'
		lilypond_text += '    \\autochange { \n'

		numOfFiles = len(os.listdir(directory))
		#for file_path in sorted(os.listdir(directory)):
		for i in range(numOfFiles):
			file_path = 'note' + str(i) + '.wav'
			detector = MIDI_Detector(directory + '/' + file_path)
			midi_numbers = detector.detect_MIDI_notes()
			print 'File: ' + str(file_path) + ' MIDI: ' + str(midi_numbers)
			if len(midi_numbers) > 0:
				lilypond_text += ' < '
				for n in midi_numbers:
					if n in self.number2note.keys():
						lilypond_text += self.number2note[n] + ' '
				lilypond_text += ' >'

		lilypond_text += '    \n}  \n}'
		with open(self.output_file, 'w') as f:
			f.write(lilypond_text)
		command = "lilypond "
		command += self.output_file
		print command
		os.system(command)



if __name__ == '__main__':
	wav_file = sys.argv[1]

	note_plotter = NotePlotter(wav_file)
	note_plotter.plot_multiple_notes()