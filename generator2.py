from mimi import MidiFile, MidiTrack, generator
from mimi.Mimi import AbsNote
import mido




for ins in range(128):
    for x in range(40, 76):
        with MidiFile() as mid:
            track = MidiTrack(channel=0, instrument=ins)
            mid.tracks.append(track)
            track.append(AbsNote(pitch=x, time=256))
            mid.save_mp3("data/mp3/%03d-%03d.mp3" % (ins, x))
            mid.save_npz("data/npz/%03d-%03d.npz" % (ins, x))

    # mid.draw_roll()



    # mid.play()