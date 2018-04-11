from mimi import MidiFile, MidiTrack, generator

for x in range(100):

    tracks = [MidiTrack(channel=x, instrument=(x+1)*8) for x in range(3)]

    for track in tracks:
        track.append_bar(generator.get_random_tab(tempo=70))

    mid = MidiFile()
    mid.tracks.extend(tracks)
    # mid.draw_roll()

    mid.save_mp3("data/mp3/%03d.mp3" % x)
    mid.save_npz("data/npz/%03d.npz" % x)

    # mid.play()