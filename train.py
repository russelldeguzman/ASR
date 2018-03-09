from hmm import HMM

WAVE_OUTPUT_FILENAMES = ['Odessa', 'Turn on the lights', 'Turn off the lights', 'What time is it', 'Play music', 'Stop music']

if __name__ == "__main__":
    m1 = HMM(WAVE_OUTPUT_FILENAMES[0], 20)
    m2 = HMM(WAVE_OUTPUT_FILENAMES[1], 20)
    m3 = HMM(WAVE_OUTPUT_FILENAMES[2], 20)
    m4 = HMM(WAVE_OUTPUT_FILENAMES[3], 20)
    m5 = HMM(WAVE_OUTPUT_FILENAMES[4], 20)
    m6 = HMM(WAVE_OUTPUT_FILENAMES[5], 20)

    m1.train(20)
    m2.train(20)
    m3.train(20)
    m4.train(20)
    m5.train(20)
    m6.train(20)
