from silence_detect import find_speech
from mfcc import ret_mfcc
import sounddevice as sd
import time
duration = 3  # 3 seconds
CHANNELS = 1  # one channel, so mono not stereo
fs = 16000    # 100Hz sample rate

if __name__ == "__main__":
    print ("Starting Odessa")
    time.sleep(2)

    print ("Loading and initing HMMs")
    hmm_list = []
    #TODO: load and init HMMS

    while True:
        #listen for speech
        print("Speak now!")
        s = sd.rec(int(duration * fs), samplerate=fs,
               channels=CHANNELS); sd.wait()
        print("Processing...")

        #silence processing
        sig = find_speech(s,fs)

        #post silence processing
        if len(sig) < (int(fs / 10)):
            print "Nothing detected!"
            continue

        #get mfccs
        mfcc = ret_mfcc(sig,fs)

        for hmm in hmm_list:
            #TODO: check HMMs for best match
