from robot import config
from robot.hardware.amp import Amp
from robot.services.speaker import Speaker


WAV_FILE = config.ROOT / "sounds/mixkit-retro-confirmation-tone-2860.wav"


Speaker(Amp(config.APLAY_DEVICE)).play(WAV_FILE)
Speaker(Amp(config.APLAY_DEVICE)).speak("I could not find a red ball.")
