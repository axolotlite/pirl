from playsound import playsound
import threading

playsound('./assets/unlockgesture.aac')

# threading.Thread(target=playsound, args=('./assets/unlockgesture.aac',)).start()