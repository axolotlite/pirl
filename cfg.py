from screeninfo import get_monitors

class CFG:
    def __init__(self,camIdx=0, mainScreen=0, handThreadScreen=0):
        self.camIdx = camIdx
        self.pmons = get_monitors()
        self.handThreadScreen = handThreadScreen
        self.width = 1280
        self.height = 720
        self.MJPG = True
        if( len(self.pmons) == 1):
            self.mainScreen = 0
        self.mainScreen = mainScreen