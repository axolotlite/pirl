from screeninfo import get_monitors

class CFG:
    camIdx = 0
    camWidth = 1280
    camHeight = 720
    handThreadScreen = 0
    MJPG = True
    
    # 0 for primary, 1 for secondary
    mainScreen = 0
    monitors = get_monitors()
    # Make the primary monitor always have index 0
    if not monitors[0].is_primary:
        monitors.reverse()
    # If there is only one monitor, set it as the primary monitor
    if len(monitors) == 1:
        mainScreen = 0
