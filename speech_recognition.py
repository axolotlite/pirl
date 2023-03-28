import speech_recognition as sr
import os
import pyttsx3
import mouse

engine = None

def main():
    global engine
    if os.name == "nt":
        engine = pyttsx3.init('sapi5')
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
    elif os.name == "posix":
        engine = pyttsx3.init()
    speak("Welcome to the house of fun")
    
    while True:
        query = takeCommand().lower()
        if 'click' in query:
            mouse.click()
        elif 'press' in query:
            mouse.hold()
        elif 'release' in query:
            mouse.release()
        elif 'options' in query:
            mouse.right_click()
        elif 'exit' in query:
            speak("Thanks for giving me your time")
            exit()
    

def speak(audio):
    engine.say(audio)
    engine.runAndWait()
    
def takeCommand():
     
    r = sr.Recognizer()
     
    with sr.Microphone() as source:
         
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
  
    try:
        print("Recognizing...")   
        query = r.recognize_google(audio, language ='en-in')
        print(f"User said: {query}\n")
  
    except Exception as e:
        print(e)   
        print("Unable to Recognize your voice.") 
        return "None"
     
    return query

if __name__ == "__main__":
    main()