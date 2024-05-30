import speech_recognition as sr
import pyaudio


def recognize_colors():
    
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak the sentence:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        sentence = recognizer.recognize_google(audio)
        print("You said:", sentence)

        colors = []
        for word in sentence.split():
            if word.lower() in ['green', 'red', 'blue', 'yellow', 'black', 'pink', 'orange', 'brown', 'purple']:  
                colors.append(word.lower())

        print("Recognized colors:", colors)

    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

if __name__ == "__main__":
    recognize_colors()
