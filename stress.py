import numpy as np
from ukrainian_word_stress import Stressifier, StressSymbol

stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)


def stress_dict(sentence: str):
    stressed = stressify(sentence.replace("+", "")).replace(StressSymbol.CombiningAcuteAccent, "+")
    new_stressed = ""
    start = 0
    last = 0

    while True:
        plus_position = stressed.find("+", start)
        if plus_position != -1:
            new_stressed += (
                stressed[last : plus_position - 1] + "+" + stressed[plus_position - 1]
            )
            start = plus_position + 1
            last = start
        else:
            new_stressed += stressed[last:]
            break
    return new_stressed


def sentence_to_stress(sentence: str, stress_function=stress_dict) -> str:
    # save custom stress positions
    all_stresses = []
    orig_words = sentence.split(" ")
    for i in range(0, len(orig_words)):
        if "+" in orig_words[i]:
            all_stresses.append(i)

    # add stress before vowel
    new_stressed = stress_function(sentence)
    
    # replace already stressed words
    if len(all_stresses) > 0:
        words = new_stressed.split(" ")
        for stressed in all_stresses:
            words[stressed] = orig_words[stressed]
        return " ".join(words)
    return new_stressed


if __name__ == "__main__":
    sentence = "Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."
    print(sentence_to_stress(sentence))
    sentence = "Привіт, як тебе звати?"
    print(sentence_to_stress(sentence))
    sentence = "АННА - український панк-рок гурт"
    print(sentence_to_stress(sentence))
    sentence = "Не тільки в Україні таке може бути."
    print(sentence_to_stress(sentence))
    sentence = "Не тільки в +Укра+їні т+аке може бути."
    print(sentence_to_stress(sentence))
    sentence = "два + два"
    print(sentence_to_stress(sentence))
    sentence = "Н тльк в крн тк мж бт."
    print(sentence_to_stress(sentence))
    sentence = "Н тльк в крн тк мж бт."
    print(sentence_to_stress(sentence))
