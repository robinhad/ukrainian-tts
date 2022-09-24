from typing import List
from ukrainian_word_stress import Stressifier, StressSymbol
import ukrainian_accentor as accentor

stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)

vowels = "аеєиіїоуюя"
consonants = "бвгґджзйклмнпрстфхцчшщь"
special = "'"
alphabet = vowels + consonants + special


def _shift_stress(stressed):
    new_stressed = ""
    start = 0
    last = 0

    # shift stress symbol by one "при+віт" -> "пр+ивіт"
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


def stress_with_model(text: str):
    text = text.lower()
    result = accentor.process(text, mode="plus")
    return result


def stress_dict(sentence: str):
    stressed = stressify(sentence.replace("+", "")).replace(
        StressSymbol.CombiningAcuteAccent, "+"
    )
    return _shift_stress(stressed)


def sentence_to_stress(sentence: str, stress_function=stress_dict) -> str:
    # save custom stress positions
    all_stresses = []
    orig_words = sentence.split(" ")
    for i in range(0, len(orig_words)):
        if "+" in orig_words[i]:
            all_stresses.append(i)

    # add stress before vowel
    new_stressed = stress_function(sentence)

    # stress single vowel words
    new_list: List[str] = new_stressed.split(" ")
    for word_index in range(0, len(new_list)):
        element = new_list[word_index]
        vowels_in_words = list(map(lambda letter: letter in vowels, element.lower()))
        if "+" in element:
            continue
        if vowels_in_words.count(True) == 0:
            continue
        elif vowels_in_words.count(True) == 1:
            vowel_index = vowels_in_words.index(True)
            new_list[word_index] = element[0:vowel_index] + "+" + element[vowel_index::]
    new_stressed = " ".join(new_list)

    # replace already stressed words
    if len(all_stresses) > 0:
        words = new_stressed.split(" ")
        for stressed in all_stresses:
            words[stressed] = orig_words[stressed]
        return " ".join(words)
    return new_stressed


if __name__ == "__main__":
    # TODO: move it to unit tests
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

    sentence = "Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."
    print(stress_with_model(sentence))
    sentence = "Привіт, як тебе звати?"
    print(stress_with_model(sentence))
    sentence = "АННА - український панк-рок гурт"
    print(stress_with_model(sentence))
    sentence = "Не тільки в Україні таке може бути."
    print(stress_with_model(sentence))
    sentence = "Не тільки в +Укра+їні т+аке може бути."
    print(stress_with_model(sentence))
    sentence = "два + два"
    print(stress_with_model(sentence))
    sentence = "Н тльк в крн тк мж бт."
    print(stress_with_model(sentence))
    sentence = "Н тльк в крн тк мж бт."
    print(stress_with_model(sentence))
