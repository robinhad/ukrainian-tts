from typing import List
from ukrainian_word_stress import Stressifier, StressSymbol
import ukrainian_accentor as accentor

stressify = Stressifier(stress_symbol=StressSymbol.CombiningAcuteAccent)

vowels = "аеєиіїоуюя"
consonants = "бвгґджзйклмнпрстфхцчшщь"
special = "'-"
alphabet = vowels + consonants + special + "+"


def _shift_stress(stressed, symboll_to_shift="+"):
    new_stressed = ""
    start = 0
    last = 0

    # shift stress symbol by one "при+віт" -> "пр+ивіт"
    while True:
        plus_position = stressed.find(symboll_to_shift, start)
        if plus_position != -1:
            new_stressed += (
                stressed[last : plus_position - 1]
                + symboll_to_shift
                + stressed[plus_position - 1]
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
    # replace acute accent with plus
    sentence = _shift_stress(sentence, "́")
    sentence = sentence.replace("́", "+")

    # save custom stress positions
    all_stresses = []
    orig_words = sentence.split(" ")
    for i in range(0, len(orig_words)):
        if "+" in orig_words[i]:
            all_stresses.append(i)

    # add stress before vowel
    new_stressed = stress_function(sentence)

    # stress single vowel words
    new_list = []
    # if letter is not in alphabet, then consider it an end of the word
    previous = 0
    for i, letter in enumerate(new_stressed):
        if letter.lower() not in alphabet:
            if previous == i:
                new_list.append(new_stressed[i])
            else:
                new_list.append(new_stressed[previous:i])
                new_list.append(new_stressed[i])
            previous = i + 1
    # add remainder
    if previous != len(new_stressed):
        new_list.append(new_stressed[previous:])

    # add stress to single-vowel words
    for word_index in range(0, len(new_list)):
        element: str = new_list[word_index]
        vowels_in_words = list(map(lambda letter: letter in vowels, element.lower()))
        if "+" in element:
            if element.count("+") > 1:
                first = element.find("+")
                new_list[word_index] = new_list[word_index][: first + 1] + new_list[
                    word_index
                ][first + 1 :].replace("+", "")
            continue
        if vowels_in_words.count(True) == 0:
            continue
        elif vowels_in_words.count(True) == 1:
            vowel_index = vowels_in_words.index(True)
            new_list[word_index] = element[0:vowel_index] + "+" + element[vowel_index::]
        elif vowels_in_words.count(True) > 1:
            new_list[word_index] = stress_with_model(element)

    new_stressed = "".join(new_list)

    # replace already stressed words
    if len(all_stresses) > 0:
        words = new_stressed.split(" ")
        for stressed in all_stresses:
            words[stressed] = orig_words[stressed]
        return " ".join(words)
    return new_stressed
