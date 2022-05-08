def preprocess_text(text):
    # replace apostrophe
    text = text.replace("`", "'")
    text = text.replace("ʼ", "'")
    # numbers
    text = text.replace("1", "один ")
    text = text.replace("2", "два ")
    text = text.replace("3", "три ")
    text = text.replace("4", "чотири ")
    text = text.replace("5", "п'ять ")
    text = text.replace("6", "шість ")
    text = text.replace("7", "сім ")
    text = text.replace("8", "вісім ")
    text = text.replace("9", "дев'ять ")
    text = text.replace("0", "нуль ")
    # speak english alphabet using brute force transliteration
    english = {
        "a": "а",
        "b": "б",
        "c": "ц",
        "d": "д",
        "e": "е",
        "f": "ф",
        "g": "ґ",
        "h": "г",
        "i": "і",
        "j": "дж",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "q": "кв",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "ю",
        "v": "в",
        "w": "в",
        "x": "кс",
        "y": "й",
        "z": "з",
    }
    for english_char in english.keys():
        # uppercase
        text = text.replace(english_char.upper(),  english[english_char].upper())
        text = text.replace(english_char, english[english_char])

    # TODO: autostress support here
    return text


if __name__ == "__main__":
    print(preprocess_text("Quality of life update"))