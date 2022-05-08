def preprocess_text(text):
    # replace apostrophe
    text = text.replace("`", "'")
    text = text.replace("ʼ", "'")
    # numbers
    text = text.replace("1", "од+ин ")
    text = text.replace("2", "дв+а ")
    text = text.replace("3", "тр+и ")
    text = text.replace("4", "чот+ири ")
    text = text.replace("5", "п'+ять ")
    text = text.replace("6", "ш+ість ")
    text = text.replace("7", "с+ім ")
    text = text.replace("8", "в+ісім ")
    text = text.replace("9", "д+ев'ять ")
    text = text.replace("0", "н+уль ")
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