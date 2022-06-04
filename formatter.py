import num2words
import re
from stress import sentence_to_stress

def preprocess_text(text, autostress=False):
    # currencies
    text = text.replace("$", "долар")
    text = text.replace("₴", "гривня")
    text = text.replace("€", "євро")
    # replace apostrophe
    text = text.replace("`", "'")
    text = text.replace("ʼ", "'")
    # numbers
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    def detect_num_and_convert(word):
        numbers = "0123456789,."
        is_number = all(map(lambda x: x in numbers, word))
        if is_number:
            try:
                return num2words.num2words(word, lang="uk")
            except:
                return word
        else:
            return word

    text = " ".join([detect_num_and_convert(word) for word in text.split(" ")])

    # fallback numbers
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
        "y": "і",
        "z": "з",
    }
    for english_char in english.keys():
        # uppercase
        text = text.replace(english_char.upper(),  english[english_char].upper())
        text = text.replace(english_char, english[english_char])

    if autostress:
        text = sentence_to_stress(text)

    return text


if __name__ == "__main__":
    print(preprocess_text("Quality of life update"))
    print(preprocess_text("Він украв 20000000 $"))
    print(preprocess_text("111 000 000 000 доларів державного боргу."))
    print(preprocess_text("11100000001 доларів державного боргу."))