from num2words import num2words
import re


def number_form(number):
    if number[-1] == "1":
        return 0
    elif number[-1] in ("2", "3", "4"):
        return 1
    else:
        return 2


CURRENCY = {
    "USD": ("долар", "долари", "доларів"),
    "UAH": ("гривня", "гривні", "гривень"),
    "EUR": ("євро", "євро", "євро"),
}


def replace_currency_with_words(text, currency, num_form):
    if currency == "USD":
        text = text.replace("$", CURRENCY[currency][num_form])

    if currency == "UAH":
        text = text.replace("₴", CURRENCY[currency][num_form])

    if currency == "EUR":
        text = text.replace("€", CURRENCY[currency][num_form])
    return text

def find_any_char(text: str, find: str, start: int):
    result = -1
    for c in find:
        index = text.find(c, start)
        if (index >= 0) and (result > index or result == -1):
            result = index

    return result

# Have to check if I can use https://github.com/lang-uk/tokenize-uk
def simple_tokenizer(text: str):
    start = 0
    index = find_any_char(text, " ,", start)
    while (index >= 0):
        word = text[start:index]
        yield word
        separator = text[index]
        yield separator
        start = index + 1
        index = find_any_char(text, " ,", start)

    yield text[start:]

def preprocess_text(text):
    text = text.lower()
    # currencies
    if "$" in text:
        currency = "USD"
        gender = "masculine"
    elif "₴" in text:
        currency = "UAH"
        gender = "feminine"
    elif "€" in text:
        currency = "EUR"
        gender = "masculine"
    else:
        currency = ""
        gender = "masculine"

    num_form = 0
    # replace apostrophe
    text = text.replace("`", "'")
    text = text.replace("ʼ", "'")
    text = text.replace("…", "...")

    symbols = {
        "”": '"',
        "“": '"',
        "’": '"',
        "‘": '"',
        "«": '"',
        "»": '"',
        "–": "-",
        "—": "-",
        "―": "-",
    }
    for symbol, value in symbols.items():
        text = text.replace(symbol, value)
    # numbers
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)

    def detect_num_and_convert(word):
        numbers = "0123456789"
        splits = ",."
        currencies = "$₴€"
        result = []
        nonlocal num_form
        parts = word.split("-")  # for handling complex words
        for part in parts:
            is_number = all(map(lambda x: x in numbers, part)) or (
                any(map(lambda x: x in numbers, part))
                and any(map(lambda x: x in splits, part))
            )
            is_currency = any(map(lambda x: x in currencies, part)) and any(
                map(lambda x: x in numbers, part)
            )  # contains both number and currency symbol
            if is_number or is_currency:
                try:
                    if is_currency:
                        cleaned_part = part

                        for part_currency in currencies:
                            if cleaned_part[0] == part_currency:
                                cleaned_part = cleaned_part[1:] + " " + part_currency
                            else:
                                cleaned_part = cleaned_part.replace(
                                    part_currency, f" {part_currency} "
                                ).strip()  # TODO: replace with regex

                        part = " ".join(
                            [
                                detect_num_and_convert(part_word)
                                for part_word in cleaned_part.split(" ")
                            ]
                        )

                    ends_with_dot = part.endswith(".")  # ugly
                    ends_with_comma = part.endswith(",")
                    if ends_with_comma or ends_with_dot:
                        part = part[:-1]
                        part = " ".join(
                            [
                                detect_num_and_convert(part_word)
                                for part_word in part.split(" ")
                            ]
                        ) + ("." if ends_with_dot else ",")

                    num_form = number_form(part)
                    result.append(num2words(part.strip(), lang="uk", gender=gender))
                except:
                    result.append(part)
            else:
                result.append(part)
        return "-".join(result)

    # print([detect_num_and_convert(word) for word in simple_tokenizer(text)])
    text = "".join([detect_num_and_convert(word) for word in simple_tokenizer(text)])
    text = replace_currency_with_words(text, currency, num_form)

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
        "qu": "кв",
        "ch": "ч",
        "sh": "ш",
        "шч": "щ",  # after previous cases
        "ph": "ф",
        "kh": "х",
        "yo": "йо",
        "yu": "ю",
        "ya": "я",
        "ye": "є",
        "yi": "ї",
        "zh": "ж",
        "ts": "ц",
        "th": "т",
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
    for english_char, english_value in english.items():
        # uppercase
        text = text.replace(english_char.upper(), english_value.upper())
        text = text.replace(english_char, english_value)

    return text
