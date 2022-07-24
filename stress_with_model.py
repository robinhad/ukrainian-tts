import ukrainian_accentor as accentor

# run
def stress_with_model(text: str):
    text = text.lower()
    try:
        result = accentor.process(text, mode='plus')
    except ValueError: # TODO: apply fix for cases when there are no vowels
        return text
    return result


if __name__ == "__main__":
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
