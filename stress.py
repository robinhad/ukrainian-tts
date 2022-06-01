from unittest import skip
from gruut import sentences
import torch

importer = torch.package.PackageImporter("ukrainian-accentor/accentor-lite.pt")
accentor = importer.load_pickle("uk-accentor", "model")
replace_accents = importer.load_pickle("uk-accentor", "replace_accents")

# Using GPU
# accentor.cuda()
# Back to CPU
# accentor.cpu()

alphabet = "абгґдеєжзиіїйклмнопрстуфхцчшщьюя"

def accent_word(word):
    with torch.no_grad():
        stressed_words = accentor.predict([word], mode='stress')
    plused_words = [replace_accents(x) for x in stressed_words]
    return plused_words[0]

def sentence_to_stress(sentence):
    words = sentence.split()
    words = sum([[word, " "] for word in words], start=[])
    new_list = []
    for word in words:
        first_word_sep = list(map(lambda letter: letter in alphabet, word.lower()))
        if all(first_word_sep):
            new_list.append(word)
        else:
            current_index = 0
            past_index = 0
            for letter in first_word_sep:
                if letter == False:
                    new_list.append(word[past_index:current_index])
                    new_list.append(word[current_index])
                    past_index = current_index + 1
                current_index += 1
            new_list.append(word[past_index:current_index])
    #print(list(filter(lambda x: len(x) > 0, new_list)))
    for word_index in range(0, len(new_list)):
        element = new_list[word_index]
        first_word_sep = list(map(lambda letter: letter in alphabet, element.lower()))
        if not all(first_word_sep) or len(element) == 0:
            continue
        else:
            new_list[word_index] = accent_word(new_list[word_index])

    return "".join(new_list)


if __name__ == "__main__":
    sentence = "Кам'янець-Подільський - місто в Хмельницькій області України, центр Кам'янець-Подільської міської об'єднаної територіальної громади і Кам'янець-Подільського району."
    print(sentence_to_stress(sentence))
    #test_words1 = ["словотворення", "архаїчний", "програма", "а-ля-фуршет"]

    stressed_words = accentor.predict(["привіт"], mode='stress')
    plused_words = [replace_accents(x) for x in stressed_words]

    print('With stress:', stressed_words)
    print('With pluses:', plused_words)