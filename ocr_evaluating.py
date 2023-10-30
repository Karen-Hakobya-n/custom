import easyocr


def easyocr_test(image_path, ground_truth_text):
    reader = easyocr.Reader(['en'], recog_network='english_g2')

    # Perform OCR on the image
    result = reader.readtext(image_path)

    recognized_text = ' '.join([text[1] for text in result])

    total_characters = len(ground_truth_text)
    correct_characters = sum([1 for a, b in zip(ground_truth_text, recognized_text) if a == b])
    character_accuracy = correct_characters / total_characters

    ground_truth_words = ground_truth_text.split()
    recognized_words = recognized_text.split()
    total_words = len(ground_truth_words)
    correct_words = sum([1 for a, b in zip(ground_truth_words, recognized_words) if a == b])
    word_accuracy = correct_words / total_words

    return (f'Recognized Text: {recognized_text}', f'Character Accuracy: {character_accuracy:.2%}',
            f'Word Accuracy: {word_accuracy:.2%}')


print(easyocr_test('/home/user/PycharmProjects/pythonProject1/Automatic-License-Plate-Recognition-using-YOLOv8-main/0a0d1748-48cd-4114-90cb-b5baf0b3cbe4___3e7fd381-0ae5-4421-8a70-279ee0ec1c61_147274518_15141875973_large.jpg', 'IND MH 46 X 9996'))
