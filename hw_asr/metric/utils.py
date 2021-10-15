# Don't forget to support cases when target_text == ''

import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        if len(predicted_text) == 0:
            return 0
        return 1
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target = target_text.split()
    predicted = predicted_text.split()
    if len(target) == 0:
        if len(predicted) == 0:
            return 0
        return 1
    return editdistance.eval(target, predicted) / len(target)
