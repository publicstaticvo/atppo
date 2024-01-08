import torch


def compute_valid_for_tpp(transcript, offset, length, mode, audio_length):
    sv = [0 for _ in range(length)]
    ev = [0 for _ in range(length)]
    start_labels, end_labels = [], []
    for i, item in enumerate(transcript):
        sv[offset + item[-4]] = 1
        ev[offset + item[-3] - 1] = 1
        sl, el = float(f"{item[-2] / audio_length:.3f}"), float(f"{item[-1] / audio_length:.3f}")
        if mode:
            start_labels.append(int(sl * 100))
            end_labels.append(int(el * 100) - 1)
        else:
            start_labels.append(sl)
            end_labels.append(el)
    return torch.BoolTensor(sv), torch.BoolTensor(ev), start_labels, end_labels
