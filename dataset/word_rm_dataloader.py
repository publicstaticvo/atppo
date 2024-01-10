from util import *
from .dataset_base import DataCollatorForAT


class DataCollatorForWordRM(DataCollatorForAT):
    def __init__(self, tokenizer, config, fp16=False):
        super(DataCollatorForWordRM, self).__init__(tokenizer, config, fp16, 0)
        self.num_negative = config.num_negative

    def __call__(self, batch):
        audios, a_mask, texts, t_mask, a_valid, t_valid, turn_id, labels, negative_indices = [], [], [], [], [], [], [], [], []
        ml = 0
        for item in batch:
            ml = max([ml, len(item[1]) + len(item[4]) + len(item[6]) - 2])
        ml = min(ml, self.config.text.max_length)
        for item in batch:
            # at和pt 有0有2 history为N-2轮 有0有2 每一轮用2分隔
            aa, at, atr, pa, pt, ptr, history = item
            aa, pa = map(torch.HalfTensor if self.fp16 else torch.FloatTensor, [aa, pa])
            aa, a_aam = pad_cut(aa, self.config.audio.max_length)
            pa, p_aam = pad_cut(pa, self.config.audio.max_length)
            text = torch.LongTensor(history + at[1:] + pt[1:])
            if text.shape[0] > ml:
                offset_p = ml - len(pt)
                offset_a = offset_p - len(at) + 1
            else:
                offset_a = len(history) - 1
                offset_p = offset_a + len(at) - 1
            text, tam = pad_cut(text, ml)

            text_marks = []
            anchor_audio_marks = []
            positive_audio_marks = []
            for i, x in enumerate(atr):
                anchor_audio_marks.append([x[3], x[4]])
                text_marks.append([x[1] + offset_a, x[2] + offset_a])
            for i, x in enumerate(ptr):
                positive_audio_marks.append([x[3], x[4]])
                text_marks.append([x[1] + offset_p, x[2] + offset_p])

            anchor_audio_marks = group_scale_audio_length(torch.LongTensor(anchor_audio_marks), self.config.audio)
            positive_audio_marks = group_scale_audio_length(torch.LongTensor(positive_audio_marks), self.config.audio)
            audio_length = scale_audio_length(0, self.config.audio.max_length, self.config.audio)
            anchor_audio_marks -= audio_length[0]
            positive_audio_marks -= audio_length[0]
            for i in range(anchor_audio_marks.shape[0] - 1, -1, -1):
                if anchor_audio_marks[i, 1] > audio_length[1]: anchor_audio_marks[i, 1] -= 1
                else: break
                if anchor_audio_marks[i, 0] == anchor_audio_marks[i, 1]: anchor_audio_marks[i, 0] -= 1
            for i in range(positive_audio_marks.shape[0] - 1, -1, -1):
                if positive_audio_marks[i, 1] > audio_length[1]: positive_audio_marks[i, 1] -= 1
                else: break
                if positive_audio_marks[i, 0] == positive_audio_marks[i, 1]: positive_audio_marks[i, 0] -= 1
            anchor_audio_marks += 1
            positive_audio_marks += (anchor_audio_marks[-1, 1] + 1)
            # 0在scale_audio_length之后会变-1
            audio_marks = torch.cat([anchor_audio_marks, positive_audio_marks], dim=0)
            audio_valid, _ = compute_valid_for_rm(audio_marks.tolist(), audio_length[1] * 2 + 2, self.config.audio.pooling_mode)
            text_valid, num = compute_valid_for_rm(text_marks, ml, self.config.text.pooling_mode)
            assert num == len(audio_valid) == len(atr) + len(ptr)
            # 负采样
            negative_indices.append(negative_sampling(atr + ptr, self.num_negative))

            audios.extend([aa, pa])
            a_mask.extend([a_aam, p_aam])
            texts.append(text)
            t_mask.append(tam)
            a_valid.append(audio_valid)
            t_valid.append(text_valid)
            turn_id.append(torch.cat([torch.zeros(offset_p + 1), torch.ones(ml - offset_p - 1)]).long())
        audios, a_mask, texts, t_mask, turn_id = map(lambda t: torch.stack(t, dim=0), [audios, a_mask, texts, t_mask, turn_id])
        return audios, a_mask, a_valid, texts, t_mask, t_valid, turn_id, negative_indices
