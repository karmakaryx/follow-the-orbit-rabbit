from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class OrbitalSequenceDataset(Dataset):
    """sequence_builder.build_sequences()의 반환값을 감싸는 Dataset"""

    def __init__(self, sequences: dict[int, dict]):
        self.norad_ids = list(sequences.keys())
        self.features = [
            torch.from_numpy(sequences[nid]["features"]).float()
            for nid in self.norad_ids
        ]

    def __len__(self) -> int:
        return len(self.norad_ids)

    def __getitem__(self, idx: int):
        feat = self.features[idx]  # [T_i, F]
        return feat, feat.shape[0], self.norad_ids[idx]


def collate_padded(batch):
    """DataLoader collate_fn: 배치 내 최대 길이로 패딩 + 마스크 생성

    반환:
      padded:    [B, T_max, F]  (뒤쪽 0-패딩)
      lengths:   [B]            (각 시퀀스 실제 길이 - pack_padded_sequence용)
      mask:      [B, T_max]     (True=실제 데이터, False=패딩 -> loss 계산에서 제외)
      norad_ids: [B]            (역추적/디버깅용)
    """
    sequences, lengths, norad_ids = zip(*batch)

    # 길이 내림차순 정렬 (pack_padded_sequence 최적화)
    order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sequences = [sequences[i] for i in order]
    lengths = [lengths[i] for i in order]
    norad_ids = [norad_ids[i] for i in order]

    padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)  # [B, T_max, F]
    lengths_t = torch.tensor(lengths, dtype=torch.long)

    t_max = padded.shape[1]
    arange = torch.arange(t_max).unsqueeze(0)  # [1, T_max]
    mask = arange < lengths_t.unsqueeze(1)  # [B, T_max] bool

    return padded, lengths_t, mask, torch.tensor(norad_ids, dtype=torch.long)


def masked_mse_loss(reconstructed: torch.Tensor, original: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ...
    # Note by Karyx💫: This code is omitted to protect my intellectual property.
