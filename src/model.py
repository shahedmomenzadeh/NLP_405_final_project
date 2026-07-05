"""BERT-based NER model with selective freezing, MLP/BiLSTM interface, and CRF layer.

Architecture (4 layers):
    Layer 1: Projection (tokenizer + BERT embeddings)
    Layer 2: BERT-base contextual representation (frozen embeddings + layers 0-9)
    Layer 3: Neural network interface (MLP or Bi-LSTM)
    Layer 4: CRF layer for sequence labeling
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertConfig, BertModel, BertPreTrainedModel

from .config import BERT_NAME, IGNORE_INDEX


class MLPInterface(nn.Module):
    """Multi-Layer Perceptron interface between BERT and CRF.

    Architecture: Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, input_dim)
        Returns:
            emissions: (batch_size, seq_len, output_dim)
        """
        x = self.fc1(hidden_states)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BiLSTMInterface(nn.Module):
    """Bidirectional LSTM interface between BERT and CRF."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, input_dim)
        Returns:
            emissions: (batch_size, seq_len, output_dim)
        """
        lstm_out, _ = self.lstm(hidden_states)
        lstm_out = self.dropout(lstm_out)
        emissions = self.fc(lstm_out)
        return emissions


class BertNERModel(BertPreTrainedModel):
    """BERT-based Named Entity Recognition model with CRF.

    Layers:
        1. Projection: BERT embeddings (frozen)
        2. Contextual: BERT encoder (layers 0-9 frozen, 10-11 trainable)
        3. Interface: MLP or Bi-LSTM
        4. CRF: Conditional Random Field for sequence labeling
    """

    def __init__(
        self,
        num_tags: int,
        interface_type: Literal["mlp", "bilstm"] = "mlp",
        hidden_dim: int = 256,
        num_lstm_layers: int = 1,
        dropout: float = 0.3,
        bert_name: str = BERT_NAME,
        freeze_bert_layers: bool = True,
        load_bert_weights: bool = True,
    ):
        """
        Args:
            num_tags: Number of BIO tags (including PAD and X)
            interface_type: "mlp" or "bilstm"
            hidden_dim: Hidden dimension for MLP or Bi-LSTM
            num_lstm_layers: Number of LSTM layers (only for Bi-LSTM)
            dropout: Dropout rate
            bert_name: Pretrained BERT model name
            freeze_bert_layers: If True, freeze embeddings and layers 0-9
            load_bert_weights: If False, build BERT with random weights (no download).
                Useful for fast architecture smoke-tests.
        """
        config = BertConfig.from_pretrained(bert_name)
        super().__init__(config)

        if load_bert_weights:
            self.bert = BertModel.from_pretrained(bert_name)
        else:
            self.bert = BertModel(config)
        self.num_tags = num_tags
        self.interface_type = interface_type

        if freeze_bert_layers:
            self._freeze_bert_layers()

        bert_hidden_size = config.hidden_size

        if interface_type == "mlp":
            self.interface = MLPInterface(
                input_dim=bert_hidden_size,
                output_dim=num_tags,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        elif interface_type == "bilstm":
            self.interface = BiLSTMInterface(
                input_dim=bert_hidden_size,
                output_dim=num_tags,
                hidden_dim=hidden_dim,
                num_layers=num_lstm_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"interface_type must be 'mlp' or 'bilstm', got {interface_type}")

        self.crf = CRF(num_tags, batch_first=True)

    def _freeze_bert_layers(self):
        """Freeze BERT embeddings and encoder layers 0-9.

        Keep layers 10-11 (the last two) trainable for fine-tuning.
        """
        for name, param in self.bert.embeddings.named_parameters():
            param.requires_grad = False

        for i in range(10):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            labels: (batch_size, seq_len) - tag indices, with IGNORE_INDEX for padding
            token_type_ids: (batch_size, seq_len) - optional

        Returns:
            Dict with 'loss' (if labels provided) and 'emissions'
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        hidden_states = outputs.last_hidden_state

        emissions = self.interface(hidden_states)

        mask = attention_mask.bool()

        result = {"emissions": emissions, "mask": mask}

        if labels is not None:
            crf_labels = labels.clone()
            crf_labels[crf_labels == IGNORE_INDEX] = 0

            log_likelihood = self.crf(emissions, crf_labels, mask=mask, reduction="mean")
            result["loss"] = -log_likelihood

        return result

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Decode the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: (batch_size, seq_len, num_tags)
            mask: (batch_size, seq_len)

        Returns:
            List of tag sequences (list of lists of tag indices)
        """
        return self.crf.decode(emissions, mask=mask)
