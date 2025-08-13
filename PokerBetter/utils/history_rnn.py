import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from typing import List
class HistoryRNN(nn.Module):
  def __init__(self,
               input_dim:int,
               hidden_dim:int=128,
               num_layers:int=2,
               dropout:float=0.1):
    super().__init__()

    # store dims for init of hidden/cell states
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers

    self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                       num_layers=num_layers, batch_first=True, dropout=dropout)

    self.output_projection = nn.Linear(hidden_dim, hidden_dim)

  def forward(self,
              history_sequence:torch.Tensor,
              sequence_lengths:List[int]):
    # History sequence contains the actual sequence of prior game states
    # Sequence lengths shows the true length of each game state prior to padding
    # BTD Batch Size, Max History Length, Input feature size per timestep

    # D is the size of every game snapshot that is to be consumed by RNN
    batch_size = history_sequence.size(0)
    seq_len = history_sequence.size(1)

    if history_sequence.dim() == 2: # in case we only have one history
      history_sequence = history_sequence.unsqueeze(0)
      batch_size = 1

    # hidden and cell state
    h0 = torch.zeros(
        self.num_layers, batch_size, self.hidden_dim).to(history_sequence.device)
    c0 = torch.zeros(
        self.num_layers, batch_size, self.hidden_dim).to(history_sequence.device)

    if sequence_lengths is not None:
      # accounts for padding since padding will pollute input with meaningless data
      # normalize lengths to a plain python list of ints on CPU
      if isinstance(sequence_lengths, torch.Tensor):
        sequence_lengths = sequence_lengths.detach().cpu().tolist()
      else:
        sequence_lengths = [int(x) for x in sequence_lengths]

      packed = rnn.pack_padded_sequence(
          input=history_sequence, lengths=sequence_lengths, batch_first=True, enforce_sorted=False
      )
      # pass (packed, (h0, c0)) to the LSTM; don't call packed as a function
      packed_output, (hn, cn) = self.rnn(packed, (h0, c0))
      output, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
    else:
      # no need to pack since we only have one example
      output, (hn, cn) = self.rnn(history_sequence, (h0, c0))

    history_encoding = self.output_projection(hn[-1])

    return history_encoding

