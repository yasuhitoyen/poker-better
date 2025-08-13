import torch
import torch.nn as nn  # alias nn so we can use nn.Sequential, nn.Linear, etc.

class DQN(nn.Module):
  def __init__(self,
               state_dim:int,
               action_dim:int,
               history_dim:int,
               hidden_dim:int=256,
               history_rnn_dim:int=128):
    super().__init__()

    # history rnn, history_dim is the size of each snapshot
    self.history_rnn = HistoryRNN(history_dim, history_rnn_dim)

    # state processor, input for "what do i see right now?"
    self.state_processor = nn.Sequential(
        nn.Linear(in_features=state_dim, out_features=hidden_dim),
        nn.ReLU(),
        nn.Dropout()
    )

    # combined dimension, combining state and history
    combined_dim = hidden_dim + history_rnn_dim
    combined_output_dim = hidden_dim // 2  # fix typo: combiend_output_dim -> combined_output_dim

    # combined processor used for q head
    self.combined_processor = nn.Sequential(
        nn.Linear(in_features=combined_dim, out_features=hidden_dim),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(in_features=hidden_dim, out_features=combined_output_dim),
        nn.ReLU()
    )

    # expected future return (now + later)
    self.q_head = nn.Linear(
        in_features=combined_output_dim, out_features=action_dim) # 5

    # reward for current action, state
    self.reward_function = FCNRewardFunction(input_dim=combined_dim) # 64

  # forward will have the current state
  def forward(self,
              history,
              state,
              sequence_lengths,  # NEW: always pass sequence lengths so RNN can ignore padding
              return_reward:bool=False):
    # history: [batch, seq_length, history_dim]
    history_vector = self.history_rnn(history, sequence_lengths)

    state_vector = self.state_processor(state)

    combined_vector = torch.cat((history_vector, state_vector), dim=1)

    combined_embedding = self.combined_processor(combined_vector)

    q_values = self.q_head(combined_embedding)

    if return_reward:
      fcn_reward = self.reward_function(combined_vector)
      return q_values, fcn_reward

    return q_values

