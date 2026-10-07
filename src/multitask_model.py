import torch
import torch.nn as nn
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union

class Wav2Vec2ForPronunciationAssessment(Wav2Vec2PreTrainedModel):
    """
    A custom Wav2Vec2 model with two heads:
    1. A CTC head for phoneme recognition (`lm_head`) -> Cross-Entropy Loss
    2. A Regression head for pronunciation scoring (`score_head`) -> MSE Loss
    """
    def __init__(self, config):
        super().__init__(config)

        # Base encoder
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        # Head 1: Phoneme prediction (CTC)
        self.target_lang = None
        if config.vocab_size is None:
            raise ValueError("You must specify `vocab_size` when instantiating this model.")
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # Head 2: Scoring prediction (Regression: 0.0 to 10.0)
        # We project the hidden state down to a single score.
        self.score_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.final_dropout),
            nn.Linear(config.hidden_size // 2, 1) # Predicts 1 continuous value
        )

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,     # CTC labels for phonemes
        score_labels: Optional[torch.Tensor] = None # Regression labels for accuracy
    ) -> Union[Tuple, SequenceClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 1. Base Encoder Pass
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # hidden_states shape: [batch_size, sequence_length, hidden_dim]
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 2. Head 1: Phoneme CTC Logits
        logits = self.lm_head(hidden_states)

        # 3. Head 2: Pronunciation Score (Pool over time, then regress)
        # Pool strategy: mean over the sequence length axis.
        # This gives a [batch_size, hidden_dim] representation of the entire audio clip.
        pooled_hidden_states = hidden_states.mean(dim=1) 
        score_preds = self.score_head(pooled_hidden_states).squeeze(-1) # Shape: [batch_size]

        loss = None
        if labels is not None:
            # Calculate CTC Loss
            # Retrieve lengths of input sequences (in frames) and target sequences
            # Flatten to vector, etc. Handled by standard CTC logic from HF
            # We must be careful as pytorch ctc_loss expects (time, batch, class)
            
            # Retrieve lengths
            if attention_mask is not None:
                # compute real output lengths according to convolution formula
                input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            else:
                input_lengths = torch.full(
                    (logits.shape[0],), logits.shape[1], dtype=torch.long, device=logits.device
                )

            # ctc_loss expects [sequence_length, batch_size, vocab_size]
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                # Calculate CTC loss
                # Remove padding from targets to create 1D flattened targets and target_lengths
                targets_mask = labels >= 0
                target_lengths = targets_mask.sum(-1)
                flattened_targets = labels[targets_mask]
                
                ctc_loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )
                
            loss = ctc_loss

            # 4. Calculate Regression Loss (If scores provided)
            if score_labels is not None:
                # Ensure float for MSE
                score_labels = score_labels.float()
                # Compute MSE
                mse_loss = nn.functional.mse_loss(score_preds, score_labels)
                
                # Combine losses (you can add a weight factor alpha here if desired)
                # alpha = 0.5 (equal weighting)
                loss = ctc_loss + (0.5 * mse_loss)

        if not return_dict:
            output = (logits, score_preds,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits, # We hijack `logits` to return standard CTC logits for HF Training tracking
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
