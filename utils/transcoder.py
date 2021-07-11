from transformers.models.bart.modeling_flax_bart import FlaxBartEncoder,FlaxBartDecoder, FlaxBartModule,FlaxBartModel,FlaxBartPreTrainedModel
from transformers.modeling_flax_outputs import FlaxSeq2SeqModelOutput,FlaxMaskedLMOutput
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from transformers import BartConfig#,FlaxSeq2SeqModelOutput,FlaxMaskedLMOutput
from transformers.utils.dummy_flax_objects import FlaxPreTrainedModel#, FlaxRobertaModel, FlaxRobertaPreTrainedModel
from typing import Callable, Optional, Tuple
from jax.random import PRNGKey

from utils.transcoder_module import FlaxBartTranscoderPreTrainedModel

class FlaxTrancoderBartModule(nn.Module):
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.shared = nn.Embed(
            self.config.vocab_size,
            self.config.d_model,
            embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
            dtype=self.dtype,
        )

        self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)#FlaxRobertaModel.from_pretrained(r"reshinthadith/transcoder-js-cs")#
        self.decoder = FlaxBartDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.init_lm_head = nn.Dense(
             self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),   
        )
        self.lm_classifier = nn.Dense(
             self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
        self.dropout = nn.Dropout(rate=self.config.dropout)


    def _get_encoder_module(self):
        return self.encoder

    def _get_decoder_module(self):
        return self.decoder

    # def __call__(
    #     self,
    #     input_ids,
    #     attention_mask,
    #     decoder_input_ids,
    #     decoder_attention_mask,
    #     position_ids,
    #     decoder_position_ids,
    #     output_attentions: bool = False,
    #     output_hidden_states: bool = False,
    #     return_dict: bool = True,
    #     deterministic: bool = True,
    #     encode_only: bool = False
    # ):
    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        decoder_input_ids: Optional[jnp.ndarray] = None,
        decoder_attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        decoder_position_ids: Optional[jnp.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,
        encode_only: bool = True,
        deterministic:bool = False
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            deterministic=deterministic,
        )
        if encode_only:
            print("===================enocde_only=====================")
            encoder_output_state = encoder_outputs["last_hidden_state"]
            encoder_output_state = self.dropout(encoder_output_state,deterministic=deterministic)
            output = nn.gelu(self.init_lm_head(encoder_output_state))
            output = self.dropout(output,deterministic=deterministic)

            logits = self.lm_classifier(output)
            
            return FlaxMaskedLMOutput(
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=decoder_position_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return FlaxSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

class FlaxTranscoderBartModel(FlaxBartPreTrainedModel):
    """FlaxTranscoderBartModel for use with Roberta Encoder"""
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    module_class = FlaxTrancoderBartModule
