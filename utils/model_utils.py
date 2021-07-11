from transformers.models.bart.modeling_flax_bart import BartConfig, FlaxBartModule
from transformers.models.bart.modeling_flax_bart import FlaxBartEncoder,FlaxBartDecoder,FlaxBartPreTrainedModel
from transformers.modeling_flax_outputs import FlaxMaskedLMOutput
from transformers.file_utils import requires_backends
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.random import PRNGKey

class FlaxBartModel:
    def __init__(self, *args, **kwargs):
        requires_backends(self, ["flax"])

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requires_backends(cls, ["flax"])

class FlaxBartCrossLM(FlaxBartModel):
    """Main CrossLingual Model Class inheriting from FlaxBartModel"""
    config: BartConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    module_class = FlaxBartModule

    def setup(self):
        # self.shared = nn.Embed(
        #     self.config.vocab_size,
        #     self.config.d_model,
        #     embedding_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        #     dtype=self.dtype,
        # )
        #self.encoder = FlaxBartEncoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        #self.decoder = FlaxBartDecoder(self.config, dtype=self.dtype, embed_tokens=self.shared)
        self.model = FlaxBartModel(self.config,dtype=getattr(jnp, self.config.dtype))
        self.init_lm_head = nn.Dense(
             self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),   
        )
        self.lm_classifier = nn.Dense(
             self.config.vocab_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range, self.dtype),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def _get_encoder_module(self):
        return self.model.encoder

    def _get_decoder_module(self):
        return self.model.decoder
    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        decoder_attention_mask,
        position_ids,
        decoder_position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        train: bool = False,
        params: dict = None,
        dropout_rng: PRNGKey = None,

    ):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            deterministic=deterministic,
            train = train,
            params = params,
            dropout_rng = dropout_rng
        )
        encoder_outputs = self.dropout(encoder_outputs,deterministic=deterministic)
        output = nn.gelu(self.init_lm_head(encoder_outputs))
        output = self.dropout(output,deterministic=deterministic)

        logits = self.lm_classifier(output)
        
        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


if __name__ == "__main__":
    from jax import random
    from flax import linen as nn
    import jax.numpy as jnp

    test_dense = nn.Dense(5)
    variables = test_dense.init(random.PRNGKey(0), jnp.ones((5,5)))

    y = test_dense.apply(variables, jnp.ones((5,5)))
    print(y)