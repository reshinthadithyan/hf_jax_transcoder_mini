from transformers.models.bart.modeling_flax_bart import BartConfig, FlaxBartModule
import jax
import jax.numpy as jnp
import flax.linen as nn


class FlaxBartEncoderMLM(FlaxBartModule):
  config : BartConfig 
  def setup(self):
    self.lm_head = nn.Dense(
            2,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.init_std, self.dtype),
        )
    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        deterministic: bool = True,
        masked_lm_flag : bool = False):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                deterministic=deterministic)
        if masked_lm_flag:
            encoder_outputs = nn.softmax(encoder_outputs)
            mlm_logits = self.lm_head(encoder_outputs)
            return mlm_logits
        else:    
            return encoder_outputs
if __name__ == "__main__":
    from jax import random
    from flax import linen as nn
    import jax.numpy as jnp

    test_dense = nn.Dense(5)
    variables = test_dense.init(random.PRNGKey(0), jnp.ones((5,5)))

    y = test_dense.apply(variables, jnp.ones((5,5)))
    print(y)