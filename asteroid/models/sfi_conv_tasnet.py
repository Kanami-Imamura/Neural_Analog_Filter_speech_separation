'''Implementations of SFI Conv-TasNet.

This code is based on https://github.com/asteroid-team/asteroid.

Copyright (c) Kanami Imamura
All rights reserved.
'''

from ..asteroid_filterbanks import make_enc_dec
from ..masknn import TDConvNet
from .base_models import BaseEncoderMaskerDecoder
import warnings


class SFIConvTasNet(BaseEncoderMaskerDecoder):
    '''SFI Conv-TasNet proposed in [1].
    
    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    '''
    def __init__(
        self,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        skip_chan=128,
        conv_kernel_size=3,
        norm_type="gLN",
        mask_act="sigmoid",
        in_chan=None,
        causal=False,
        fb_name="free",
        kernel_size=64,
        n_filters=512,
        stride=32,
        encoder_activation=None,
        sample_rate=32000,
        n_samples=None,
        **fb_kwargs,
    ):
        encoder, decoder = make_enc_dec(
            fb_name,
            kernel_size=kernel_size,
            n_filters=n_filters,
            stride=stride,
            sample_rate=sample_rate,
            n_samples=n_samples,
            **fb_kwargs,
        )
        n_feats = encoder.n_feats_out
        if in_chan is not None:
            assert in_chan == n_feats, (
                "Number of filterbank output channels"
                " and number of input channels should "
                "be the same. Received "
                f"{n_feats} and {in_chan}"
            )
        if causal and norm_type not in ["cgLN", "cLN"]:
            norm_type = "cLN"
            warnings.warn(
                "In causal configuration cumulative layer normalization (cgLN)"
                "or channel-wise layer normalization (chanLN)  "
                f"must be used. Changing {norm_type} to cLN"
            )

        masker = TDConvNet(
            n_feats,
            n_src,
            out_chan=out_chan,
            n_blocks=n_blocks,
            n_repeats=n_repeats,
            bn_chan=bn_chan,
            hid_chan=hid_chan,
            skip_chan=skip_chan,
            conv_kernel_size=conv_kernel_size,
            norm_type=norm_type,
            mask_act=mask_act,
            causal=causal,
        )
        super().__init__(encoder, masker, decoder, encoder_activation=encoder_activation)


    def inference(self, wav, sample_rate):
        # adjusting kernel_size,stride
        self.encoder.prepare(sample_rate)
        self.decoder.prepare(sample_rate)
        # separate
        return self.forward(wav)

    def get_model_args(self):
        fb_config = self.encoder.get_config()
        masknet_config = self.masker.get_config()
        # Assert both dict are disjoint
        if not all(k not in fb_config for k in masknet_config):
            raise AssertionError(
                "Filterbank and Mask network config share common keys. Merging them is not safe."
            )
        # Merge all args under model_args.
        model_args = {
            **fb_config,
            **masknet_config,
            "encoder_activation": self.encoder_activation,
        }
        return model_args