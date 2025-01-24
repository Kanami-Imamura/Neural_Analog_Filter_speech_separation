'''Implementations of Sampling-frequency-independent convolutional layers.

This code is based on https://github.com/TomohikoNakamura/sfi_convtasnet.

Copyright (c) Kanami Imamura
All rights reserved.
'''
import torch
from torch import nn

import numpy

from .enc_dec import _EncDec, multishape_conv1d, multishape_conv_transpose1d
from .naf_fb import TDNAFFilterBank, FDNAFFilterBank, TDMGFFilterBank, FDMGFFilterBank

class SFITDEncoder(_EncDec):
    '''Sampling-frequency-independent encoder class proposed in [1].
    
    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    '''
    def __init__(
        self, filterbank, nonlineartity="relu", is_pinv=False, as_conv1d=True, padding=0
    ):
        r'''
        Args:
            filterbank (:class:`Filterbank`): The filterbank to use
                as an encoder.
            is_pinv (bool): Whether to be the pseudo inverse of filterbank.
            as_conv1d (bool): Whether to behave like nn.Conv1d.
                If True (default), forwarding input with shape :math:`(batch, 1, time)`
                will output a tensor of shape :math:`(batch, freq, conv\_time)`.
                If False, will output a tensor of shape :math:`(batch, 1, freq, conv\_time)`.
            padding (int): Zero-padding added to both sides of the input.
        '''
        if not (isinstance(filterbank, TDNAFFilterBank) or isinstance(filterbank, TDMGFFilterBank)):
            raise NotImplementedError
        super(SFITDEncoder, self).__init__(filterbank, is_pinv=is_pinv)
        self.as_conv1d = as_conv1d
        self.n_feats_out = self.filterbank.n_feats_out
        self.padding = padding

        if nonlineartity is None:
            self.nonlinearity = nn.Sequential()
        elif nonlineartity == "relu":
            self.nonlinearity = nn.ReLU(True)
        else:
            raise ValueError(f'Unknown nonlinearity [{nonlineartity}]')

    @property
    def is_SFI(self):
        return True

    def prepare(self, sample_rate:int, padding: int=None):
        kernel_size, stride = self.filterbank.prepare(sample_rate)
        self.sample_rate = sample_rate
        self.stride = stride
        if padding is None:
            self.padding = (kernel_size-stride)//2
        else:
            self.padding = 0

    def change_n_samples(self, n_samples):
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        self.conv.n_samples = n_samples

    def get_n_samples(self):
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        return self.conv.n_samples

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        raise NotImplementedError
        # """Returns an :class:`~.Encoder`, pseudo inverse of a
        # :class:`~.Filterbank` or :class:`~.Decoder`."""
        # if isinstance(filterbank, Filterbank):
        #     return cls(filterbank, is_pinv=True, **kwargs)
        # elif isinstance(filterbank, Decoder):
        #     return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {"padding": self.padding}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def forward(self, waveform):
        """Convolve input waveform with the filters from a filterbank.

        Args:
            waveform (:class:`torch.Tensor`): any tensor with samples along the
                last dimension. The waveform representation with and
                batch/channel etc.. dimension.

        Returns:
            :class:`torch.Tensor`: The corresponding TF domain signal.

        Shapes
            >>> (time, ) -> (freq, conv_time)
            >>> (batch, time) -> (batch, freq, conv_time)  # Avoid
            >>> if as_conv1d:
            >>>     (batch, 1, time) -> (batch, freq, conv_time)
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> else:
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> (batch, any, dim, time) -> (batch, any, dim, freq, conv_time)
        """
        filters = self.filterbank.get_impulse_responses(self.sample_rate).unsqueeze(1)
        waveform = self.filterbank.pre_analysis(waveform)
        spec = multishape_conv1d(
            waveform,
            filters=filters,
            stride=self.stride,
            padding=self.padding,
            as_conv1d=self.as_conv1d,
        )
        return self.nonlinearity(self.filterbank.post_analysis(spec))

class SFITDDecoder(_EncDec):
    '''Sampling-frequency-independent encoder class proposed in [1].
    
    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    '''
    def __init__(self, filterbank, nonlineartity="relu", is_pinv=False, as_conv1d=True, padding=0, output_padding=0):
        r'''
        Args:
            filterbank (:class:`Filterbank`): The filterbank to use
                as an encoder.
            is_pinv (bool): Whether to be the pseudo inverse of filterbank.
            as_conv1d (bool): Whether to behave like nn.Conv1d.
                If True (default), forwarding input with shape :math:`(batch, 1, time)`
                will output a tensor of shape :math:`(batch, freq, conv\_time)`.
                If False, will output a tensor of shape :math:`(batch, 1, freq, conv\_time)`.
            padding (int): Zero-padding added to both sides of the input.
        '''
        if not (isinstance(filterbank, TDNAFFilterBank) or isinstance(filterbank, TDMGFFilterBank)):
            raise NotImplementedError
        super(SFITDDecoder, self).__init__(filterbank, is_pinv=is_pinv)
        self.as_conv1d = as_conv1d
        self.n_feats_out = self.filterbank.n_feats_out
        self.padding = padding
        self.output_padding = output_padding

        if nonlineartity is None:
            self.nonlinearity = nn.Sequential()
        elif nonlineartity == "relu":
            self.nonlinearity = nn.ReLU(True)
        else:
            raise ValueError(f'Unknown nonlinearity [{nonlineartity}]')

    def change_n_samples(self, n_samples):
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        self.conv.n_samples = n_samples

    def get_n_samples(self):
        if self._fir_computation_type != "freq_resp_samp":
            raise ValueError
        return self.conv.n_samples

    def prepare(self, sample_rate:int, padding: int=None, output_padding: int=0):
        kernel_size, stride = self.filterbank.prepare(sample_rate)
        self.sample_rate = sample_rate
        self.stride = stride
        if padding is None:
            self.padding = (kernel_size-stride)//2
        else:
            self.padding = 0
        self.output_padding = (int(output_padding),)

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        raise NotImplementedError
        # """Returns an :class:`~.Encoder`, pseudo inverse of a
        # :class:`~.Filterbank` or :class:`~.Decoder`."""
        # if isinstance(filterbank, Filterbank):
        #     return cls(filterbank, is_pinv=True, **kwargs)
        # elif isinstance(filterbank, Decoder):
        #     return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {"padding": self.padding, "output_padding": self.output_padding}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def forward(self, spec, length=None):
        """Convolve input waveform with the filters from a filterbank.

        Args:
            waveform (:class:`torch.Tensor`): any tensor with samples along the
                last dimension. The waveform representation with and
                batch/channel etc.. dimension.

        Returns:
            :class:`torch.Tensor`: The corresponding TF domain signal.

        Shapes
            >>> (time, ) -> (freq, conv_time)
            >>> (batch, time) -> (batch, freq, conv_time)  # Avoid
            >>> if as_conv1d:
            >>>     (batch, 1, time) -> (batch, freq, conv_time)
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> else:
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> (batch, any, dim, time) -> (batch, any, dim, freq, conv_time)
        """
        filters = self.filterbank.get_impulse_responses(self.sample_rate).unsqueeze(1)
        spec = self.filterbank.pre_synthesis(spec)
        wav = multishape_conv_transpose1d(
            spec,
            filters,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        wav = self.filterbank.post_synthesis(wav)
        if length is not None:
            length = min(length, wav.shape[-1])
            return wav[..., :length]
        return wav

class _SFIFDEncDec(_EncDec):
    SAMPLING_STRATEGY = ["fixed", "randomized", "completely_randomized"]
    def __init__(self, filterbank, n_samples, nonlineartity="relu", is_pinv=False, as_conv1d=True, padding=0, frequency_sampling_strategy=["fixed", "fixed"]):
        if not (isinstance(filterbank, FDNAFFilterBank) or isinstance(filterbank, FDMGFFilterBank)):
            raise NotImplementedError
        super().__init__(filterbank, is_pinv=is_pinv)
        self.as_conv1d = as_conv1d
        self.n_feats_out = self.filterbank.n_feats_out
        self.padding = padding
        
        self.n_samples = n_samples
        for i in range(len(frequency_sampling_strategy)):
            if frequency_sampling_strategy[i] not in self.SAMPLING_STRATEGY:
                raise NotImplementedError(f'Undefined frequency sampling strategy [{frequency_sampling_strategy[i]}]')
        self.frequency_sampling_strategy = frequency_sampling_strategy
        self._cache = dict()

        if nonlineartity is None:
            self.nonlinearity = nn.Sequential()
        elif nonlineartity == "relu":
            self.nonlinearity = nn.ReLU(True)
        else:
            raise ValueError(f'Unknown nonlinearity [{nonlineartity}]')

    def change_n_samples(self, n_samples):
        self.n_samples = n_samples

    @classmethod
    def pinv_of(cls, filterbank, **kwargs):
        raise NotImplementedError
        # """Returns an :class:`~.Encoder`, pseudo inverse of a
        # :class:`~.Filterbank` or :class:`~.Decoder`."""
        # if isinstance(filterbank, Filterbank):
        #     return cls(filterbank, is_pinv=True, **kwargs)
        # elif isinstance(filterbank, Decoder):
        #     return cls(filterbank.filterbank, is_pinv=True, **kwargs)

    def _compute_pinvW(self):
        device = self.filterbank.device
        kernel_size = self.filterbank.kernel_size
        sample_rate = self.sample_rate            
        P = (kernel_size-1)//2 if kernel_size%2 == 1 else kernel_size//2
        M = self.n_samples
        nyquist_rate = sample_rate / 2
        #
        strategy = self.frequency_sampling_strategy[0 if self.training else 1]
        if strategy == "fixed":
            ang_freqs = torch.linspace(0, nyquist_rate*2.0*numpy.pi, M).float().to(device)
        elif strategy == "randomized":
            ang_freqs = torch.linspace(0, nyquist_rate*2.0*numpy.pi, M).float()
            ang_freqs.requires_grad_(False)
            delta_val = ang_freqs[1] - ang_freqs[0]
            delta = torch.zeros_like(ang_freqs).uniform_(-delta_val/2, delta_val/2)
            delta.requires_grad_(False)
            if delta[0]<0:
                delta[0] = -delta[0] 
            if delta[-1]>0:
                delta[-1] = - delta[-1]
            ang_freqs = ang_freqs + delta
            ang_freqs = ang_freqs.to(device)
        elif strategy == "completely_randomized":
            ang_freqs = torch.zeros((M,), device=device).float().uniform_(0.0, nyquist_rate*2.0*numpy.pi)
            ang_freqs.requires_grad_(False)
            ang_freqs, _ = torch.sort(ang_freqs, descending=False)
        else:
            raise NotImplementedError(f'Undefined frequency sampling strategy [{strategy}]')
        normalized_ang_freqs = ang_freqs / float(sample_rate)
        if kernel_size%2 == 1:
            seq_P = torch.arange(-P, P+1).float()[None,:].to(device)
            ln_W = -normalized_ang_freqs[:,None]*seq_P # M x 2P+1
        else:
            seq_P = torch.arange(-(P-1), P+1).float()[None,:].to(device)
            ln_W = -normalized_ang_freqs[:,None]*seq_P # M x 2P
        ln_W = ln_W.to(device)
        W = torch.cat((torch.cos(ln_W), torch.sin(ln_W)), dim=0) # 2*M x 2P
        ###
        pinvW = torch.pinverse(W) # 2P x 2M
        pinvW.requires_grad_(False)
        ang_freqs.requires_grad_(False)
        return ang_freqs, pinvW
    
    def approximate_by_FIR(self):
        '''Approximate frequency responses of analog filters with those of digital filters

        Args:
            device (torch.Device): Computation device
        
        Return:
            torch.Tensor: Time-reversed impulse responses of digital filters (n_filters x filter_degree (-P to P))
        '''
        device = self.filterbank.device
        kernel_size = self.filterbank.kernel_size
        stride = self.filterbank.stride
        strategy = self.frequency_sampling_strategy[0 if self.training else 1]
        if strategy == "fixed":
            cache_tag = (self.sample_rate, kernel_size, stride)
            if cache_tag in self._cache:
                ang_freqs, pinvW = self._cache[cache_tag]
                ang_freqs = ang_freqs.detach().to(device)
                pinvW = pinvW.detach().to(device)
            else:
                ang_freqs, pinvW = self._compute_pinvW()
                self._cache[cache_tag] = (ang_freqs.detach().cpu(), pinvW.detach().cpu())
        elif "randomized" in strategy:
            ang_freqs, pinvW = self._compute_pinvW()
        else:
            raise NotImplementedError(f'Undefined frequency sampling strategy [{strategy}]')            
        ###
        resp_r, resp_i = self.filterbank.get_frequency_responses(ang_freqs) # n_filters x M
        resp = torch.cat((resp_r, resp_i), dim=1) # n_filters x 2M
        ###
        fir_coeffs = (pinvW[None,:,:] @ resp[:,:,None])[:,:,0] # n_filters x 2P
        kernel_size = int(kernel_size / 2) * 2
        return fir_coeffs[:,torch.arange(kernel_size-1,-1,-1)] # time-reversed impulse response
    
class SFIFDEncoder(_SFIFDEncDec):
    '''Sampling-frequency-independent encoder class proposed in [1].
    
    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    '''
    def __init__(self, filterbank, n_samples, nonlineartity="relu", is_pinv=False, as_conv1d=True, padding=0, frequency_sampling_strategy=["fixed", "fixed"]):
        r'''
        Args:
            filterbank (:class:`Filterbank`): The filterbank to use
                as an encoder.
            is_pinv (bool): Whether to be the pseudo inverse of filterbank.
            as_conv1d (bool): Whether to behave like nn.Conv1d.
                If True (default), forwarding input with shape :math:`(batch, 1, time)`
                will output a tensor of shape :math:`(batch, freq, conv\_time)`.
                If False, will output a tensor of shape :math:`(batch, 1, freq, conv\_time)`.
            padding (int): Zero-padding added to both sides of the input.
        '''
        super().__init__(filterbank, n_samples, nonlineartity, is_pinv, as_conv1d, padding, frequency_sampling_strategy)

    def prepare(self, sample_rate:int, padding: int=None):
        kernel_size, stride = self.filterbank.prepare(sample_rate)
        self.sample_rate = sample_rate
        self.stride = stride
        if padding is None:
            self.padding = (kernel_size-stride)//2
        else:
            self.padding = 0

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {"n_samples": self.n_samples, "padding": self.padding}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def forward(self, waveform):
        """Convolve input waveform with the filters from a filterbank.

        Args:
            waveform (:class:`torch.Tensor`): any tensor with samples along the
                last dimension. The waveform representation with and
                batch/channel etc.. dimension.

        Returns:
            :class:`torch.Tensor`: The corresponding TF domain signal.

        Shapes
            >>> (time, ) -> (freq, conv_time)
            >>> (batch, time) -> (batch, freq, conv_time)  # Avoid
            >>> if as_conv1d:
            >>>     (batch, 1, time) -> (batch, freq, conv_time)
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> else:
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> (batch, any, dim, time) -> (batch, any, dim, freq, conv_time)
        """
        filters = self.approximate_by_FIR().unsqueeze(1)
        waveform = self.filterbank.pre_analysis(waveform)
        spec = multishape_conv1d(
            waveform,
            filters=filters,
            stride=self.stride,
            padding=self.padding,
            as_conv1d=self.as_conv1d,
        )
        return self.nonlinearity(self.filterbank.post_analysis(spec))
    
class SFIFDDecoder(_SFIFDEncDec):
    '''Sampling-frequency-independent decoder class proposed in [1].
    
    [1] K. Saito, T. Nakamura, K. Yatabe, and H. Saruwatari, ``Sampling-frequency-independent convolutional layer and its application to audio source separation,'' IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 30, pp. 2928--2943, Sep. 2022.
    '''
    def __init__(self, filterbank, n_samples, nonlineartity="relu", is_pinv=False, as_conv1d=True, padding=0, output_padding=0, frequency_sampling_strategy=["fixed", "fixed"]):
        r'''
        Args:
            filterbank (:class:`Filterbank`): The filterbank to use
                as an encoder.
            is_pinv (bool): Whether to be the pseudo inverse of filterbank.
            as_conv1d (bool): Whether to behave like nn.Conv1d.
                If True (default), forwarding input with shape :math:`(batch, 1, time)`
                will output a tensor of shape :math:`(batch, freq, conv\_time)`.
                If False, will output a tensor of shape :math:`(batch, 1, freq, conv\_time)`.
            padding (int): Zero-padding added to both sides of the input.
        '''
        super().__init__(filterbank, n_samples, nonlineartity, is_pinv, as_conv1d, padding, frequency_sampling_strategy)
        self.output_padding = output_padding

    def prepare(self,sample_rate, padding: int=None, output_padding: int=0):
        kernel_size, stride = self.filterbank.prepare(sample_rate)
        self.sample_rate = sample_rate
        self.stride = stride
        if padding is None:
            self.padding = (kernel_size-stride)//2
        else:
            self.padding = padding
        self.output_padding = int(output_padding)

    def get_config(self):
        """ Returns dictionary of arguments to re-instantiate the class."""
        config = {"n_samples": self.n_samples, "padding": self.padding, "output_padding": self.output_padding}
        base_config = self.filterbank.get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def forward(self, spec, length=None):
        """Convolve input waveform with the filters from a filterbank.

        Args:
            waveform (:class:`torch.Tensor`): any tensor with samples along the
                last dimension. The waveform representation with and
                batch/channel etc.. dimension.

        Returns:
            :class:`torch.Tensor`: The corresponding TF domain signal.

        Shapes
            >>> (time, ) -> (freq, conv_time)
            >>> (batch, time) -> (batch, freq, conv_time)  # Avoid
            >>> if as_conv1d:
            >>>     (batch, 1, time) -> (batch, freq, conv_time)
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> else:
            >>>     (batch, chan, time) -> (batch, chan, freq, conv_time)
            >>> (batch, any, dim, time) -> (batch, any, dim, freq, conv_time)
        """
        filters = self.approximate_by_FIR().unsqueeze(1)
        spec = self.filterbank.pre_synthesis(spec)
        wav = multishape_conv_transpose1d(
            spec,
            filters,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        wav = self.filterbank.post_synthesis(wav)
        if length is not None:
            length = min(length, wav.shape[-1])
            return wav[..., :length]
        return wav