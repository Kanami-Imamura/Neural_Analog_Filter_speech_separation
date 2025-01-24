'''Implementations of neural analog filters and modulated Gaussian filters.

Implementations of the modulated Gaussian filters is based on https://github.com/TomohikoNakamura/sfi_convtasnet.

Copyright (c) Kanami Imamura
All rights reserved.
'''
import torch
from torch import nn

import functools
import numpy
import librosa

from .enc_dec import Filterbank

def resample(ir, org_sr, target_sr):
    return librosa.core.resample(ir, org_sr, target_sr, res_type="kaiser_best", fix=True, scale=False)

def erb_to_hz(x):
    return (numpy.exp(x/9.265)-1)*24.7*9.265

def hz_to_erb(x):
    return numpy.log(1+x/(24.7*9.265))*9.265

class TDNAFFilterBank(Filterbank):
    '''Nueral analog filter (NAF) for time-domain sampling-frequency-independent convolutional layer proposed in [1].
    
   [1] Kanami Imamura, Tomohiko Nakamura, Kohei Yatabe, and Hiroshi Saruwatari, ``Neural analog filter for sampling-frequency-independent convolutional layer," APSIPA Transactions on Signal and Information Processing, vol. 13, no. 1, e28, Nov. 2024.
    '''
    def __init__(
        self, n_filters, kernel_size, stride, sample_rate, ch_list=[224,224], n_RFFs=128, train_RFF=True, nonlinearity="relu", use_layer_norm=True
    ):
        '''
            Args:
                n_filters (int): Number of filters.
                train_kernel_size (int): Length of the filters (i.e the window).
                train_stride (int, optional): Stride of the convolution (hop size). If None
                    (default), set to ``kernel_size // 2``.
                train_sample_rate (float): Sample rate of the expected audio.
                    Defaults to 32000.
                ch_list: channels of MLP.
                n_RFFs: length of RFF.
                train_RFF: bool
                nonliniearity: 
                use_layer_norm: bool
        '''
        super().__init__(n_filters, kernel_size, stride=stride, sample_rate=sample_rate)
        if nonlinearity == "relu":
            NonlinearityClass = functools.partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.train_kernel_size = kernel_size
        self.stride = stride
        self.train_stride = stride
        self.sample_rate = sample_rate
        self.train_sample_rate = sample_rate
        self.ch_list = ch_list
        self.n_RFFs = n_RFFs
        self.train_RFF = train_RFF
        self.nonlinearity = nonlinearity
        self.use_layer_norm = use_layer_norm
        layers = []
        in_ch_list = [n_RFFs*2 if n_RFFs > 0 else 1] + [i for i in ch_list]
        out_ch_list = [i for i in ch_list] + [n_filters]
        for (i, in_ch), out_ch in zip(enumerate(in_ch_list), out_ch_list):
            layers.append(nn.Conv1d(in_ch, out_ch, 1))
            if i < len(in_ch_list) - 1:
                if use_layer_norm:
                    layers.append(nn.GroupNorm(1, out_ch))
                layers.append(NonlinearityClass())
        self.implicit_filter = nn.Sequential(*layers) # Conversion from angular frequency to frequency component (real and imag) at that frequency

        if n_RFFs > 0:
            self.RFF_param = nn.Parameter(
                torch.zeros((n_RFFs,), dtype=torch.float).normal_(0.0, 2.0 * numpy.pi * 10.0), requires_grad=train_RFF
            )
        else:
            self.RFF_param = None

        def set_zero_bias(m):
            if isinstance(m, nn.Conv1d):
                m.bias.data.fill_(0.0)

        self.implicit_filter.apply(set_zero_bias)

    @property
    def device(self):
        return self.implicit_filter[0].weight.device
    
    def prepare(self, sample_rate):
        self.sample_rate = sample_rate
        self.kernel_size = int(self.train_kernel_size * sample_rate / self.train_sample_rate)
        self.stride = int(self.train_stride * sample_rate / self.train_sample_rate)
        return self.kernel_size, self.stride
    
    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class.
        Needs to be subclassed if the filterbanks takes additional arguments
        than ``n_filters`` ``kernel_size`` ``stride`` and ``sample_rate``.
        """
        config = {
            "fb_name": self.__class__.__name__,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "ch_list": self.ch_list,
            "n_RFFs": self.n_RFFs,
            "train_RFF": self.train_RFF,
            "nonlinearity": self.nonlinearity,
            "use_layer_norm": self.use_layer_norm,
        }
        return config
    
    def _get_ir(self, normalized_time):
        '''
        Args:
            normalized_time (torch.Tensor): Normalized time (time)
        
        Return:
            torch.Tensor: Discrete-time impulse responses (n_filters x time)
        '''
        if self.RFF_param is not None:
            RFF = self.RFF_param[:,None] @ normalized_time[None,:] # n_RFFs x time
            RFF = torch.cat((RFF.sin(), RFF.cos()), dim=0) # n_RFFs*2 x time
            ir = self.implicit_filter(RFF[None,:,:]) # 1 x n_filters x time
        else:
            ir = self.implicit_filter(normalized_time[None,None,:]) # 1 x n_filters x time
        return ir.view(*(ir.shape[1:]))

    def get_impulse_responses(self, sample_rate: int):
        use_oversampling = True
        if not self.training and hasattr(self, "use_oversampling"):
            use_oversampling = self.use_oversampling
        if use_oversampling and not self.training:
            normalized_time = torch.linspace(-1.0, 1.0, self.train_kernel_size, device=self.device, requires_grad=False) # time #temporary
            ir = self._get_ir(normalized_time)
            resampled_ir = resample(ir.cpu().detach().numpy(), int(self.train_sample_rate), int(sample_rate))
            ir = torch.tensor(resampled_ir).float().to(self.device)
        else:
            normalized_time = torch.linspace(-1.0, 1.0, self.kernel_size, device=self.device, requires_grad=False) # time
            ir = self._get_ir(normalized_time)
        return ir

    def get_impulse_responses_oversampling(self, sample_rate: int, kernel_size: int):
        normalized_time = torch.linspace(-1.0, 1.0, int(160), device=self.device, requires_grad=False) # time
        ir = self._get_ir(normalized_time)
        resampled_ir = resample(ir.cpu().numpy(), int(self.train_sample_rate.item()), int(sample_rate)) #resampling
        ir = torch.tensor(resampled_ir).float().to(self.device)
        return ir

class FDNAFFilterBank(Filterbank):
    '''Nueral analog filter (NAF) for frequency-domain sampling-frequency-independent convolutional layer proposed in [1].
    
    [1] Kanami Imamura, Tomohiko Nakamura, Kohei Yatabe, and Hiroshi Saruwatari, ``Neural analog filter for sampling-frequency-independent convolutional layer," APSIPA Transactions on Signal and Information Processing, vol. 13, no. 1, e28, Nov. 2024.
    '''
    def __init__(self, n_filters, kernel_size, stride, sample_rate, max_freq=16000, ch_list=[224,224], n_RFFs=128, train_RFF=True, nonlinearity="relu", use_layer_norm=True, gain_normalization=False):
        '''
            Args:
                n_filters (int): Number of filters.
                train_kernel_size (int): Length of the filters (i.e the window).
                train_stride (int, optional): Stride of the convolution (hop size). If None
                    (default), set to ``kernel_size // 2``.
                train_sample_rate (float): Sample rate of the expected audio.
                    Defaults to 32000.
                max_freq (float): Max. of frequency (i.e., Nyquist frequency of training data)
                ch_list: channels of MLP.
                n_RFFs: length of RFF.
                train_RFF: bool
                nonliniearity: 
                use_layer_norm: bool
                gain_normlization: bool
        '''
        super().__init__(n_filters, kernel_size, stride, sample_rate)
        if nonlinearity == "relu":
            NonlinearityClass = functools.partial(nn.ReLU, inplace=True)
        else:
            raise NotImplementedError
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.train_kernel_size = kernel_size
        self.stride = stride
        self.train_stride = stride
        self.sample_rate = sample_rate
        self.train_sample_rate = sample_rate

        self.max_freq = max_freq
        self.ch_list = ch_list
        self.n_RFFs = n_RFFs
        self.train_RFF = train_RFF
        self.nonlinearity = nonlinearity
        self.use_layer_norm = use_layer_norm
        self.gain_normalization = gain_normalization
        self.use_RFFs = n_RFFs > 0
        
        self.register_buffer("max_ang_freq", torch.tensor(max_freq*2.0*numpy.pi))
        layers = []
        in_ch_list = [n_RFFs*2 if self.use_RFFs else 1] + [i for i in ch_list]
        out_ch_list = [i for i in ch_list] + [n_filters*2]
        for (i, in_ch), out_ch in zip(enumerate(in_ch_list), out_ch_list):
            layers.append(nn.Conv1d(in_ch, out_ch, 1))
            if i < len(in_ch_list) - 1:
                if use_layer_norm:
                    layers.append(nn.GroupNorm(1, out_ch))
                layers.append(NonlinearityClass())
        self.implicit_filter = nn.Sequential(*layers) # Conversion from angular frequency to frequency component (real and imag) at that frequency

        if self.use_RFFs:
            self.RFF_param = nn.Parameter(
                torch.zeros((n_RFFs,), dtype=torch.float).normal_(0.0, 2.0 * numpy.pi * 10.0), requires_grad=train_RFF
            )
        
        def set_zero_bias(m):
            if isinstance(m, nn.Conv1d):
                m.bias.data.fill_(0.0)

        self.implicit_filter.apply(set_zero_bias)
        self.use_ideal_low_pass_filter = True

    @property
    def device(self):
        return self.implicit_filter[0].weight.device
    
    def prepare(self, sample_rate):
        self.sample_rate = sample_rate
        self.kernel_size = int(self.train_kernel_size * sample_rate / self.train_sample_rate)
        self.stride = int(self.train_stride * sample_rate / self.train_sample_rate)
        return self.kernel_size, self.stride
    
    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class.
        Needs to be subclassed if the filterbanks takes additional arguments
        than ``n_filters`` ``kernel_size`` ``stride`` and ``sample_rate``.
        """
        config = {
            "fb_name": self.__class__.__name__,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "max_freq": self.max_freq,
            "ch_list": self.ch_list,
            "n_RFFs": self.n_RFFs,
            "train_RFF": self.train_RFF,
            "nonlinearity": self.nonlinearity,
            "use_layer_norm": self.use_layer_norm,
            "gain_normalization": self.gain_normalization,
        }
        return config
    
    def normalize_gain(self, freq_resps):
        '''Normalize frequency responses of analog filters so that their means equal unity.

        Args:
            freq_resps (torch.Tensor): 1 x n_filters (real, imag) x n_angfreqs
        
        Return:
            torch.Tensor: Frequency responses
        '''
        N = freq_resps.shape[1]//2
        freq_resps_reim = freq_resps.reshape(freq_resps.shape[0], N, 2, freq_resps.shape[2]) # 1 x n_filters//2 x 2 x n_angfreqs
        gain = (freq_resps_reim**2).sum(dim=2, keepdim=True).mean(dim=3, keepdim=True).sqrt() # 1 x n_filters//2 x 1 x 1
        freq_resps_reim_normalized = freq_resps_reim/gain # 1 x n_filters//2 x 2 x n_angfreqs
        return freq_resps_reim_normalized.reshape(*freq_resps.shape)

    def get_frequency_responses(self, omega: torch.Tensor):
        '''

        Args:
            omega (torch.Tensor): (Unnormalized) angular frequencies (n_angfreqs)

        Return:
            Tuple[torch.Tensor,torch.Tensor]: Real and imaginary parts of frequency characteristics (pair of n_filters x n_angfreqs as tuple)
        '''
        omega = omega / self.max_ang_freq # n_angfreqs
        if self.use_RFFs:
            x = self.RFF_param[:,None] @ omega[None,:] # n_RFFs x n_angfreqs
            x = torch.cat((x.cos(), x.sin()), dim=0) # n_RFFs*2 x n_angfreqs
        else:
            x = omega[None,:] # 1 x n_angfreqs
        freq_resps = self.implicit_filter(x[None,:,:]) # 1 x n_RFFs*2 (or 1 (ang. freq.)) x n_angfreqs -> 1 x n_filters*2 x n_angfreqs

        # Gain normalize
        if self.gain_normalization:
            freq_resps = self.normalize_gain(freq_resps)

        # Apply ideal low pass filter
        if not self.training and omega.max() > 1.0 and self.use_ideal_low_pass_filter:
            freq_resps *= (omega <= 1.0).float()[None,None,:]

        return freq_resps[0, :self.n_filters, :], freq_resps[0, self.n_filters:, :]

class TDMGFFilterBank(Filterbank):
    '''Modulated Gaussian filter (MGF) filterbank for time-domain filter design.'''
    def __init__(self, n_filters, kernel_size, stride, sample_rate, init_type="erb", min_bw=1.0*2.0*numpy.pi, initial_freq_range=[50.0, 32000/2], one_sided=False, init_sigma=20.0*2.0*numpy.pi, trainable=True):
        '''
            Args:
                n_filters (int): Number of filters.
                kernel_size (int): Length of the filters (i.e the window).
                stride (int): Stride of the convolution (hop size). If None
                    (default), set to ``kernel_size // 2``.
                sample_rate (int): Sample rate of the expected audio.
                    Defaults to 32000.
                init_type (str): Initialization type of center frequencies.
                    If "erb", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the ERB scale.
                    If "linear", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the linear frequency scale.
                min_bw (float): Minimum bandwidth in radian
                initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
                one_sided (bool): If True, ignore the term in the negative frequency region. If False, the corresponding impulse response is modulated Gaussian window.
                init_sigma (float): Initial sigma.
                trainable: If True, train parameters with DNN.
        '''
        super().__init__(n_filters, kernel_size, stride, sample_rate)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.train_kernel_size = kernel_size
        self.stride = stride
        self.train_stride = stride
        self.sample_rate = sample_rate
        self.train_sample_rate = sample_rate

        self.init_type = init_type
        self.min_bw = min_bw
        self.initial_freq_range = initial_freq_range
        self.init_sigma = init_sigma
        self.trainable = trainable
        # Initialization of MGF
        lf, hf = initial_freq_range
        if init_type == "linear":
            mus = numpy.linspace(lf, hf, n_filters)*2.0*numpy.pi
            sigma2s = init_sigma**2 * numpy.ones((n_filters,), dtype='f')
        elif init_type == "erb":
            erb_mus = numpy.linspace(hz_to_erb(lf), hz_to_erb(hf), n_filters)
            mus = erb_to_hz(erb_mus)*2.0*numpy.pi
            sigma2s = init_sigma**2 * numpy.ones((n_filters,), dtype='f')
        else:
            raise ValueError
        self.min_ln_sigma2s = numpy.log(min_bw**2)

        self.mus = nn.Parameter(torch.from_numpy(mus).float(), requires_grad=trainable)
        self._ln_sigma2s = nn.Parameter(torch.from_numpy(numpy.log(sigma2s)).float().clamp(min=self.min_ln_sigma2s), requires_grad=trainable)
        self.phase = nn.Parameter(torch.zeros((n_filters,), dtype=torch.float), requires_grad=trainable)
        self.phase.data.uniform_(0.0, numpy.pi)
        self.one_sided = one_sided

    @property
    def sigma2s(self):
        return self._ln_sigma2s.clamp(min=self.min_ln_sigma2s).exp()

    @property
    def device(self):
        return self.mus.device

    def extra_repr(self):
        s = f'n_filters={int(self.mus.shape[0])}, one_sided={self.one_sided}'
        return s.format(**self.__dict__)
    
    def prepare(self, sample_rate):
        self.sample_rate = sample_rate
        self.kernel_size = int(self.train_kernel_size * sample_rate / self.train_sample_rate)
        self.stride = int(self.train_stride * sample_rate / self.train_sample_rate)
        return self.kernel_size, self.stride
    
    def get_config(self):
        '''Returns dictionary of arguments to re-instantiate the class.
        Needs to be subclassed if the filterbanks takes additional arguments
        than ``n_filters`` ``kernel_size`` ``stride`` and ``sample_rate``.
        '''
        config = {
            "fb_name": self.__class__.__name__,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "init_type": self.init_type,
            "min_bw": self.min_bw,
            "initial_freq_range": self.initial_freq_range,
            "init_sigma": self.init_sigma,
            "trainable": self.trainable,
        }
        return config
    
    def _get_ir(self, normalized_time):
        c = 2.0*(2.0*numpy.pi*self.sigma2s[:,None]).sqrt()*(-self.sigma2s[:,None]*(normalized_time**2)/2.0).exp()
        filter_coeffs = c*(self.mus[:,None] @ normalized_time + self.phase[:,None]).cos() # n_filters x kernel_size
        return filter_coeffs
    
    def get_impulse_responses(self, sample_rate: int):
        self.prepare(sample_rate=sample_rate)
        center_freqs_in_hz = self.mus/(2.0*numpy.pi)
        
        if self.one_sided:
            raise NotImplementedError
        else:
            t = (torch.arange(0.0, self.kernel_size, 1).type_as(center_freqs_in_hz)/sample_rate)
            t = (t - t.mean())[None,:]
            ir = self._get_ir(t)
            # check whether the center frequencies are below Nyquist rate
            if self.train_sample_rate > sample_rate:
                mask = center_freqs_in_hz <= sample_rate/2
            ###
            if self.train_sample_rate > sample_rate:
                ir = ir * mask[:,None]
        return ir[:,torch.arange(self.kernel_size-1,-1,-1)]

class FDMGFFilterBank(Filterbank):
    '''Modulated Gaussian filter (MGF) filterbank for frequency-domain filter design.'''
    def __init__(self, n_filters, kernel_size, stride, sample_rate, init_type="erb", min_bw=1.0*2.0*numpy.pi, initial_freq_range=[50.0, 32000/2], one_sided=False, init_sigma=20.0*2.0*numpy.pi, trainable=True):
        '''
            Args:
                n_filters (int): Number of filters.
                kernel_size (int): Length of the filters (i.e the window).
                stride (int): Stride of the convolution (hop size). If None
                    (default), set to ``kernel_size // 2``.
                sample_rate (int): Sample rate of the expected audio.
                    Defaults to 32000.
                init_type (str): Initialization type of center frequencies.
                    If "erb", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the ERB scale.
                    If "linear", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the linear frequency scale.
                min_bw (float): Minimum bandwidth in radian
                initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
                one_sided (bool): If True, ignore the term in the negative frequency region. If False, the corresponding impulse response is modulated Gaussian window.
                init_sigma (float): Initial sigma.
                trainable: If True, train parameters with DNN.
        '''
        super().__init__(n_filters, kernel_size, stride, sample_rate)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.train_kernel_size = kernel_size
        self.stride = stride
        self.train_stride = stride
        self.sample_rate = sample_rate
        self.train_sample_rate = sample_rate

        self.init_type = init_type
        self.min_bw = min_bw
        self.initial_freq_range = initial_freq_range
        self.init_sigma = init_sigma
        self.trainable = trainable
        # Initialization of MGF
        lf, hf = initial_freq_range
        if init_type == "linear":
            mus = numpy.linspace(lf, hf, n_filters)*2.0*numpy.pi
            sigma2s = init_sigma**2 * numpy.ones((n_filters,), dtype='f')
        elif init_type == "erb":
            erb_mus = numpy.linspace(hz_to_erb(lf), hz_to_erb(hf), n_filters)
            mus = erb_to_hz(erb_mus)*2.0*numpy.pi
            sigma2s = init_sigma**2 * numpy.ones((n_filters,), dtype='f')
        else:
            raise ValueError
        self.min_ln_sigma2s = numpy.log(min_bw**2)

        self.mus = nn.Parameter(torch.from_numpy(mus).float(), requires_grad=trainable)
        self._ln_sigma2s = nn.Parameter(torch.from_numpy(numpy.log(sigma2s)).float().clamp(min=self.min_ln_sigma2s), requires_grad=trainable)
        self.phase = nn.Parameter(torch.zeros((n_filters,), dtype=torch.float), requires_grad=trainable)
        self.phase.data.uniform_(0.0, numpy.pi)
        self.one_sided = one_sided

    @property
    def sigma2s(self):
        return self._ln_sigma2s.clamp(min=self.min_ln_sigma2s).exp()
    
    @property
    def device(self):
        return self.mus.device
    
    def extra_repr(self):
        s = f'n_filters={int(self.mus.shape[0])}, one_sided={self.one_sided}'
        return s.format(**self.__dict__)
    
    def prepare(self, sample_rate):
        self.sample_rate = sample_rate
        self.kernel_size = int(self.train_kernel_size * sample_rate / self.train_sample_rate)
        self.stride = int(self.train_stride * sample_rate / self.train_sample_rate)
        return self.kernel_size, self.stride
    
    def get_config(self):
        """Returns dictionary of arguments to re-instantiate the class.
        Needs to be subclassed if the filterbanks takes additional arguments
        than ``n_filters`` ``kernel_size`` ``stride`` and ``sample_rate``.
        """
        config = {
            "fb_name": self.__class__.__name__,
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "sample_rate": self.sample_rate,
            "init_type": self.init_type,
            "min_bw": self.min_bw,
            "initial_freq_range": self.initial_freq_range,
            "init_sigma": self.init_sigma,
            "trainable": self.trainable,
        }
        return config

    def get_frequency_responses(self, omega: torch.Tensor):
        if self.one_sided:
            resp_abs = torch.exp(-(omega[None,:] - self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # n_filters x n_angfreqs
            resp_r = resp_abs * self.phase.cos()[:,None]
            resp_i = resp_abs * self.phase.sin()[:,None]
        else:
            resp_abs = torch.exp(-(omega[None,:] - self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # n_filters x n_angfreqs
            resp_abs2 = torch.exp(-(omega[None,:] + self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # to ensure filters whose impulse responses are real.
            resp_r = resp_abs * self.phase.cos()[:,None] + resp_abs2 * ((-self.phase).cos()[:,None])
            resp_i = resp_abs * self.phase.sin()[:,None] + resp_abs2 * ((-self.phase).sin()[:,None])
        return resp_r, resp_i
