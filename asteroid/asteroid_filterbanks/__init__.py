from .enc_dec import Filterbank, Encoder, Decoder
from .analytic_free_fb import AnalyticFreeFB
from .free_fb import FreeFB
from .param_sinc_fb import ParamSincFB
from .stft_fb import STFTFB
from .torch_stft_fb import TorchSTFTFB
from .griffin_lim import griffin_lim, misi
from .multiphase_gammatone_fb import MultiphaseGammatoneFB
from .melgram_fb import MelGramFB
from .naf_fb import TDNAFFilterBank, FDNAFFilterBank, TDMGFFilterBank, FDMGFFilterBank
from .sfi_enc_dec import SFITDEncoder, SFITDDecoder, SFIFDEncoder, SFIFDDecoder

__version__ = "0.4.0"


def make_enc_dec(
    fb_name,
    n_filters,
    kernel_size,
    stride=None,
    sample_rate=8000.0,
    who_is_pinv=None,
    padding=0,
    output_padding=0,
    n_samples=0,
    **kwargs,
):
    """Creates congruent encoder and decoder from the same filterbank family.

    Args:
        fb_name (str, className): Filterbank family from which to make encoder
            and decoder. To choose among [``'free'``, ``'analytic_free'``,
            ``'param_sinc'``, ``'stft'``]. Can also be a class defined in a
            submodule in this subpackade (e.g. :class:`~.FreeFB`).
        n_filters (int): Number of filters.
        kernel_size (int): Length of the filters.
        stride (int, optional): Stride of the convolution.
            If None (default), set to ``kernel_size // 2``.
        sample_rate (float): Sample rate of the expected audio.
            Defaults to 8000.0.
        who_is_pinv (str, optional): If `None`, no pseudo-inverse filters will
            be used. If string (among [``'encoder'``, ``'decoder'``]), decides
            which of ``Encoder`` or ``Decoder`` will be the pseudo inverse of
            the other one.
        padding (int): Zero-padding added to both sides of the input.
            Passed to Encoder and Decoder.
        output_padding (int): Additional size added to one side of the output shape.
            Passed to Decoder.
        **kwargs: Arguments which will be passed to the filterbank class
            additionally to the usual `n_filters`, `kernel_size` and `stride`.
            Depends on the filterbank family.
    Returns:
        :class:`.Encoder`, :class:`.Decoder`
    """
    fb_class = get(fb_name)

    if who_is_pinv in ["dec", "decoder"]:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = Encoder(fb, padding=padding)
        # Decoder filterbank is pseudo inverse of encoder filterbank.
        dec = Decoder.pinv_of(fb)
    elif who_is_pinv in ["enc", "encoder"]:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
        # Encoder filterbank is pseudo inverse of decoder filterbank.
        enc = Encoder.pinv_of(fb)
    elif ("sfi_td" in fb_name) or ("TDNAFFilterBank" == fb_name) or ("TDMGFFilterBank" == fb_name):
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = SFITDEncoder(fb, padding=padding)
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = SFITDDecoder(fb, padding=padding, output_padding=output_padding)
    elif ("sfi_fd" in fb_name) or ("FDNAFFilterBank" == fb_name) or ("FDMGFFilterBank" == fb_name):
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = SFIFDEncoder(fb, n_samples=n_samples, padding=padding)
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = SFIFDDecoder(fb, n_samples=n_samples, padding=padding, output_padding=output_padding)
    else:
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        enc = Encoder(fb, padding=padding)
        # Filters between encoder and decoder should not be shared.
        fb = fb_class(n_filters, kernel_size, stride=stride, sample_rate=sample_rate, **kwargs)
        dec = Decoder(fb, padding=padding, output_padding=output_padding)
    return enc, dec


def register_filterbank(custom_fb):
    """Register a custom filterbank, gettable with `filterbanks.get`.

    Args:
        custom_fb: Custom filterbank to register.

    """
    if custom_fb.__name__ in globals().keys() or custom_fb.__name__.lower() in globals().keys():
        raise ValueError(f"Filterbank {custom_fb.__name__} already exists. Choose another name.")
    globals().update({custom_fb.__name__: custom_fb})


def get(identifier):
    """Returns a filterbank class from a string. Returns its input if it
    is callable (already a :class:`.Filterbank` for example).

    Args:
        identifier (str or Callable or None): the filterbank identifier.

    Returns:
        :class:`.Filterbank` or None
    """
    if identifier is None:
        return None
    elif callable(identifier):
        return identifier
    elif isinstance(identifier, str):
        cls = globals().get(identifier)
        if cls is None:
            raise ValueError("Could not interpret filterbank identifier: " + str(identifier))
        return cls
    else:
        raise ValueError("Could not interpret filterbank identifier: " + str(identifier))


# Aliases.
free = FreeFB
analytic_free = AnalyticFreeFB
param_sinc = ParamSincFB
stft = STFTFB
torch_stft = TorchSTFTFB
multiphase_gammatone = mpgtf = MultiphaseGammatoneFB
sfi_tdnaf = TDNAFFilterBank
sfi_fdnaf = FDNAFFilterBank
sfi_tdmgf = TDMGFFilterBank
sfi_fdmgf = FDMGFFilterBank
# sfi_tdmgf

# For the docs
__all__ = [
    "Filterbank",
    "Encoder",
    "Decoder",
    "FreeFB",
    "STFTFB",
    "TorchSTFTFB",
    "AnalyticFreeFB",
    "ParamSincFB",
    "MultiphaseGammatoneFB",
    "MelGramFB",
    "griffin_lim",
    "misi",
    "make_enc_dec",
    "TDNAFFilterBank"
]
