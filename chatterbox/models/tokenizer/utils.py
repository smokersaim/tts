from .config import SOS, EOS

def drop_invalid_tokens(x):
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1)
    if SOS in x:
        s = (x == SOS).nonzero(as_tuple=True)[0].squeeze(0) + 1
    else:
        s = 0
    if EOS in x:
        e = (x == EOS).nonzero(as_tuple=True)[0].squeeze(0)
    else:
        e = None
    return x[s:e]
