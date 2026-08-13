import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Emitter_class import Emitter
# ----------------------------------------------------------------------
# 1. PRI dizisi:
# ----------------------------------------------------------------------
def make_pri_agile(transmitter_: dict) -> np.ndarray:
    N        = int(transmitter_.get("number_of_pulse", 0))
    mode     = str(transmitter_.get("PRI_mode", "stable")).lower()
    mean_PRI = float(transmitter_.get("PRI_mean", 0.0))
    PRI_amp      = float(transmitter_.get("PRI_agile_ampl",
                                      transmitter_.get("PRI_agileness_amplitude", 0.0)))


    if mode == "stable":
        PRI_seq = np.full(N, mean_PRI)
    elif mode == "staggered":
        d       = PRI_amp * mean_PRI
        PRI_seq = np.tile([mean_PRI - d, mean_PRI + d], int(np.ceil(N/2)))[:N]
    elif mode == "jittered":
        PRI_seq = mean_PRI * (1 + PRI_amp * (2*np.random.rand(N) - 1))
    elif mode == "wobulated":
        PRI_seq = mean_PRI + (PRI_amp * mean_PRI) * np.sin(2*np.pi*np.arange(N)/N)
    elif mode in {"sliding", "sliding+"}:
        d       = PRI_amp * mean_PRI
        PRI_seq = np.linspace(mean_PRI - d, mean_PRI + d, N)
    elif mode == "sliding-":
        d       = PRI_amp * mean_PRI
        PRI_seq = np.linspace(mean_PRI + d, mean_PRI - d, N)
    else:
        raise ValueError(f"Unknown PRI_mode: {mode}")
    return np.concatenate(([0.0], np.cumsum(PRI_seq[:-1])))

def start_time_calculator(PRI_values:np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(PRI_values[:-1])))

def pri_values(transmitter:'Emitter') -> np.ndarray:
    N        = int(transmitter.number_of_pulse)
    mode     = str(transmitter.PRI_type).lower()
    mean_PRI = float(transmitter.PRI_mean)
    PRI_amp      = float(transmitter.PRI_agile_ampl)


    if mode == "stable":
        PRI_seq = np.full(N, mean_PRI)
    elif mode == "staggered":
        d       = PRI_amp * mean_PRI
        PRI_seq = np.tile([mean_PRI - d, mean_PRI + d], int(np.ceil(N/2)))[:N]
    elif mode == "jittered":
        PRI_seq = mean_PRI * (1 + PRI_amp * (2*np.random.rand(N) - 1))
    elif mode == "wobulated":
        PRI_seq = mean_PRI + (PRI_amp * mean_PRI) * np.sin(2*np.pi*np.arange(N)/N)
    elif mode in {"sliding", "sliding+"}:
        d       = PRI_amp * mean_PRI
        PRI_seq = np.linspace(mean_PRI - d, mean_PRI + d, N)
    elif mode == "sliding-":
        d       = PRI_amp * mean_PRI
        PRI_seq = np.linspace(mean_PRI + d, mean_PRI - d, N)
    else:
        raise ValueError(f"Unknown PRI_mode: {mode}")
    return PRI_seq


# ----------------------------------------------------------------------
# 2. Darbe (pulse) üreticileri
# ----------------------------------------------------------------------
def _rect_pulse(tx:'Emitter'):
    Ts  = 1.0 / tx.fs
    tau = tx.pulse_width
    n_samples = int(tau//Ts)
    t = np.arange(0.0, n_samples*Ts, Ts); 
    return np.cos(2*np.pi*tx.fc*t)

def _barker_pulse(tx:'Emitter'):
    seq_tbl = {2:[1,-1],3:[1,1,-1],4:[1,1,-1,1],5:[1,1,1,-1,1],
               7:[1,1,1,-1,-1,1,-1],
               11:[1,1,1,-1,-1,-1,1,-1,-1,1,-1],
               13:[1,1,1,1,1,-1,-1,1,1,-1,1,-1,1]}
    L   = int(tx.wave_param_L)
    if L not in seq_tbl:
        raise ValueError("Unsupported Barker length.")
    Ts  = 1.0 / tx.fs
    tau = tx.pulse_width
    Tc  = tau / L
    n_samples = int(Tc//Ts)
    t_c = np.arange(0.0, n_samples*Ts, Ts); spc = t_c.size
    pulse = np.zeros(L*spc)
    for k, chip in enumerate(seq_tbl[L]):
        idx = slice(k*spc, (k+1)*spc)
        pulse[idx] = chip * np.cos(2*np.pi*tx.fc*(t_c + k*Tc))
    return pulse

def _welch_costas_sequence(L, p=2):
    """
    Generate a Costas sequence using Welch construction.
    L must be a prime number, p is the primitive root modulo L (default 2).
    Returns a numpy array of the permutation.
    """
    if L==7:
        p=3
    elif L==11:
        p=2
    elif L==13:
        p=2
    elif L==17:
        p=3
    if L < 2 or not _is_prime(L):
        raise ValueError("Welch construction requires L to be a prime number >= 2.")
    # Find a primitive root if not given
    if p is None:
        for cand in range(2, L):
            if _is_primitive_root(cand, L):
                p = cand
                break
        else:
            raise ValueError(f"No primitive root found for L={L}")
    seq = np.array([(pow(p, j, L)) for j in range(L-1)])
    # Map to 0-based permutation
    perm = np.argsort(seq)
    return seq - 1  # 0-based permutation

def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5)+1):
        if n % i == 0:
            return False
    return True

def _is_primitive_root(g, p):
    required = set(range(1, p))
    actual = set(pow(g, k, p) for k in range(1, p))
    return required == actual

def _costas_pulse(tx:'Emitter'):
    L   = int(tx.wave_param_L)-1
    B   = tx.wave_param_B
    B=B/(L-1)
    Ts  = 1.0 / tx.fs
    tau = tx.pulse_width
    Tc  = tau / L
    n_samples = int(Tc//Ts)
    t_c = np.arange(0.0, n_samples*Ts, Ts); spc = t_c.size
    pulse = np.zeros(L*spc)
    perm  = _welch_costas_sequence(L+1)
    for k, p in enumerate(perm):
        f_inst    = tx.fc + (p)*B
        pulse[k*spc:(k+1)*spc] = np.cos(2*np.pi*f_inst*t_c)
    return pulse

def _frank_pulse(tx:'Emitter'):
    M   = int(tx.wave_param_L)
    L   = M*M
    Ts  = 1.0/tx.fs; tau = tx.pulse_width
    Tc = tau/L
    n_pulse = int(tau/Ts)
    n_samples = int(Tc//Ts)
    nn_pulse = int(L*n_samples)
    t_c = np.arange(0.0, (n_samples-1)*Ts, Ts); spc = t_c.size
    pulse = np.zeros(L*spc)
    phi = [2*np.pi*(p*q)/M for p in range(M) for q in range(M)]
    
    for k, ph in enumerate(phi):
        idx = slice(k*spc, (k+1)*spc)
        carrier = np.cos(2*np.pi*tx.fc*(t_c + k*Tc) + ph)
        pulse[idx] = carrier 
    return pulse

def _lfm_pulse(tx:'Emitter'):
    B   = tx.wave_param_B
    tau = tx.pulse_width
    mu  = B / tau
    Ts  = 1.0/tx.fs
    t_p = np.arange(0.0, tau, Ts)
    return np.cos(2*np.pi*tx.fc*t_p + np.pi*mu*t_p**2)

def _px_pulse(tx:'Emitter'):
    code = tx.wave_type  # p1, p2, p3, p4
    L0   = int(tx.wave_param_L)
    Ts   = 1.0/tx.fs; tau = tx.pulse_width
    if code in {"p1","p2"}:
        L = L0*L0
    else:
        L = L0
    # faz dizisi
    if code == "p1":
        phi = [np.pi/L0*(2*a*b - b - L0*a + L0 + 1)
               for a in range(L0) for b in range(L0)]
    elif code == "p2":
        phi = [np.pi/(2*L0)*((L0-1-2*a)*(L0-1-2*b))
               for a in range(L0) for b in range(L0)]
    elif code == "p3":
        k = np.arange(L); phi = np.pi/L0 * (k**2)
    elif code == "p4":
        k = np.arange(L); phi = np.pi/L0 * (k**2)-np.pi*(k)
    else:
        raise ValueError("code must be p1…p4")

    Tc   = tau / L
    n_samples = int(Tc//Ts)
    t_c = np.arange(0.0, (n_samples-1)*Ts, Ts); spc = t_c.size
    pulse = np.zeros(L*spc)
    for k, ph in enumerate(phi):
        idx = slice(k*spc, (k+1)*spc)
        carrier = np.cos(2*np.pi*tx.fc*(t_c + k*Tc) + ph)
        pulse[idx] = carrier
    return pulse

def _fmcw_pulse(tx:'Emitter'):
    B   = tx.wave_param_B
    tau = tx.pulse_width
    k   = B / tau
    Ts  = 1.0/tx.fs
    t_p = np.arange(0.0, tau-Ts, Ts)
    f0  = tx.fc - B/2.0     # süpürme fc çevresinde simetrik
    return np.cos(2*np.pi*(f0*t_p + 0.5*k*t_p**2))


# ----------------------------------------------------------------------
# 3. Darbeyi ana dalgaformuna yazdırmak
# ----------------------------------------------------------------------
def _place_pulses(pulse, start_times, Ts):
    L = pulse.size
    pulses_placed = 0
    waveform = np.array([], dtype=complex)
    
    for t0 in start_times:
        waveform = np.concatenate((waveform, pulse))
        pulses_placed += 1
        if pulses_placed >= len(start_times):
            break
        n1 = int(start_times[pulses_placed]/Ts)
        if n1 - waveform.size < 0:
            a=1
        waveform = np.concatenate((waveform, np.zeros(n1 - waveform.size, dtype=waveform.dtype)))

    return waveform


# ----------------------------------------------------------------------
# 4. Dışarıdan çağrılacak ana fonksiyon
# ----------------------------------------------------------------------
def generate_waveform(transmitter: 'Emitter') -> np.ndarray:

    # Zaman ekseni
    Ts  = 1.0 / transmitter.fs
    transmitter.Ts = Ts   # alt fonksiyonlar isterse kullanabilir
    PRI_values=transmitter.PRI_values
    # Darbe başlangıç zamanları
    start_times = start_time_calculator(PRI_values)
    start_indexes = np.round(start_times*transmitter.fs).astype(int)
    # Darbe tipi seçimi
    wt = transmitter.wave_type
    if   wt == "rect":   pulse = _rect_pulse(transmitter)
    elif wt == "barker": pulse = _barker_pulse(transmitter)
    elif wt == "costas": pulse = _costas_pulse(transmitter)
    elif wt == "frank":  pulse = _frank_pulse(transmitter)
    elif wt == "lfm":    pulse = _lfm_pulse(transmitter)
    elif wt in {"p1","p2","p3","p4"}: pulse = _px_pulse(transmitter)
    elif wt == "fmcw":   pulse = _fmcw_pulse(transmitter)
    else:
        raise ValueError(f"Unknown transmitter_type: {wt}")
    waveform = _place_pulses(pulse, start_times,Ts)
    return waveform

def pulse_indices(waveform):
    abs_waveform = np.abs(waveform)
    is_pulse = abs_waveform > 0
    diff = np.diff(is_pulse.astype(int))
    pulse_starts = np.where(diff == 1)[0] + 1
    pulse_ends = np.where(diff == -1)[0] + 1
    if is_pulse[0]:
        pulse_starts = np.insert(pulse_starts, 0, 0)
    if is_pulse[-1]:
        pulse_ends = np.append(pulse_ends, len(waveform))
    pulse_ranges = [f"{start}:{end}" for start, end in zip(pulse_starts, pulse_ends)]
    return pulse_ranges

def reflector(waveform,reflection_snr_db,reflection_time_shift,fs):
    samples_shift = int(reflection_time_shift*fs)
    reflection_snr_linear = 10**(reflection_snr_db / 10)
    reflection_power = 1 / reflection_snr_linear
    reflection_amplitude = np.sqrt(reflection_power)
    reflected_Waveform = waveform * reflection_amplitude
    phase_shift = np.random.uniform(0, 2*np.pi)
    reflected_Waveform = reflected_Waveform * np.exp(1j * phase_shift)
    reflected_Waveform = np.concatenate((np.zeros(samples_shift), reflected_Waveform))
    return reflected_Waveform