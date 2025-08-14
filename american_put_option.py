import os
import math
import numpy as np

def price_american_put_options(input_text: str | bytes) -> None:
    """
    Processes American put option parameters from a multi-line string or from a text file containing such data.

    This function expects each line to contain tab-separated values with the following parameters:
    risk_free_rate, time in years, number of steps, volatility, initial price, and strike price. It computes and prints
    the value (one per line) of the American put option for each line of input.

    Parameters:
    - input_data (str | bytes): A multi-line string with tab-separated values or a path to a text file containing such
                                data.

    Note:
    - The function directly prints (one output per line) the calculated option prices, hence no return value.
    """

    # read file or str
    if isinstance(input_text, bytes):
        try:
            text = input_text.decode("utf-8")
        except Exception:
            text = input_text.decode("latin-1")
    else:
        text = input_text

    if isinstance(text, str) and os.path.exists(text):
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()

    # reading csv or str from files 
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip() and not ln.strip().startswith("#")]
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) != 6:
            raise ValueError(f"expected 6 tab-separated values per line, got {len(parts)} in line: {ln!r}")
        r, T, N, sigma, S0, K = parts
        price = value_of_american_put_option(
            float(r), float(T), int(float(N)), float(sigma), float(S0), float(K)
        )
        print(f"{price:.6f}")

def value_of_american_put_option(risk_free_rate: float, time: float, num_steps: int, volatility: float,
                                 initial_price: float, strike_price: float) -> float:
    """
    Calculates the value of an American put option using the Binomial Tree model.

    Parameters:
    - risk_free_rate (float): The risk-free interest rate (annualized).
    - time (float): Time to expiration of the option (in years).
    - num_steps (int): Number of steps in the binomial model.
    - volatility (float): Annual volatility of the underlying asset.
    - initial_price (float): The initial stock price.
    - strike_price (float): The strike price of the option.

    Returns:
    - float: The estimated value of the American put option.

    notes:
    - uses risk-neutral probability p = (e^{r dt} - d) / (u - d), with u = e^{sigma sqrt(dt)}, d = 1/u.
    - handles degenerate cases (e.g., sigma ≈ 0 or dt ≈ 0) by falling back to intrinsic/discounted payoff logic.
    """
    # basic guards
    if time <= 0 or num_steps <= 0:
        return max(strike_price - initial_price, 0.0)
    if initial_price <= 0 or strike_price <= 0:
        return 0.0

    # delta t = T / N
    dt = time / float(num_steps)

    # handle near-zero volatility: avoid u≈d numerical issues
    if volatility <= 1e-12:
        # risk-neutral drift path; american feature => 
        # compare immediate exercise vs discounted terminal payoff

        # expected terminal = S0 * e^(r * T)
        expected_terminal = initial_price * math.exp(risk_free_rate * time)
        intrinsic_now = max(strike_price - initial_price, 0.0)
        # discounted_eur_like = e^(-r * T) * max(K - expected_terminal, 0)
        discounted_eur_like = max(strike_price - expected_terminal, 0.0) * math.exp(-risk_free_rate * time)
        return max(intrinsic_now, discounted_eur_like)

    # u = e^(volatility * sqrt(dt))
    u = math.exp(volatility * math.sqrt(dt))
    # d = 1 / u
    d = 1.0 / u
    # discount factor = e^(-r * delta t)
    disc = math.exp(-risk_free_rate * dt)

    # denom = (u - d) for p formula
    denom = (u - d)
    if abs(denom) < 1e-18:
        # fallback if u≈d numerically
        expected_terminal = initial_price * math.exp(risk_free_rate * time)
        intrinsic_now = max(strike_price - initial_price, 0.0)
        discounted_eur_like = max(strike_price - expected_terminal, 0.0) * math.exp(-risk_free_rate * time)
        return max(intrinsic_now, discounted_eur_like)

    # p = ( e^(r * Δt) - d ) / ( u - d )
    p = (math.exp(risk_free_rate * dt) - d) / denom
    p = min(1.0, max(0.0, p))
    # q = 1 - p
    q = 1.0 - p

    # terminal stock prices s_T(j) for j=0..N
    j = np.arange(num_steps + 1)
    sT = initial_price * (u ** j) * (d ** (num_steps - j))
    values = np.maximum(strike_price - sT, 0.0)  # terminal put payoffs

    # backward induction with early exercise
    for step in range(num_steps - 1, -1, -1):
        # continuation (expected discounted)
        values = disc * (p * values[1:] + q * values[:-1])

        # underlying at this step for nodes j=0..step
        j = np.arange(step + 1)
        s_node = initial_price * (u ** j) * (d ** (step - j))
        exercise = np.maximum(strike_price - s_node, 0.0)

        # american max
        values = np.maximum(values, exercise)

    return float(values[0])

if __name__ == "__main__":
    s = "0.05\t1.0\t200\t0.2\t100\t100\n0.03\t0.5\t100\t0.25\t80\t90"
    price_american_put_options(s)
