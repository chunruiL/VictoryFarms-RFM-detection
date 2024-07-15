
import numpy as np


class CusumMeanDetector:
    """
    Cumulative Sum (Cusum) Mean Detector for identifying change points in time series data.

    Parameters
    ----------
    t_warmup : int
        Number of initial observations to warm up and estimate initial mean and standard deviation.
    p_limit : float
        P-value limit for determining significant change points.
    """

    def __init__(self, t_warmup: int, p_limit: float):
        """
        Initialize the CusumMeanDetector with the given warm-up period and p-value limit.

        Parameters
        ----------
        t_warmup : int
            Number of initial observations to warm up and estimate initial mean and standard deviation.
        p_limit : float
            P-value limit for determining significant change points.
        """
        self.t_warmup = t_warmup
        self.p_limit = p_limit
        self.current_t = 0
        self.current_obs = []
        self.mean = 0
        self.std = 0

    def predict_next(self, y: float):
        """
        Predict the next observation and check for change points.

        Parameters
        ----------
        y : float
            The next observation in the time series.

        Returns
        -------
        prob : float
            The probability that the current observation is part of the same distribution as previous observations.
        is_changepoint : bool
            Whether the current observation is identified as a change point.
        """
        self.current_t += 1
        self.current_obs.append(y)

        if self.current_t <= self.t_warmup:
            if self.current_t == self.t_warmup:
                self.mean = np.mean(self.current_obs)
                self.std = np.std(self.current_obs, ddof=1)
            return 0.0, False

        # Update mean and standard deviation
        self.mean = np.mean(self.current_obs)
        self.std = np.std(self.current_obs, ddof=1)

        if self.std < 1e-10:
            self.std = 1e-10

        standardized_value = (y - self.mean) / self.std
        prob = 2 * (1 - self._normal_cdf(abs(standardized_value)))

        is_changepoint = prob < self.p_limit
        return prob, is_changepoint

    def _normal_cdf(self, x: float) -> float:
        """
        Calculate the cumulative distribution function (CDF) of a standard normal distribution.

        Parameters
        ----------
        x : float
            The value for which to calculate the CDF.

        Returns
        -------
        float
            The CDF value for the input x.
        """
        return (1.0 + np.math.erf(x / np.sqrt(2.0))) / 2.0





def flag_customers_clv(customer_list, CLV_trajectory, p_limit=0.01, indices=[22, 26, 30]):
    """
    Flag customers based on changes in their CLV values.

    Parameters
    ----------
    customer_list : list
        List of customer IDs.
    CLV_trajectory : dict
        Dictionary where keys are customer IDs and values are lists of CLV values.
    p_limit : float, optional
        The p-value limit for the CusumMeanDetector. Default is 0.01.
    indices : tuple, optional
        Indices to check for changepoints. Default is (22, 26, 30).

    Returns
    -------
    dict
        Dictionary with index keys and lists of flagged customer IDs as values.
    """
    flagged_customers = {idx: [] for idx in indices}

    for customer_id in customer_list:
        detector = CusumMeanDetector(t_warmup=18, p_limit=p_limit)
        clv_values = CLV_trajectory[customer_id]
        previous = 0

        for idx, clv in enumerate(clv_values):
            _, is_changepoint = detector.predict_next(clv)
            if is_changepoint and (clv < previous) and (idx in indices):
                flagged_customers[idx].append(customer_id)
            previous = clv
        
            all_flagged_customers = set()
            
    for idx in indices:
        current_set = set(flagged_customers[idx])
        unique_set = current_set - all_flagged_customers
        flagged_customers[idx] = list(unique_set)
        all_flagged_customers.update(unique_set)


    return flagged_customers