"""The PanelGP bilinear variational Gaussian-process factor model.

Implements Sections 3-4 and Appendix A-C of the paper: smooth loading and
factor Gaussian-process priors (Eq. 9-12), the sparse-inducing whitened
variational family (Eq. 13-17, App. C.3), the row likelihood with the
loading analytically integrated out and the factor estimated by Monte Carlo
(Eq. A.8-A.11), and the predictive moments used for out-of-sample forecasts
and the variance decomposition (Eq. 19-20, 24-27).

No existing Gaussian-process library (GPyTorch, GPflow) implements this
bilinear, panel-structured likelihood with a factor latent state shared
across assets and a loading latent state shared across dates, so the model
is implemented directly in PyTorch following the paper's equations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.kernels import rbf_kernel, affine_kernel, safe_cholesky
from src import config as cfg


def _softplus_inv(y: torch.Tensor) -> torch.Tensor:
    """Inverse softplus, used only to set informative initial values."""
    return y + torch.log(-torch.expm1(-y))


class PanelGPModel(nn.Module):
    """Bilinear variational GP factor model of paper Sections 3-4.

    Args:
        P: number of loading (characteristic) input coordinates.
        Q: number of factor (context) input coordinates.
        T: number of training dates (columns of the realized-factor state).
        H: number of factors.
        M: number of trainable inducing inputs per component.
        loading_kernel: "rbf" or "affine" (paper Eq. C.3).
        factor_kernel: "rbf" or "constant" (paper Eq. 12, C.4).
        isotropic_innovations: if True, D_beta and D_phi are each a single
            scalar shared across components (paper Appendix C.2,
            "Simulation"); otherwise each component has its own innovation
            variance (paper Appendix C.2, "In the application").
        Z_init: (H, M, P) initial inducing inputs, placed at distinct
            training rows (paper Appendix C.3).
        C_train: (T, Q) training context values, one row per training date.
        loading_diag_init: scalar used for the initial training-average
            loading-kernel diagonal d (paper App. C.4): 1 for RBF,
            1 + mean(||x||^2) for the affine kernel.
    """

    def __init__(self, P: int, Q: int, T: int, H: int, M: int,
                 loading_kernel: str, factor_kernel: str,
                 isotropic_innovations: bool, Z_init: torch.Tensor,
                 C_train: torch.Tensor, loading_diag_init: float):
        super().__init__()
        assert loading_kernel in ("rbf", "affine")
        assert factor_kernel in ("rbf", "constant")
        self.P, self.Q, self.T, self.H, self.M = P, Q, T, H, M
        self.loading_kernel = loading_kernel
        self.factor_kernel = factor_kernel
        self.isotropic = isotropic_innovations
        self.register_buffer("C_train", C_train)

        self.Z = nn.Parameter(Z_init.clone())

        # --- Loading kernel parameters (paper Eq. C.3) ---
        gamma_beta = cfg.SIM_GAMMA_BETA
        eta0 = 0.9 / (H * (loading_diag_init + gamma_beta))
        self.raw_eta = nn.Parameter(_softplus_inv(torch.full((H,), eta0)))
        if loading_kernel == "rbf":
            ell0 = (P / 12.0) ** 0.5
            self.raw_ell_beta = nn.Parameter(
                _softplus_inv(torch.full((H, P), ell0)))

        # --- Factor kernel parameters (paper Eq. 12, C.4) ---
        if factor_kernel == "rbf":
            # Paper App. C.4: ell_0 = sqrt(sum_p zeta_p^2), zeta_p^2 the
            # per-coordinate input variance (1/12 for a Uniform[0,1] input),
            # matching the loading-kernel initialization two lines above.
            ell0 = (Q / 12.0) ** 0.5
            self.raw_ell_phi = nn.Parameter(
                _softplus_inv(torch.full((H, Q), ell0)))
            if isotropic_innovations:
                # shared rho via sigmoid, per-h (a,b) split via two-way softmax
                self.raw_rho_shared = nn.Parameter(torch.logit(torch.tensor(0.1)))
                self.raw_frac_logits = nn.Parameter(
                    torch.log(torch.tensor([0.8, 0.1])).repeat(H, 1).clone())
            else:
                init_shares = torch.log(torch.tensor([0.8, 0.1, 0.1]))
                self.raw_share_logits = nn.Parameter(init_shares.repeat(H, 1).clone())
        else:  # constant factor kernel: a_h = 0, rho_h via sigmoid
            self.raw_rho_logit = nn.Parameter(torch.logit(torch.full((H,), 0.1)))

        # --- Loading innovation variance tau_beta = [D_beta]_hh (Eq. C.3) ---
        if isotropic_innovations:
            self.raw_tau_beta_shared = nn.Parameter(_softplus_inv(torch.tensor(0.1)))
        else:
            self.raw_tau_beta = nn.Parameter(_softplus_inv(torch.full((H,), 0.1)))

        # --- Observation noise (paper App. C.4: sigma_r^2 = 0.1 initial) ---
        self.raw_sigma_r2 = nn.Parameter(_softplus_inv(torch.tensor(0.1)))

        # --- Variational state (whitened, Eq. C.5) ---
        dv = (M + T) * H
        self.dv = dv
        self.m_omega = nn.Parameter(torch.zeros(dv))
        L0 = torch.eye(dv) * 0.1
        self.L_omega_raw = nn.Parameter(L0.clone())

    # ------------------------------------------------------------------
    # Parameter transforms
    # ------------------------------------------------------------------
    def eta(self) -> torch.Tensor:
        return F.softplus(self.raw_eta)

    def sigma_r2(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_r2)

    def tau_beta(self) -> torch.Tensor:
        if self.isotropic:
            return F.softplus(self.raw_tau_beta_shared).expand(self.H)
        return F.softplus(self.raw_tau_beta)

    def factor_shares(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (a_h, b_h, rho_h), each (H,), with a_h+b_h+rho_h=1 (Eq. 12)."""
        if self.factor_kernel == "constant":
            rho = torch.sigmoid(self.raw_rho_logit)
            b = 1.0 - rho
            a = torch.zeros_like(rho)
            return a, b, rho
        if self.isotropic:
            rho = torch.sigmoid(self.raw_rho_shared).expand(self.H)
            frac = torch.softmax(self.raw_frac_logits, dim=-1)  # (H,2)
            a = frac[:, 0] * (1 - rho)
            b = frac[:, 1] * (1 - rho)
            return a, b, rho
        shares = torch.softmax(self.raw_share_logits, dim=-1)  # (H,3)
        return shares[:, 0], shares[:, 1], shares[:, 2]

    def L_omega(self) -> torch.Tensor:
        """Dense trainable Cholesky factor of S_omega, lower-triangular with
        a softplus-transformed positive diagonal."""
        tril = torch.tril(self.L_omega_raw, diagonal=-1)
        diag = F.softplus(torch.diagonal(self.L_omega_raw))
        return tril + torch.diag(diag)

    # ------------------------------------------------------------------
    # Kernels
    # ------------------------------------------------------------------
    def _kbar_beta(self, x1: torch.Tensor, x2: torch.Tensor, h: int) -> torch.Tensor:
        if self.loading_kernel == "rbf":
            ell = F.softplus(self.raw_ell_beta[h])
            return rbf_kernel(x1, x2, ell)
        return affine_kernel(x1, x2)

    def _kbar_beta_diag(self, x: torch.Tensor, h: int) -> torch.Tensor:
        if self.loading_kernel == "rbf":
            return torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        return 1.0 + (x ** 2).sum(-1)

    def _kbar_phi(self, c1: torch.Tensor, c2: torch.Tensor, h: int,
                  a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """kbar_phi,h(c,c') = a_h*RBF(c,c') + b_h, without the rho_h I term
        (paper Eq. 12); used both for the training prior (plus rho*I) and
        for smooth-function prediction (without rho)."""
        n1, n2 = c1.shape[0], c2.shape[0]
        out = b[h] * torch.ones(n1, n2, device=c1.device, dtype=c1.dtype)
        if self.factor_kernel == "rbf":
            ell = F.softplus(self.raw_ell_phi[h])
            out = out + a[h] * rbf_kernel(c1, c2, ell)
        return out

    # ------------------------------------------------------------------
    # Prior Cholesky factors (block-diagonal whitening, paper App. C.3)
    # ------------------------------------------------------------------
    def prior_cholesky(self):
        """Returns (L_beta list of H (M,M), L_phi list of H (T,T), a, b, rho)."""
        eta = self.eta()
        a, b, rho = self.factor_shares()
        L_beta, L_phi = [], []
        for h in range(self.H):
            K_beta_ZZ = eta[h] * self._kbar_beta(self.Z[h], self.Z[h], h)
            L_b, _ = safe_cholesky(K_beta_ZZ, cfg.JITTER_INIT, cfg.JITTER_GROWTH,
                                    cfg.JITTER_MAX_TRIES)
            L_beta.append(L_b)
            K_phi_CC = self._kbar_phi(self.C_train, self.C_train, h, a, b)
            K_phi_CC = K_phi_CC + rho[h] * torch.eye(
                self.T, device=K_phi_CC.device, dtype=K_phi_CC.dtype)
            L_p, _ = safe_cholesky(K_phi_CC, cfg.JITTER_INIT, cfg.JITTER_GROWTH,
                                    cfg.JITTER_MAX_TRIES)
            L_phi.append(L_p)
        return L_beta, L_phi, a, b, rho

    def variational_moments(self, L_beta, L_phi):
        """Builds the block-diagonal whitening L_v, and returns
        m_v (dv,), S_v (dv,dv) dense (paper Eq. C.5, 16)."""
        L_v = torch.block_diag(*L_beta, *L_phi)
        m_v = L_v @ self.m_omega
        L_om = self.L_omega()
        S_omega = L_om @ L_om.T
        S_v = L_v @ S_omega @ L_v.T
        return m_v, S_v

    def kl_divergence(self) -> torch.Tensor:
        """KL(q_psi(omega) || N(0,I)), paper Equation (C.5)."""
        L_om = self.L_omega()
        S_omega = L_om @ L_om.T
        logdet = 2.0 * torch.log(torch.diagonal(L_om)).sum()
        return 0.5 * (torch.trace(S_omega) + (self.m_omega ** 2).sum()
                       - self.dv - logdet)

    # ------------------------------------------------------------------
    # Row-level Gaussian-process projections (paper Eq. A.8-A.9)
    # ------------------------------------------------------------------
    def _row_projection(self, X_batch: torch.Tensor, L_beta) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns G_batch (H,B,M) and R_batch (H,B) of Eq. (A.8)."""
        H, M = self.H, self.M
        eta = self.eta()
        B = X_batch.shape[0]
        G = torch.empty(H, B, M, device=X_batch.device, dtype=X_batch.dtype)
        R = torch.empty(H, B, device=X_batch.device, dtype=X_batch.dtype)
        for h in range(H):
            k_bZ = eta[h] * self._kbar_beta(X_batch, self.Z[h], h)  # (B,M)
            Y = torch.cholesky_solve(k_bZ.T, L_beta[h])  # (M,B)
            G[h] = Y.T
            k_diag = eta[h] * self._kbar_beta_diag(X_batch, h)  # (B,)
            R[h] = k_diag - (Y.T * k_bZ).sum(-1)
        return G, R

    def _gather_blocks(self, m_v, S_v, G, R, date_idx):
        """Assembles per-row (B,H,H) covariance blocks and (B,H) means
        needed for Eq. (A.9)-(A.11)."""
        H, M, T = self.H, self.M, self.T
        m_u = m_v[:H * M].reshape(H, M)
        m_phi_all = m_v[H * M:].reshape(H, T)
        S_uu_all = S_v[:H * M, :H * M].reshape(H, M, H, M).permute(0, 2, 1, 3)
        S_uphi_all = S_v[:H * M, H * M:].reshape(H, M, H, T).permute(0, 2, 1, 3)
        S_phiphi_all = S_v[H * M:, H * M:].reshape(H, T, H, T).permute(0, 2, 1, 3)

        B = G.shape[1]
        device, dtype = G.device, G.dtype
        S_beta_beta = torch.zeros(B, H, H, device=device, dtype=dtype)
        S_beta_phi = torch.zeros(B, H, H, device=device, dtype=dtype)
        S_phi_phi = torch.zeros(B, H, H, device=device, dtype=dtype)
        m_beta = torch.einsum("hbm,hm->bh", G, m_u)
        m_phi = m_phi_all[:, date_idx].T  # (B,H)

        for h in range(H):
            for h2 in range(H):
                tmp = G[h] @ S_uu_all[h, h2]  # (B,M)
                val = (tmp * G[h2]).sum(-1)
                if h == h2:
                    val = val + R[h]
                S_beta_beta[:, h, h2] = val

                gathered = S_uphi_all[h, h2][:, date_idx].T  # (B,M)
                S_beta_phi[:, h, h2] = (G[h] * gathered).sum(-1)

                S_phi_phi[:, h, h2] = S_phiphi_all[h, h2][date_idx, date_idx]
        return m_beta, m_phi, S_beta_beta, S_beta_phi, S_phi_phi

    # ------------------------------------------------------------------
    # ELBO (paper Eq. 14, 17, A.11)
    # ------------------------------------------------------------------
    def expected_log_lik(self, X_batch, date_idx, r_batch, n_mc: int):
        """Monte Carlo estimate of E_q[log p(r_n | beta(x_n), phi~_t(n))]
        summed over the batch (paper Eq. A.11)."""
        L_beta, L_phi, a, b, rho = self.prior_cholesky()
        m_v, S_v = self.variational_moments(L_beta, L_phi)
        G, R = self._row_projection(X_batch, L_beta)
        m_beta, m_phi, S_bb, S_bphi, S_phiphi = self._gather_blocks(
            m_v, S_v, G, R, date_idx)

        jitter = cfg.JITTER_INIT * torch.eye(
            self.H, device=X_batch.device, dtype=X_batch.dtype)
        L_phiphi = torch.linalg.cholesky(S_phiphi + jitter)

        half = n_mc // 2
        e = torch.randn(half, X_batch.shape[0], self.H,
                         device=X_batch.device, dtype=X_batch.dtype)
        e_full = torch.cat([e, -e], dim=0)  # antithetic sampling, App. C.4
        phi_samples = m_phi.unsqueeze(0) + torch.einsum(
            "bhg,kbg->kbh", L_phiphi, e_full)

        L_T = L_phiphi.transpose(-1, -2)
        rhs = e_full.permute(1, 2, 0)  # (B,H,K)
        y = torch.linalg.solve_triangular(L_T, rhs, upper=True).permute(2, 0, 1)

        bar_beta = m_beta.unsqueeze(0) + torch.einsum("bhg,kbg->kbh", S_bphi, y)

        S_phi_beta = S_bphi.transpose(-1, -2)
        corr = torch.cholesky_solve(S_phi_beta, L_phiphi)  # Sphiphi^{-1} S_phibeta
        V_n = S_bb - torch.einsum("bhg,bgk->bhk", S_bphi, corr)
        V_n = 0.5 * (V_n + V_n.transpose(-1, -2)) + 1e-6 * torch.eye(
            self.H, device=X_batch.device, dtype=X_batch.dtype)

        tau_beta = self.tau_beta()
        sigma_r2 = self.sigma_r2()
        nu = sigma_r2 + torch.einsum("kbh,h,kbh->kb", phi_samples, tau_beta, phi_samples)

        pred_mean = torch.einsum("kbh,kbh->kb", bar_beta, phi_samples)
        quad1 = (r_batch.unsqueeze(0) - pred_mean) ** 2
        quad2 = torch.einsum("kbh,bhg,kbg->kb", phi_samples, V_n, phi_samples)

        log_lik = -0.5 * (torch.log(2 * torch.pi * nu) + (quad1 + quad2) / nu)
        return log_lik.mean(0).sum()  # sum over batch rows, MC-averaged

    def elbo(self, X_batch, date_idx, r_batch, n_total: int, n_mc: int):
        """Evidence lower bound (paper Eq. 14, 17), scaled for minibatches."""
        batch_size = X_batch.shape[0]
        ell = self.expected_log_lik(X_batch, date_idx, r_batch, n_mc)
        ell_scaled = ell * (n_total / batch_size)
        kl = self.kl_divergence()
        return ell_scaled - kl

    # ------------------------------------------------------------------
    # Prediction (paper Eq. 19-20); used for dense-grid function recovery
    # and out-of-sample forecasts (Section 4.2-4.3)
    # ------------------------------------------------------------------
    def predict_loading(self, X_star: torch.Tensor):
        """Smooth loading posterior mean/covariance at new characteristics
        X_star (no added innovation: loadings have no realized state,
        paper Eq. 10)."""
        L_beta, L_phi, a, b, rho = self.prior_cholesky()
        m_v, S_v = self.variational_moments(L_beta, L_phi)
        G, R = self._row_projection(X_star, L_beta)
        H, M = self.H, self.M
        m_u = m_v[:H * M].reshape(H, M)
        S_uu_all = S_v[:H * M, :H * M].reshape(H, M, H, M).permute(0, 2, 1, 3)
        m_beta = torch.einsum("hbm,hm->bh", G, m_u)  # (B,H)
        B = X_star.shape[0]
        S_beta = torch.zeros(B, H, H, device=X_star.device, dtype=X_star.dtype)
        for h in range(H):
            for h2 in range(H):
                tmp = G[h] @ S_uu_all[h, h2]
                val = (tmp * G[h2]).sum(-1)
                if h == h2:
                    val = val + R[h]
                S_beta[:, h, h2] = val
        return m_beta, S_beta

    def predict_factor(self, C_star: torch.Tensor, add_innovation: bool):
        """Factor posterior mean/covariance at new contexts C_star.

        ``add_innovation=False`` recovers the smooth map phi_h(c) (used for
        the function-R^2 evaluation); ``add_innovation=True`` predicts the
        realized factor at a genuinely new date, adding D_phi to the
        self-variance as in Equation (25)-(27).
        """
        L_beta, L_phi, a, b, rho = self.prior_cholesky()
        m_v, S_v = self.variational_moments(L_beta, L_phi)
        H, T = self.H, self.T
        m_phi_all = m_v[H * self.M:].reshape(H, T)
        S_phiphi_all = S_v[H * self.M:, H * self.M:].reshape(
            H, T, H, T).permute(0, 2, 1, 3)
        B = C_star.shape[0]
        G = torch.empty(H, B, T, device=C_star.device, dtype=C_star.dtype)
        Rs = torch.empty(H, B, device=C_star.device, dtype=C_star.dtype)
        for h in range(H):
            k_cC = self._kbar_phi(C_star, self.C_train, h, a, b)  # (B,T)
            Y = torch.cholesky_solve(k_cC.T, L_phi[h])  # (T,B)
            G[h] = Y.T
            k_diag = self._kbar_phi(C_star, C_star, h, a, b).diagonal()
            Rs[h] = k_diag - (Y.T * k_cC).sum(-1)
            if add_innovation:
                Rs[h] = Rs[h] + rho[h]
        m_phi = torch.einsum("hbt,ht->bh", G, m_phi_all)
        S_phi = torch.zeros(B, H, H, device=C_star.device, dtype=C_star.dtype)
        for h in range(H):
            for h2 in range(H):
                tmp = G[h] @ S_phiphi_all[h, h2]
                val = (tmp * G[h2]).sum(-1)
                if h == h2:
                    val = val + Rs[h]
                S_phi[:, h, h2] = val
        return m_phi, S_phi

    def realized_factors_at_training_dates(self):
        """Fitted realized factor posterior mean/covariance m_phi, S_phi at
        every training date, read directly off the variational state
        (paper Section 4.1, Eq. 15-16): no GP extrapolation is needed since
        training dates are literally part of the latent state v."""
        L_beta, L_phi, a, b, rho = self.prior_cholesky()
        m_v, S_v = self.variational_moments(L_beta, L_phi)
        H, M, T = self.H, self.M, self.T
        m_phi = m_v[H * M:].reshape(H, T).T  # (T,H)
        S_phiphi_all = S_v[H * M:, H * M:].reshape(H, T, H, T).permute(0, 2, 1, 3)
        S_phi = torch.zeros(T, H, H, device=m_v.device, dtype=m_v.dtype)
        idx = torch.arange(T, device=m_v.device)
        for h in range(H):
            for h2 in range(H):
                S_phi[:, h, h2] = S_phiphi_all[h, h2][idx, idx]
        return m_phi, S_phi
