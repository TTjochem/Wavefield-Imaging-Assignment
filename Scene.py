from Functions import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from  sklearn.decomposition._truncated_svd import TruncatedSVD
from sklearn.feature_extraction.image import PatchExtractor

class Scene:
    def __init__(self, kb, x_range, y_range, stepsize, boundary='free',
                 auto_pad=True, pad_amount=None):
        self.kb = kb
        self.mu = 2 * np.pi / kb
        self.stepsize = stepsize
        self.boundary = boundary
        self.auto_pad = auto_pad
        self.pad_amount = pad_amount if pad_amount is not None else self.mu

        self.x_range = list(x_range)
        self.y_range = list(y_range)

        self._build_mesh()

        self.domains = {}
        self.sources = {}
        self.receivers = {}
        self.contrasts = {}
        self.U = None
        self.U_inc = None
        self.U_sc = None
        self.U_sc_grid = None
        self.Chi = None

        if self.boundary == 'closed':
            self._setup_closed_space()

    # ==============================================================
    # HANDLING LOGIC
    # ==============================================================
    def _build_mesh(self):
        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        nx = int((xmax - xmin) / self.stepsize) + 1
        ny = int((ymax - ymin) / self.stepsize) + 1

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        self.X, self.Y = np.meshgrid(x, y)
        self.rho_grid = np.stack([self.X, self.Y], axis=-1)

    def _check_and_pad(self):
        if not self.auto_pad:
            return False

        xmin, xmax = self.x_range
        ymin, ymax = self.y_range

        required_xmin = xmin
        required_xmax = xmax
        required_ymin = ymin
        required_ymax = ymax

        for name, d in self.domains.items():
            required_xmin = min(required_xmin, d['rho'][0])
            required_xmax = max(required_xmax, d['rho'][0] + d['width'])
            required_ymin = min(required_ymin, d['rho'][1])
            required_ymax = max(required_ymax, d['rho'][1] + d['height'])

        for name, s in self.sources.items():
            if isinstance(s, dict):
                rho = s['rho']
                if s.get('source_type') == 'line' and s.get('rho_end') is not None:
                    rho_end = s['rho_end']
                    required_xmin = min(required_xmin, rho[0], rho_end[0])
                    required_xmax = max(required_xmax, rho[0], rho_end[0])
                    required_ymin = min(required_ymin, rho[1], rho_end[1])
                    required_ymax = max(required_ymax, rho[1], rho_end[1])
                else:
                    required_xmin = min(required_xmin, rho[0])
                    required_xmax = max(required_xmax, rho[0])
                    required_ymin = min(required_ymin, rho[1])
                    required_ymax = max(required_ymax, rho[1])
            else:
                required_xmin = min(required_xmin, s[0])
                required_xmax = max(required_xmax, s[0])
                required_ymin = min(required_ymin, s[1])
                required_ymax = max(required_ymax, s[1])

        for name, rec in self.receivers.items():
            positions = rec['positions']
            required_xmin = min(required_xmin, positions[:, 0].min())
            required_xmax = max(required_xmax, positions[:, 0].max())
            required_ymin = min(required_ymin, positions[:, 1].min())
            required_ymax = max(required_ymax, positions[:, 1].max())

        for name, c in self.contrasts.items():
            rho_0 = c['rho_0']
            size = c['size']
            required_xmin = min(required_xmin, rho_0[0] - size / 2)
            required_xmax = max(required_xmax, rho_0[0] + size / 2)
            required_ymin = min(required_ymin, rho_0[1] - size / 2)
            required_ymax = max(required_ymax, rho_0[1] + size / 2)

        expand_left = xmin - required_xmin
        expand_right = required_xmax - xmax
        expand_top = ymin - required_ymin
        expand_bottom = required_ymax - ymax

        max_expand = max(expand_left, expand_right, expand_top, expand_bottom, 0)
        if max_expand > 10 * self.mu:
            print("=" * 60)
            print("WARNING: Object placed far outside the grid!")
            print(f"  Required expansion: {max_expand:.2f} (>{10 * self.mu:.2f} = 10λ)")
            print("  Consider redefining your grid bounds manually.")
            print("=" * 60)

        new_xmin = xmin
        new_xmax = xmax
        new_ymin = ymin
        new_ymax = ymax

        needs_pad = False

        if expand_left > 0:
            new_xmin = required_xmin - self.pad_amount
            needs_pad = True
        if expand_right > 0:
            new_xmax = required_xmax + self.pad_amount
            needs_pad = True
        if expand_top > 0:
            new_ymin = required_ymin - self.pad_amount
            needs_pad = True
        if expand_bottom > 0:
            new_ymax = required_ymax + self.pad_amount
            needs_pad = True

        if needs_pad:
            self.x_range = [new_xmin, new_xmax]
            self.y_range = [new_ymin, new_ymax]
            self._build_mesh()
            self._update_all_masks()

        return needs_pad

    def _update_all_masks(self):
        for name, d in self.domains.items():
            d['mask'] = ((self.X >= d['rho'][0]) & (self.X <= d['rho'][0] + d['width']) &
                         (self.Y >= d['rho'][1]) & (self.Y <= d['rho'][1] + d['height']))

        for name, c in self.contrasts.items():
            self._recompute_contrast_mask(name)

    def _recompute_contrast_mask(self, name):
        c = self.contrasts[name]
        shape_funcs = {
            'rectangle': make_rectangle,
            'triangle': make_triangle,
            'star': make_star,
            'letter': make_letter,
            'circle': make_circle,
            'ellipse': make_ellipse,
        }
        func = shape_funcs[c['shape']]
        kwargs = {k: v for k, v in c.items()
                  if k not in ['Chi', 'patches', 'rho_0', 'size', 'intensity', 'shape']}
        Chi_shape, patches_list = func(self.X, self.Y, c['rho_0'],
                                       c['size'], c['intensity'], **kwargs)
        c['Chi'] = Chi_shape
        c['patches'] = patches_list
        self._update_chi()

    # ==============================================================
    # COMPUTATIONAL LOGIC
    # ==============================================================

    def compute(self, field_type='total', domain_name=None):
        if domain_name is not None:
            self._compute_on_domain(field_type, domain_name)
        else:
            self._compute_full_grid(field_type)

    def _compute_full_grid(self, field_type):
        # print("Computing incident field on full grid...")
        self.U_inc = self.compute_incident_field(self.rho_grid)

        if field_type == 'incident':
            self.U = self.U_inc.copy()
            return

        rho_obj, chi_obj, u_inc_obj, dV, kb_s, has_sources = self._get_scattering_sources()

        if not has_sources:
            print("No contrast defined — scattered field is zero.")
            self.U_sc_grid = np.zeros_like(self.U_inc, dtype=complex)
            self.U = self.U_inc.copy()
            return

        rho_flat = self.rho_grid.reshape(-1, 2)
        u_sc_flat = np.zeros(len(rho_flat), dtype=complex)
        chunk_size = 5000

        for start in range(0, len(rho_flat), chunk_size):
            end = min(start + chunk_size, len(rho_flat))
            u_sc_flat[start:end] = self._compute_scattered_field_at_points(
                rho_flat[start:end], rho_obj, chi_obj, u_inc_obj, kb_s, dV
            )
            print(f"  {end}/{len(rho_flat)} ({100 * end / len(rho_flat):.1f}%)")

        self.U_sc_grid = u_sc_flat.reshape(self.X.shape)

        if field_type == 'scattered':
            self.U = self.U_sc_grid.copy()
        else:
            self.U = self.U_inc + self.U_sc_grid

    def _compute_on_domain(self, field_type, domain_name):
        assert domain_name in self.domains, f"Domain '{domain_name}' not found."

        mask = self.domains[domain_name]['mask']
        rho_dom = self.rho_grid[mask]

        u_inc_dom = self.compute_incident_field(rho_dom)

        # Store full incident field (needed for correct scattering sources)
        self.U_inc = self.compute_incident_field(self.rho_grid)

        if field_type == 'incident':
            U_inc_domain_only = np.zeros_like(self.X, dtype=complex)
            U_inc_domain_only[mask] = u_inc_dom
            self.U = U_inc_domain_only
            return

        rho_obj, chi_obj, u_inc_obj, dV, kb_s, has_sources = self._get_scattering_sources()

        if not has_sources:
            self.U_sc_grid = np.zeros_like(self.X, dtype=complex)
            U_inc_domain_only = np.zeros_like(self.X, dtype=complex)
            U_inc_domain_only[mask] = u_inc_dom
            self.U = U_inc_domain_only
            return

        u_sc_dom = self._compute_scattered_field_at_points(
            rho_dom, rho_obj, chi_obj, u_inc_obj, kb_s, dV
        )

        self.U_sc_grid = np.zeros_like(self.X, dtype=complex)
        self.U_sc_grid[mask] = u_sc_dom

        U_inc_domain_only = np.zeros_like(self.X, dtype=complex)
        U_inc_domain_only[mask] = u_inc_dom

        if field_type == 'scattered':
            self.U = self.U_sc_grid.copy()
        else:
            self.U = U_inc_domain_only + self.U_sc_grid

    def compute_incident_field(self, rho):
        field = np.zeros(rho.shape[:-1], dtype=complex)

        for name, s in self.sources.items():
            if isinstance(s, dict):
                intensity = s.get('intensity', 1.0)
                kb_s = s.get('kb', self.kb)
                source_type = s.get('source_type', 'point')
                directivity = s.get('directivity', 'isotropic')
                theta_0 = s.get('theta_0', 0)
                n_beam = s.get('n_beam', 1)
            else:
                intensity = 1.0
                kb_s = self.kb
                source_type = 'point'
                directivity = 'isotropic'
                theta_0 = 0
                n_beam = 1

            if source_type == 'point':
                rho_s = s['rho'] if isinstance(s, dict) else s
                R = np.linalg.norm(rho - rho_s, axis=-1)
                R = np.maximum(R, 1e-10)
                base_field = -1j / 4 * hankel2(0, kb_s * R)

                if directivity != 'isotropic':
                    theta = np.arctan2(rho[..., 1] - rho_s[1],
                                       rho[..., 0] - rho_s[0])
                    if directivity == 'beam':
                        D = np.maximum(np.cos(theta - theta_0), 0) ** n_beam
                        base_field *= D
                    elif directivity == 'cardioid':
                        D = (1 + np.cos(theta - theta_0)) / 2
                        base_field *= D

                field += intensity * base_field

            elif source_type == 'line':
                rho_start = np.array(s['rho'])
                rho_end = np.array(s['rho_end'])
                num_points = s.get('num_points', 20)

                t = np.linspace(0, 1, num_points)
                line_points = rho_start + t[:, np.newaxis] * (rho_end - rho_start)
                ds = np.linalg.norm(rho_end - rho_start) / num_points

                for rho_p in line_points:
                    R = np.linalg.norm(rho - rho_p, axis=-1)
                    R = np.maximum(R, 1e-10)
                    field += intensity * ds * (-1j / 4 * hankel2(0, kb_s * R))

        return field

    def compute_scattered_field(self, source_indices=None):
        """
        Compute scattered field at receivers from specified sources.

        Parameters
        ----------
        source_indices : list or None
            Which sources to use. If None, uses all.
        """
        assert self.U_inc is not None, "Run compute('incident') first."
        assert self.Chi is not None, "No contrast defined."

        source_names = list(self.sources.keys())
        if source_indices is not None:
            source_names = [source_names[i] for i in source_indices
                            if i < len(source_names)]

        mask_chi = (self.Chi > 0)
        if not np.any(mask_chi):
            print("Warning: No contrast. Scattered field is zero.")
            for rec_name in self.receivers:
                self.U_sc[rec_name] = np.zeros(self.receivers[rec_name]['M_total'] * len(source_names),
                                               dtype=complex)
            return self.U_sc

        rho_obj = self.rho_grid[mask_chi]
        chi_obj = self.Chi[mask_chi]
        u_inc_obj = self.U_inc[mask_chi]
        dV = self.stepsize ** 2

        self.U_sc = {}

        for rec_name, rec in self.receivers.items():
            rho_R = rec['positions']
            M = rec['M_total']

            u_sc_all = []

            for src_name in source_names:
                s = self.sources[src_name]
                kb_s = s['kb'] if isinstance(s, dict) else self.kb

                u_sc = np.zeros(M, dtype=complex)

                for i in range(M):
                    R = np.linalg.norm(rho_R[i] - rho_obj, axis=1)
                    R = np.maximum(R, self.stepsize / 2)
                    G = -1j / 4 * hankel2(0, kb_s * R)
                    u_sc[i] = kb_s ** 2 * np.sum(G * chi_obj * u_inc_obj) * dV

                u_sc_all.append(u_sc)

            # Stack all source measurements
            self.U_sc[rec_name] = np.concatenate(u_sc_all) if len(u_sc_all) > 1 else u_sc_all[0]

        return self.U_sc

    def _compute_scattered_field_at_points(self, rho_obs, rho_obj, chi_obj, u_inc_obj, kb_s, dV):
        M = len(rho_obs)
        u_sc = np.zeros(M, dtype=complex)

        for i in range(M):
            R = np.linalg.norm(rho_obs[i] - rho_obj, axis=1)
            R = np.maximum(R, self.stepsize / 2)  # regularize: avoid hankel2(0,0)=inf
            G = -1j / 4 * hankel2(0, kb_s * R)
            u_sc[i] = kb_s ** 2 * np.sum(G * chi_obj * u_inc_obj) * dV

        return u_sc

    def _get_scattering_sources(self):
        if self.Chi is None or not np.any(self.Chi > 0):
            return None, None, None, self.stepsize ** 2, self.kb, False

        mask_chi = (self.Chi > 0)
        rho_obj = self.rho_grid[mask_chi]
        chi_obj = self.Chi[mask_chi]

        if self.U_inc is None:
            self.U_inc = self.compute_incident_field(self.rho_grid)

        u_inc_obj = self.U_inc[mask_chi]
        dV = self.stepsize ** 2

        s = list(self.sources.values())[0]
        kb_s = s['kb'] if isinstance(s, dict) else self.kb

        return rho_obj, chi_obj, u_inc_obj, dV, kb_s, True

    def green_function(self, rho, rho_prime, kb):
        R = np.linalg.norm(rho - rho_prime, axis=-1)

        if self.boundary == 'free':
            return -1j / 4 * hankel2(0, kb * R)

        elif self.boundary == 'closed':
            x, y = rho[..., 0], rho[..., 1]
            xp, yp = rho_prime[0], rho_prime[1]

            G = np.zeros_like(x, dtype=complex)

            for m in range(1, self.M_modes + 1):
                for n in range(1, self.N_modes + 1):
                    k_mn_sq = (m * np.pi / self.a) ** 2 + (n * np.pi / self.b) ** 2

                    phi_mn = np.sqrt(4 / (self.a * self.b))
                    phi_at_rho = phi_mn * np.sin(m * np.pi * x / self.a) * np.sin(n * np.pi * y / self.b)
                    phi_at_source = phi_mn * np.sin(m * np.pi * xp / self.a) * np.sin(n * np.pi * yp / self.b)

                    G += phi_at_rho * phi_at_source / (kb ** 2 - k_mn_sq - 1j * 1e-6)

            return G

    def _setup_closed_space(self):
        self.a = self.x_range[1] - self.x_range[0]
        self.b = self.y_range[1] - self.y_range[0]
        self.M_modes = 50
        self.N_modes = 50

    # ==============================================================

    def solve_inverse(self, domain_name, receiver_name, method='pinv',
                      use_noisy=False, alpha=0.01, K=None, plot_svd=True,
                      matrix_method='born', reflection_coeff=0.0,
                      source_indices=None, beamformer_source='none'):

        A = self.build_system_matrix(domain_name, receiver_name,
                                     method=matrix_method,
                                     reflection_coeff=reflection_coeff,
                                     source_indices=source_indices,
                                     beamformer_source=beamformer_source)

        # Get scattered field data
        if use_noisy:
            if not hasattr(self, 'U_sc_noisy') or receiver_name not in self.U_sc_noisy:
                raise ValueError("No noisy data. Call add_noise() first.")
            u_sc = self.U_sc_noisy[receiver_name]
            print(f"  Using NOISY data (SNR={self.noise_info.get('SNR_dB', '?')} dB)")
        else:
            u_sc = self.U_sc[receiver_name]
            print(f"  Using CLEAN data")

        # Validate
        if np.any(np.isnan(u_sc)) or np.any(np.isinf(u_sc)):
            print("  WARNING: NaN/Inf in u_sc!")

        # SVD
        try:
            U_svd, S, Vh = np.linalg.svd(A, full_matrices=False)
        except np.linalg.LinAlgError:
            print("  SVD failed, adding regularization...")
            A_reg = A.T @ A + 1e-8 * np.eye(A.shape[1])
            U_svd, S, Vh = np.linalg.svd(A_reg, full_matrices=False)

        if plot_svd:
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].semilogy(S, 'b.-', markersize=4)
            ax[0].set_xlabel('Index k');
            ax[0].set_ylabel('σ_k')
            ax[0].set_title(f'SVD (M={A.shape[0]}, N={A.shape[1]})')
            ax[0].grid(True, alpha=0.3)

            cumsum = np.cumsum(S ** 2) / np.sum(S ** 2)
            ax[1].plot(cumsum, 'r.-')
            ax[1].axhline(0.95, color='gray', linestyle='--')
            ax[1].axhline(0.99, color='gray', linestyle='--')
            ax[1].set_xlabel('Index k');
            ax[1].set_ylabel('Cumulative energy')
            ax[1].grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Solve
        if method == 'pinv':
            # Use 1% of max singular value as threshold, keeps physically meaningful
            # components only and avoids amplifying noise through near-zero singular values
            tol = 1e-2 * S[0]
            S_inv = np.where(S > tol, 1 / S, 0)
            print(f"  Pinv: {np.sum(S > tol)}/{len(S)} singular values kept")
        elif method == 'tsvd':
            K = min(K or len(S) // 2, len(S))
            S_inv = np.zeros(len(S))
            S_inv[:K] = 1 / S[:K]
            print(f"  TSVD: K={K}/{len(S)}")
        elif method == 'auto_tsvd':
            threshold = 0.01 * S[0]
            K = max(np.sum(S > threshold), 1)
            S_inv = np.zeros(len(S))
            S_inv[:K] = 1 / S[:K]
            print(f"  Auto TSVD: K={K}/{len(S)}")
        elif method == 'tikhonov':
            # alpha should be relative to signal scale: use alpha * S[0]^2 as the
            # regularization parameter so it scales automatically with A's energy
            alpha_scaled = alpha * S[0] ** 2
            S_inv = S / (S ** 2 + alpha_scaled)
            print(f"  Tikhonov: alpha={alpha} (scaled={alpha_scaled:.2e})")

        chi_recon = Vh.T @ np.diag(S_inv) @ U_svd.conj().T @ u_sc
        chi_recon = np.real(chi_recon)

        # Clamp extreme values
        if np.abs(chi_recon).max() > 100:
            print(f"  WARNING: chi exploding, clamping...")
            chi_recon = np.clip(chi_recon, -10, 10)

        print(f"  Chi range: [{chi_recon.min():.4f}, {chi_recon.max():.4f}]")

        return chi_recon

    def build_system_matrix(self, domain_name, receiver_name, method='born',
                            reflection_coeff=0.0, source_indices=None,
                            beamformer_source='none'):
        """
        Build system matrix A: u_sc = A @ chi_vec.

        Parameters
        ----------
        domain_name : str
        receiver_name : str
        method : str
            'born', 'rytov', or 'total'
        reflection_coeff : float
            Fresnel reflection coefficient (0 to 1).
        source_indices : list or None
            Which sources to use. None = all.
        beamformer_source : str
            Source-side beamforming: 'none', 'delay_and_sum', 'mvdr'

        Receiver beamforming is read from the receiver's own parameters:
            rec['beamformer']: 'none', 'delay_and_sum', 'mvdr'
            rec['steer_angle']: steering angle in degrees
        """
        d = self.domains[domain_name]
        mask = d['mask']

        X_dom = self.X[mask]
        Y_dom = self.Y[mask]
        rho_dom = np.column_stack([X_dom.ravel(), Y_dom.ravel()])
        N = len(rho_dom)

        # Domain center for source beamforming focus point
        rho_focus = np.array([X_dom.mean(), Y_dom.mean()])

        rec = self.receivers[receiver_name]
        rho_R = rec['positions']
        M_total = rec['M_total']

        # Receiver parameters
        rec_pattern = rec.get('pattern', 'isotropic')
        rec_theta_0 = rec.get('theta_0', 0)
        n_beam_rec = rec.get('n_beam', 1)

        # Compute weights for receiver beamformers
        rec_beamformer = rec.get('beamformer', 'none')
        rec_steer_angle = rec.get('steer_angle', 0)

        if rec_beamformer == 'delay_and_sum':
            angle = np.radians(rec_steer_angle)
            k = 2 * np.pi / self.mu
            steering = np.exp(-1j * k * (rho_R[:, 0] * np.cos(angle) +
                                         rho_R[:, 1] * np.sin(angle)))
            w_rec_bf = steering / M_total
            print(f"  Receiver beamformer: delay_and_sum @ {rec_steer_angle}°")

        # Check for ESPRIT+MVDR pre-computed weights
        if rec.get('beamformer') == 'esprit_mvdr' and '_mvdr_weights' in rec:
            w_rec_bf = rec['_mvdr_weights']

        elif rec_beamformer == 'mvdr':
            angle = np.radians(rec_steer_angle)
            k = 2 * np.pi / self.mu
            steering = np.exp(-1j * k * (rho_R[:, 0] * np.cos(angle) +
                                         rho_R[:, 1] * np.sin(angle)))

            # Estimate covariance from U_sc if available
            if hasattr(self, 'U_sc') and receiver_name in self.U_sc:
                u_sc = self.U_sc[receiver_name]
                R_cov = np.outer(u_sc, u_sc.conj())
                R_reg = R_cov + 0.01 * np.eye(M_total) * np.trace(R_cov) / M_total
                try:
                    R_inv = np.linalg.inv(R_reg)
                    w_rec_bf = R_inv @ steering / (steering.conj() @ R_inv @ steering)
                except np.linalg.LinAlgError:
                    w_rec_bf = steering / M_total  # Fallback
            else:
                w_rec_bf = steering / M_total  # Fallback to DAS
            print(f"  Receiver beamformer: MVDR @ {rec_steer_angle}°")
        else:
            w_rec_bf = np.ones(M_total)  # Uniform (no beamforming)

        dV = self.stepsize ** 2

        # Determine sources
        source_names = list(self.sources.keys())
        if source_indices is not None:
            source_names = [source_names[i] for i in source_indices if i < len(source_names)]

        n_sources = len(source_names)

        # Build A blocks for each source
        A_blocks = []

        for src_name in source_names:
            s = self.sources[src_name]
            kb_s = s['kb'] if isinstance(s, dict) else self.kb
            source_type = s.get('source_type', 'point') if isinstance(s, dict) else 'point'

            # Choose field inside object
            if method == 'born' or method == 'rytov':
                u_field = self.U_inc[mask].ravel() if self.U_inc is not None else self.compute_incident_field(rho_dom)
            elif method == 'total':
                u_field = self.U[mask].ravel()
            else:
                raise ValueError(f"Unknown method: '{method}'.")

            A_src = np.zeros((M_total, N), dtype=complex)
            A_refl = np.zeros((M_total, N), dtype=complex) if reflection_coeff > 0 else None

            # Compute source beamformer weights
            if beamformer_source == 'delay_and_sum':
                if source_type == 'point':
                    rho_s = s['rho'] if isinstance(s, dict) else s
                    tau_focus = np.linalg.norm(rho_focus - rho_s)
                    beam_weight_src = np.exp(1j * kb_s * tau_focus)
                elif source_type == 'line':
                    rho_start = np.array(s['rho'])
                    rho_end = np.array(s['rho_end'])
                    rho_mid = (rho_start + rho_end) / 2
                    tau_focus = np.linalg.norm(rho_focus - rho_mid)
                    beam_weight_src = np.exp(1j * kb_s * tau_focus)
            elif beamformer_source == 'mvdr':
                if source_type == 'point':
                    rho_s = s['rho'] if isinstance(s, dict) else s
                    R_focus = np.linalg.norm(rho_focus - rho_s)
                    beam_weight_src = np.exp(1j * kb_s * R_focus)
            else:
                beam_weight_src = 1.0

            # Build measurements matrix row-wise
            if source_type == 'point':
                rho_s = s['rho'] if isinstance(s, dict) else s

                for i in range(M_total):
                    R_vec = rho_R[i] - rho_dom
                    R = np.linalg.norm(R_vec, axis=1)
                    G = -1j / 4 * hankel2(0, kb_s * R)

                    # Per-element receiver directivity
                    G = self._apply_receiver_directivity(G, R_vec, rec_pattern, rec_theta_0, n_beam_rec)

                    # Rytov normalization
                    if method == 'rytov':
                        u_inc_R = -1j / 4 * hankel2(0, kb_s * np.linalg.norm(rho_R[i] - rho_s))
                        G = G / u_inc_R

                    A_src[i, :] = kb_s ** 2 * G * u_field * dV * beam_weight_src

                    # Reflection term
                    if reflection_coeff > 0:
                        for j in range(N):
                            R1 = np.linalg.norm(rho_s - rho_dom[j])
                            u_inc_j = -1j / 4 * hankel2(0, kb_s * max(R1, self.stepsize / 2))
                            A_refl[i, j] = kb_s ** 2 * reflection_coeff * G[j] * u_inc_j * dV * beam_weight_src

            elif source_type == 'line':
                rho_start = np.array(s['rho'])
                rho_end = np.array(s['rho_end'])
                num_pts = s.get('num_points', 20) if isinstance(s, dict) else 20

                t = np.linspace(0, 1, num_pts)
                line_points = rho_start + t[:, np.newaxis] * (rho_end - rho_start)
                ds = np.linalg.norm(rho_end - rho_start) / num_pts

                for i in range(M_total):
                    G_total = np.zeros(N, dtype=complex)

                    for rho_p in line_points:
                        R_vec_p = rho_R[i] - rho_dom
                        R_p = np.linalg.norm(R_vec_p, axis=1)
                        G_p = -1j / 4 * hankel2(0, kb_s * R_p)
                        G_total += G_p * ds

                    G_total = self._apply_receiver_directivity(G_total, R_vec_p, rec_pattern, rec_theta_0, n_beam_rec)
                    A_src[i, :] = kb_s ** 2 * G_total * u_field * dV * beam_weight_src

            if reflection_coeff > 0:
                A_src += A_refl

            # Weight each row (receiver) by the beamformer coefficient
            A_src_weighted = np.diag(w_rec_bf) @ A_src  # (M, M) @ (M, N) = (M, N)

            A_blocks.append(A_src_weighted)

        # Stack all sources
        A = np.vstack(A_blocks) if len(A_blocks) > 1 else A_blocks[0]

        if n_sources > 1:
            print(f"  Built A: {n_sources} sources → shape {A.shape}")
        if reflection_coeff > 0:
            print(f"  Added reflection (R={reflection_coeff:.2f})")
        if beamformer_source != 'none':
            print(f"  Source beamformer: {beamformer_source}")

        return A

    def _apply_receiver_directivity(self, G, R_vec, pattern, theta_0, n_beam):
        """Apply receiver directivity pattern to Green's function."""
        if pattern == 'isotropic':
            return G
        elif pattern == 'beam':
            theta = np.arctan2(R_vec[:, 1], R_vec[:, 0])
            D = np.maximum(np.cos(theta - theta_0), 0) ** n_beam
            return G * D
        elif pattern == 'cardioid':
            theta = np.arctan2(R_vec[:, 1], R_vec[:, 0])
            D = (1 + np.cos(theta - theta_0)) / 2
            return G * D
        return G

    def build_system_matrix_rytov(self, domain_name, receiver_name):
        """Rytov approximation: linearize in phase, not amplitude."""
        d = self.domains[domain_name]
        mask = d['mask']

        X_dom = self.X[mask]
        Y_dom = self.Y[mask]
        rho_dom = np.column_stack([X_dom.ravel(), Y_dom.ravel()])
        N = len(rho_dom)

        rec = self.receivers[receiver_name]
        rho_R = rec['positions']
        M_total = rec['M_total']

        s = list(self.sources.values())[0]
        kb_s = s['kb'] if isinstance(s, dict) else self.kb

        if self.U_inc is not None:
            u_inc_dom = self.U_inc[mask].ravel()
        else:
            u_inc_dom = self.compute_incident_field(rho_dom)

        # Rytov: linearize phi = ln(u_total/u_inc)
        # The data becomes the complex phase: phi_R = ln(u_total/u_inc) at receivers
        dV = self.stepsize ** 2
        A = np.zeros((M_total, N), dtype=complex)

        for i in range(M_total):
            R = np.linalg.norm(rho_R[i] - rho_dom, axis=1)
            G = -1j / 4 * hankel2(0, kb_s * R)
            # Rytov kernel includes division by u_inc at observation point
            u_inc_R = self.compute_incident_field(rho_R[i].reshape(1, -1))
            A[i, :] = kb_s ** 2 * G * u_inc_dom / u_inc_R * dV

        return A

    def solve_inverse_dbim(self, domain_name, receiver_name, max_iter=10, tol=1e-3, K=15,
                           source_indices=None):
        """Distorted Born Iterative Method."""
        d = self.domains[domain_name]
        mask = d['mask']

        source_names = list(self.sources.keys())
        if source_indices is not None:
            source_names = [source_names[i] for i in source_indices if i < len(source_names)]
        n_sources = len(source_names)

        print(f"DBIM: {n_sources} sources, max {max_iter} iterations")

        # Initial Born for single source
        chi_k = self.solve_inverse(domain_name, receiver_name, method='tsvd', K=K,
                                   plot_svd=False, source_indices=source_indices)
        chi_k = np.maximum(chi_k, 0)  # positivity

        u_sc_full = self.U_sc[receiver_name]

        # Store original state
        chi_original = self.Chi.copy() if self.Chi is not None else np.zeros_like(self.X)
        U_inc_original = self.U_inc.copy() if self.U_inc is not None else None

        for iteration in range(max_iter):
            print(f"\nDBIM Iteration {iteration + 1}/{max_iter}")

            # Update chi
            chi_2d = np.zeros_like(self.X)
            chi_2d[mask] = chi_k
            self.Chi = chi_2d

            # Compute total field ONLY on domain points
            rho_dom = self.rho_grid[mask]
            u_inc_dom = self.compute_incident_field(rho_dom)

            # Compute scattered field on domain using current chi
            mask_chi = (self.Chi > 0)
            if np.any(mask_chi):
                rho_obj = self.rho_grid[mask_chi]
                chi_obj = self.Chi[mask_chi]
                u_inc_obj = self.compute_incident_field(rho_obj)
                dV = self.stepsize ** 2

                s = self.sources[source_names[0]]
                kb_s = s['kb'] if isinstance(s, dict) else self.kb

                u_sc_dom = np.zeros(len(rho_dom), dtype=complex)
                for i, rho_R in enumerate(rho_dom):
                    R = np.linalg.norm(rho_R - rho_obj, axis=1)
                    R = np.maximum(R, self.stepsize / 2)
                    G = -1j / 4 * hankel2(0, kb_s * R)
                    u_sc_dom[i] = kb_s ** 2 * np.sum(G * chi_obj * u_inc_obj) * dV

                u_total_dom = u_inc_dom + u_sc_dom
            else:
                u_total_dom = u_inc_dom

            # Validate
            if np.any(np.isnan(u_total_dom)) or np.any(np.isinf(u_total_dom)):
                print("  NaN/Inf in total field! Reverting to incident field.")
                u_total_dom = u_inc_dom

            # Build distorted A using total field on domain
            U_inc_temp = self.U_inc.copy()
            U_temp = np.zeros_like(self.U_inc, dtype=complex)
            U_temp[mask] = u_total_dom
            self.U_inc = U_temp  # Consider total field as incident for building A

            A_distorted = self.build_system_matrix(domain_name, receiver_name,
                                                   method='born',
                                                   source_indices=source_indices)

            self.U_inc = U_inc_temp  # Restore

            # Validate A
            if np.any(np.isnan(A_distorted)) or np.any(np.isinf(A_distorted)):
                print("  NaN/Inf in A_distorted! Skipping iteration.")
                break

            # Compute residual
            residual = u_sc_full - A_distorted @ chi_k
            residual_norm = np.linalg.norm(residual) / (np.linalg.norm(u_sc_full) + 1e-10)

            # Solve for update using LSQR (more stable than SVD for large matrices)
            from scipy.sparse.linalg import lsqr
            damping = 0.01

            # Build regularized least squares
            A_aug = np.vstack([A_distorted, damping * np.eye(A_distorted.shape[1])])
            res_aug = np.concatenate([residual, np.zeros(A_distorted.shape[1])])

            delta_chi, _, _, _ = np.linalg.lstsq(A_aug, res_aug, rcond=None)
            delta_chi = np.real(delta_chi)

            # Damped update
            step_size = 0.3
            chi_new = chi_k + step_size * delta_chi
            chi_new = np.maximum(chi_new, 0)
            chi_new = np.minimum(chi_new, 2.0)  # Upper bound

            change = np.linalg.norm(chi_new - chi_k) / (np.linalg.norm(chi_k) + 1e-10)

            print(f"  Residual: {residual_norm:.4f}, Change: {change:.4f}")
            print(f"  Chi range: [{chi_new.min():.4f}, {chi_new.max():.4f}]")

            chi_k = chi_new

            if change < tol:
                print("  Converged!")
                break

        # Restore
        self.Chi = chi_original
        self.U_inc = U_inc_original

        return chi_k

    def build_data_matrix_from_sources(self, receiver_name, source_indices=None):
        """
        Build data matrix X where each column is the scattered field
        from one source at all receivers.

        X.shape = (M, N_sources) — perfect for ESPRIT!

        This requires computing U_sc separately for each source.
        """
        rec = self.receivers[receiver_name]
        M = rec['M_total']

        source_names = list(self.sources.keys())
        if source_indices is not None:
            source_names = [source_names[i] for i in source_indices if i < len(source_names)]

        n_sources = len(source_names)
        X = np.zeros((M, n_sources), dtype=complex)

        # Backup original sources and Chi
        original_sources = self.sources.copy()
        chi_backup = self.Chi.copy() if self.Chi is not None else None

        for col, src_name in enumerate(source_names):
            # Use only this one source
            self.sources = {src_name: original_sources[src_name]}

            # Compute incident field
            self.compute(field_type='incident')

            # Compute scattered field at receivers
            self.compute_scattered_field()

            # Store as column of X
            X[:, col] = self.U_sc[receiver_name]

        # Restore
        self.sources = original_sources
        if chi_backup is not None:
            self.Chi = chi_backup

        return X, source_names

    # ==============================================================

    def esprit_mvdr_reconstruct(self, domain_name, receiver_name,
                                source_indices=None, d=1, K=15, element_spacing=.5):
        """
        1. Build data matrix X from sequential source firings
        2. ESPRIT → estimate object angle(s)
        3. MVDR → beamform receivers toward object
        4. Build A with MVDR weights → solve for chi
        """
        # Step 1: Build data matrix X (each column corresponds to one source)
        X, _ = self._build_esprit_data_matrix(receiver_name, source_indices)

        # Step 2: ESPRIT
        angles = esprit(X, d=1, element_spacing=element_spacing, wavelength=self.mu)
        steer_angle = angles[0]
        print(f"ESPRIT estimated angle: {steer_angle:.1f}°")

        # Step 3: Compute MVDR weights from the scattered field
        self.compute(field_type='incident')
        self.compute_scattered_field()
        u_sc = self.U_sc[receiver_name]

        # Use only the first source block for MVDR weight computation
        rec = self.receivers[receiver_name]
        M = rec['M_total']
        u_sc_single = u_sc[:M]

        w_mvdr = mvdr_beamformer(u_sc_single, rec['positions'], self.mu, steer_angle)

        # Step 4: Store weights for build_system_matrix
        rec['_mvdr_weights'] = w_mvdr
        rec['beamformer'] = 'esprit_mvdr'

        # Step 5: Build A and solve
        chi = self.solve_inverse(domain_name, receiver_name, method='tsvd', K=K,
                                 source_indices=source_indices)

        rec['_mvdr_weights'] = None
        rec['beamformer'] = 'none'
        return np.real(chi), steer_angle

    def _build_esprit_data_matrix(self, receiver_name, source_indices=None):
        """
        Build M × N_sources data matrix for ESPRIT.
        Each column = scattered field from one source firing alone.
        """
        rec = self.receivers[receiver_name]
        M = rec['M_total']

        source_names = list(self.sources.keys())
        if source_indices is not None:
            source_names = [source_names[i] for i in source_indices if i < len(source_names)]

        n_sources = len(source_names)
        X = np.zeros((M, n_sources), dtype=complex)

        # Backup
        original_sources = dict(self.sources)

        for col, src_name in enumerate(source_names):
            self.sources = {src_name: original_sources[src_name]}
            self.compute(field_type='incident')
            self.compute_scattered_field()
            X[:, col] = self.U_sc[receiver_name][:M]

        # Restore all sources and recompute
        self.sources = original_sources
        self.compute(field_type='incident')
        self.compute_scattered_field()

        return X, source_names

    # ==============================================================
    # CONFIG LOGIC
    # ==============================================================

    def add_noise(self, receiver_name, SNR_dB=20, noise_type='gaussian'):
        """Add noise to the scattered field at receivers."""
        u_sc = self.U_sc[receiver_name]
        M = len(u_sc)

        # Signal power (average over receivers)
        P_signal = np.mean(np.abs(u_sc) ** 2)

        # Convert SNR from dB to linear
        SNR_linear = 10 ** (SNR_dB / 10)
        P_noise = P_signal / SNR_linear

        if noise_type == 'gaussian':
            noise_std = np.sqrt(P_noise / 2)
            noise = noise_std * (np.random.randn(M) + 1j * np.random.randn(M))
        elif noise_type == 'uniform':
            phase_noise = np.random.uniform(-np.pi, np.pi, M)
            noise = np.sqrt(P_noise) * np.exp(1j * phase_noise)

        u_sc_noisy = u_sc + noise

        # Store as simple dict
        if not hasattr(self, 'U_sc_noisy'):
            self.U_sc_noisy = {}
        self.U_sc_noisy[receiver_name] = u_sc_noisy

        # Store noise info
        self.noise_info = {
            'SNR_dB': SNR_dB,
            'P_signal': P_signal,
            'P_noise': P_noise,
            'noise_std': noise_std if noise_type == 'gaussian' else None,
            'noise_type': noise_type
        }

        print(f"  SNR = {SNR_dB} dB, P_signal = {P_signal:.2e}, P_noise = {P_noise:.2e}")

        return u_sc_noisy

    def add_noise_per_element(self, receiver_name, noise_figure_dB=3, T=290, bandwidth=1e6):
        k_B = 1.38e-23

        u_sc = self.U_sc[receiver_name]
        M = len(u_sc)

        NF_linear = 10 ** (noise_figure_dB / 10)
        P_noise = k_B * T * bandwidth * NF_linear

        noise_std = np.sqrt(P_noise / 2)
        noise = noise_std * (np.random.randn(M) + 1j * np.random.randn(M))

        u_sc_noisy = u_sc + noise

        P_signal = np.mean(np.abs(u_sc) ** 2)
        SNR_dB = 10 * np.log10(P_signal / P_noise)

        if not hasattr(self, 'U_sc_noisy'):
            self.U_sc_noisy = {}
        self.U_sc_noisy[receiver_name] = u_sc_noisy

        return u_sc_noisy

    # ==============================================================

    def add_domain(self, name, rho, width, height, label_pos='above'):
        rho = np.array(rho)

        self.domains[name] = {
            'rho': rho,
            'width': width,
            'height': height,
            'mask': None,
            'label_pos': label_pos
        }

        self._check_and_pad()

        self.domains[name]['mask'] = (
                (self.X >= rho[0]) & (self.X <= rho[0] + width) &
                (self.Y >= rho[1]) & (self.Y <= rho[1] + height)
        )

    def get_domain_field(self, name):
        mask = self.domains[name]['mask']
        return self.U[mask]

    def get_domain_points(self, name):
        mask = self.domains[name]['mask']
        return self.X[mask], self.Y[mask]

    # ==============================================================

    def add_source(self, name, rho, intensity=1.0, kb=None,
                   source_type='point', rho_end=None, num_points=20,
                   directivity='isotropic', theta_0=0, n_beam=1):
        if kb is None:
            kb = self.kb

        rho = np.array(rho)
        if source_type == 'line' and rho_end is None:
            raise ValueError("rho_end required for line sources")

        self.sources[name] = {
            'rho': rho,
            'intensity': intensity,
            'kb': kb,
            'mu': 2 * np.pi / kb,
            'source_type': source_type,
            'rho_end': np.array(rho_end) if rho_end is not None else None,
            'num_points': num_points,
            'directivity': directivity,
            'theta_0': theta_0,
            'n_beam': n_beam
        }

        self._check_and_pad()

    # ==============================================================

    def add_contrast(self, name, rho_0, size, intensity, shape='rectangle', **kwargs):
        shape_funcs = {
            'rectangle': make_rectangle,
            'triangle': make_triangle,
            'star': make_star,
            'letter': make_letter,
            'circle': make_circle,
            'ellipse': make_ellipse
        }

        if shape not in shape_funcs:
            raise ValueError(f"Unknown shape '{shape}'. Choose from: {list(shape_funcs.keys())}")

        func = shape_funcs[shape]
        Chi_shape, patches_list = func(self.X, self.Y, rho_0, size, intensity, **kwargs)

        self.contrasts[name] = {
            'rho_0': np.array(rho_0),
            'size': size,
            'intensity': intensity,
            'shape': shape,
            'Chi': Chi_shape,
            'patches': patches_list,
            **kwargs
        }

        self._update_chi()
        self._check_and_pad()

    def _update_chi(self):
        if not self.contrasts:
            self.Chi = np.zeros_like(self.X)
        else:
            self.Chi = np.sum([c['Chi'] for c in self.contrasts.values()], axis=0)

    def remove_contrast(self, name):
        if name in self.contrasts:
            del self.contrasts[name]
            self._update_chi()

    def get_contrast(self, name):
        assert self.Chi is not None, "No contrast defined."
        assert name in self.domains, f"Domain '{name}' not found."
        mask = self.domains[name]['mask']
        return self.Chi[mask]

    def get_contrast_vector(self, name):
        assert self.Chi is not None, "No contrast defined."
        assert name in self.domains, f"Domain '{name}' not found."
        mask = self.domains[name]['mask']
        return self.Chi[mask]

    # ==============================================================

    def add_receiver_array(self, name, rho_start=None, rho_end=None, M=20, N=None,
                           geometry='line', rho_center=[0, 0], radius=1,
                           theta_start=0, theta_end=180,
                           pattern='isotropic', theta_0_rec=0, n_beam_rec=1,
                           beamformer='none', steer_angle_deg=0):
        """
        Add a receiver array.

        Parameters
        ----------
        pattern : str
            'isotropic', 'beam', 'cardioid' — per-element directivity
        theta_0_rec : float
            Directivity pointing angle (radians)
        n_beam_rec : float
            Beam width parameter
        beamformer : str
            'none' — no beamforming
            'delay_and_sum' — phase-align to steer_angle
            'mvdr' — adaptive beamforming (requires noise covariance)
        steer_angle_deg : float
            Steering angle for beamformer (degrees, 0° = +x, 90° = +y)
        """
        if geometry == 'arc' or geometry == 'circle':
            if rho_center is None:
                raise ValueError("rho_center required for arc/circle")
            if radius is None:
                raise ValueError("radius required for arc/circle")

            rho_center = np.array(rho_center)

            if geometry == 'circle':
                theta_start = 0
                theta_end = 360

            theta = np.linspace(np.radians(theta_start), np.radians(theta_end), M)
            x = rho_center[0] + radius * np.cos(theta)
            y = rho_center[1] + radius * np.sin(theta)
            positions = np.column_stack([x, y])

            self.receivers[name] = {
                'type': 'arc',
                'rho_center': rho_center,
                'radius': radius,
                'theta_start': theta_start,
                'theta_end': theta_end,
                'M': M, 'N': None, 'M_total': M,
                'geometry': geometry,
                'positions': positions,
                'pattern': pattern,
                'theta_0': theta_0_rec,
                'n_beam': n_beam_rec,
                'beamformer': beamformer,
                'steer_angle': steer_angle_deg,
            }

        elif geometry == 'line':
            if rho_start is None or rho_end is None:
                raise ValueError("rho_start and rho_end required for line")

            rho_start = np.array(rho_start)
            rho_end = np.array(rho_end)
            t = np.linspace(0, 1, M)
            positions = rho_start + t[:, np.newaxis] * (rho_end - rho_start)

            self.receivers[name] = {
                'type': 'line',
                'rho_start': rho_start,
                'rho_end': rho_end,
                'M': M, 'N': None, 'M_total': M,
                'geometry': geometry,
                'positions': positions,
                'pattern': pattern,
                'theta_0': theta_0_rec,
                'n_beam': n_beam_rec,
                'beamformer': beamformer,
                'steer_angle': steer_angle_deg,
            }

        elif geometry == 'rect':
            if rho_start is None or rho_end is None:
                raise ValueError("rho_start and rho_end required for rect")
            if N is None:
                raise ValueError("N required for rect")

            rho_start = np.array(rho_start)
            rho_end = np.array(rho_end)
            x_vals = np.linspace(rho_start[0], rho_end[0], M)
            y_vals = np.linspace(rho_start[1], rho_end[1], N)
            X_rec, Y_rec = np.meshgrid(x_vals, y_vals)
            positions = np.column_stack([X_rec.ravel(), Y_rec.ravel()])

            self.receivers[name] = {
                'type': 'rect',
                'rho_start': rho_start,
                'rho_end': rho_end,
                'M': M, 'N': N, 'M_total': M * N,
                'geometry': geometry,
                'positions': positions,
                'pattern': pattern,
                'theta_0': theta_0_rec,
                'n_beam': n_beam_rec,
                'beamformer': beamformer,
                'steer_angle': steer_angle_deg,
            }

        else:
            raise ValueError(f"Unknown geometry '{geometry}'.")

        self._check_and_pad()

    def get_receiver_positions(self, name):
        return self.receivers[name]['positions']

    def get_receiver_count(self, name):
        return self.receivers[name]['M_total']

    def get_receiver_shape(self, name):
        rec = self.receivers[name]
        return rec['M'], rec['N']

    # ==============================================================
    # PLOTTING LOGIC
    # ==============================================================

    def plot_scene(self):
        fig, ax = make_grid_2D(x_range=self.x_range, y_range=self.y_range,
                               stepsize=self.stepsize, mu=self.mu)

        fig.set_size_inches(14, 8)
        handles = []

        for name, s in self.sources.items():
            if isinstance(s, dict):
                rho = s['rho']
                intensity = s.get('intensity', 1.0)
                kb_s = s.get('kb', self.kb)
                source_type = s.get('source_type', 'point')
                directivity = s.get('directivity', 'isotropic')
                rho_end = s.get('rho_end', None)
            else:
                rho = s
                intensity = 1.0
                kb_s = self.kb
                source_type = 'point'
                directivity = 'isotropic'
                rho_end = None

            if source_type == 'point':
                label = f'Source: {name} (I={intensity:.1f}, $k_b$={kb_s:.1f})'
                if directivity != 'isotropic':
                    label += f' [{directivity}]'

                proxy = Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='red', markeredgecolor='darkred',
                               markersize=10, label=label)
                handles.append(proxy)
                circle(ax, rho, label=name)

            elif source_type == 'line':
                label = f'Line Source: {name} (I={intensity:.1f}, $k_b$={kb_s:.1f})'

                ax.plot([rho[0], rho_end[0]], [rho[1], rho_end[1]],
                        'r-', linewidth=3, alpha=0.7, zorder=4)
                ax.plot(rho[0], rho[1], 'ro', markersize=6, zorder=5)
                ax.plot(rho_end[0], rho_end[1], 'ro', markersize=6, zorder=5)

                mid_x = (rho[0] + rho_end[0]) / 2
                mid_y = (rho[1] + rho_end[1]) / 2
                ax.annotate(name, xy=(mid_x, mid_y),
                            xytext=(10, -10), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='darkred',
                            ha='left', va='top',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                      edgecolor='darkred', alpha=0.8),
                            zorder=6)

                proxy = Line2D([0], [0], color='red', linewidth=3, label=label)
                handles.append(proxy)

        for name, d in self.domains.items():
            square(ax, d['rho'], d['width'], d['height'], label=name)
            from matplotlib.patches import Patch
            proxy = Patch(facecolor='none', edgecolor='blue', linewidth=2,
                          label=f'Domain: {name}')
            handles.append(proxy)

        for name, rec in self.receivers.items():
            positions = rec['positions']
            M_total = rec['M_total']
            geometry = rec.get('geometry', 'line')
            rec_type = rec.get('type', geometry)

            if rec_type == 'arc' or geometry == 'arc':
                ax.plot(positions[:, 0], positions[:, 1], 'g^',
                        markersize=8, markerfacecolor='green',
                        markeredgecolor='darkgreen', zorder=5)
                ax.plot(positions[:, 0], positions[:, 1], 'g-',
                        linewidth=1.5, alpha=0.5, zorder=4)

                proxy = Line2D([0], [0], marker='^', color='w',
                               markerfacecolor='green', markeredgecolor='darkgreen',
                               markersize=10,
                               label=f'Receiver Array: {name} (arc, M={M_total})')
                handles.append(proxy)

            elif rec_type == 'line' or geometry == 'line':
                ax.plot(positions[:, 0], positions[:, 1], 'gs',
                        markersize=8, markerfacecolor='green',
                        markeredgecolor='darkgreen', zorder=5)
                ax.plot(positions[:, 0], positions[:, 1], 'g--',
                        linewidth=1, alpha=0.5, zorder=4)

                proxy = Line2D([0], [0], marker='s', color='w',
                               markerfacecolor='green', markeredgecolor='darkgreen',
                               markersize=10,
                               label=f'Receiver Array: {name} (M={M_total})')
                handles.append(proxy)

            elif rec_type == 'rect' or geometry == 'rect':
                M = rec['M']
                N = rec['N']
                ax.plot(positions[:, 0], positions[:, 1], 'gs',
                        markersize=6, markerfacecolor='green',
                        markeredgecolor='darkgreen', zorder=5)

                proxy = Line2D([0], [0], marker='s', color='w',
                               markerfacecolor='green', markeredgecolor='darkgreen',
                               markersize=10,
                               label=f'Receiver Array: {name} ({M}×{N}={M_total})')
                handles.append(proxy)

        for name, c in self.contrasts.items():
            shape = c['shape']
            intensity = c['intensity']
            label = f'Contrast: {name} ({shape}, χ={intensity:.2f})'

            for patch in c['patches']:
                patch.set_label(label)
                ax.add_patch(patch)
                handles.append(patch)

        make_legend(ax, handles)
        plt.tight_layout()
        plt.show()
        return fig, ax

    def plot_contrast(self, domain_name=None):
        assert self.Chi is not None, "No contrast defined."

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_aspect('equal')

        scene_width = self.x_range[1] - self.x_range[0]
        scene_height = self.y_range[1] - self.y_range[0]

        if domain_name is not None and domain_name in self.domains:
            d = self.domains[domain_name]
            x0, y0 = d['rho']
            x1, y1 = x0 + d['width'], y0 + d['height']
            mask = d['mask']
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            title = f'Contrast $\\chi(\\rho)$ — {domain_name}'

            domain_chi = self.Chi[mask]
            vmin = domain_chi.min()
            vmax = domain_chi.max()
            if vmin == vmax:
                vmin -= 0.01
                vmax += 0.01

            domain_size = min(d['width'], d['height'])
            fractions = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            target = domain_size / 5
            major_tick = min([f * self.mu for f in fractions],
                             key=lambda s: abs(s - target))
        else:
            ax.set_xlim(self.x_range[0], self.x_range[1])
            ax.set_ylim(self.y_range[1], self.y_range[0])
            title = r'Contrast $\chi(\rho)$ — Full Scene'
            vmin = self.Chi.min()
            vmax = self.Chi.max()
            if vmin == vmax:
                vmin -= 0.01
                vmax += 0.01

            domain_size = min(scene_width, scene_height)
            fractions = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
            target = domain_size / 5
            major_tick = min([f * self.mu for f in fractions],
                             key=lambda s: abs(s - target))

        im = ax.pcolormesh(self.X, self.Y, self.Chi,
                           cmap='inferno', shading='auto', vmin=vmin, vmax=vmax)

        ax.xaxis.set_minor_locator(MultipleLocator(self.mu))
        ax.yaxis.set_minor_locator(MultipleLocator(self.mu))
        ax.xaxis.set_major_locator(MultipleLocator(major_tick))
        ax.yaxis.set_major_locator(MultipleLocator(major_tick))
        ax.grid(True, which='minor', color='white', linewidth=0.4, alpha=0.3)
        ax.grid(True, which='major', color='white', linewidth=0.6, alpha=0.6)

        axis_formatter(ax, self.mu)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r'$\chi$')
        plt.show()
        return fig, ax

    def plot_field(self, field_type='total', vmin=0, vmax=0):
        assert self.U is not None, "Run compute() first."

        if field_type == 'incident':
            assert self.U_inc is not None
            U_plot = self.U_inc
            title = 'Incident Field $u_{inc}$'
        elif field_type == 'scattered':
            assert self.U_sc_grid is not None
            U_plot = self.U_sc_grid
            title = 'Scattered Field $u_{sc}$'
        else:
            if self.U_sc_grid is not None:
                U_plot = self.U
                title = 'Total Field $u_{inc} + u_{sc}$'
            else:
                U_plot = self.U_inc
                title = 'Incident Field $u_{inc}$'

        field_plot(U_plot, self.X, self.Y, self.x_range, self.y_range, self.mu,
                   title=title, vmin=vmin, vmax=vmax)

    def plot_domain_field(self, name, field_type='total'):
        assert self.U is not None, "Run compute() first."
        d = self.domains[name]
        x0, y0 = d['rho']
        x1, y1 = x0 + d['width'], y0 + d['height']
        mask = d['mask']

        if field_type == 'incident':
            U_full = self.U_inc
            title_base = 'Incident Field'
        elif field_type == 'scattered':
            U_full = self.U_sc_grid
            title_base = 'Scattered Field'
        else:
            U_full = self.U
            title_base = 'Total Field'

        domain_size = min(d['width'], d['height'])
        fractions = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        target = domain_size / 5
        major_tick = min([f * self.mu for f in fractions],
                         key=lambda s: abs(s - target))

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        operations = [np.abs, np.real, np.imag]
        titles = ['Absolute Value', 'Real Part', 'Imaginary Part']

        # Mask out contrast support to avoid singularity-driven color scale
        chi_mask = (self.Chi > 0) if self.Chi is not None else np.zeros_like(mask)
        valid_mask = mask & ~chi_mask  # domain pixels excluding contrast support

        for a, op, sub_title in zip(ax, operations, titles):
            a.set_aspect('equal')
            field_values = op(U_full)

            # Use only non-singular pixels for color scale
            if valid_mask.any():
                domain_values = field_values[valid_mask]
            else:
                domain_values = field_values[mask]

            vmin, vmax = domain_values.min(), domain_values.max()
            if vmin == vmax:
                vmin -= 0.1 * abs(vmin) + 1e-10
                vmax += 0.1 * abs(vmax) + 1e-10

            im = a.pcolormesh(self.X, self.Y, field_values,
                              cmap='viridis', vmin=vmin, vmax=vmax)
            a.set_xlim(x0, x1)
            a.set_ylim(y1, y0)
            a.set_title(sub_title)
            a.xaxis.set_minor_locator(MultipleLocator(self.mu))
            a.yaxis.set_minor_locator(MultipleLocator(self.mu))
            a.xaxis.set_major_locator(MultipleLocator(major_tick))
            a.yaxis.set_major_locator(MultipleLocator(major_tick))
            a.grid(True, which='minor', color='white', linewidth=0.4, alpha=0.3)
            a.grid(True, which='major', color='white', linewidth=0.6, alpha=0.6)
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

        axis_formatter(ax, self.mu)
        fig.suptitle(f'{title_base} — {name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_reconstruction(self, domain_name, chi_recon, shape_result=None):
        d = self.domains[domain_name]
        x0, y0 = d['rho']
        x1, y1 = x0 + d['width'], y0 + d['height']
        mask = d['mask']

        chi_true_2d = self.Chi.copy()
        chi_recon_2d = np.zeros_like(self.X)
        chi_recon_2d[mask] = chi_recon

        chi_true_vec = self.Chi[mask]
        error = np.linalg.norm(chi_recon - chi_true_vec) / (np.linalg.norm(chi_true_vec) + 1e-10)

        ncols = 4 if shape_result is not None else 3
        fig, ax = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
        if ncols == 3:
            ax = list(ax)

        vmax_chi = max(chi_true_2d.max(), chi_recon_2d.max(), 0.01)

        im0 = ax[0].pcolormesh(self.X, self.Y, chi_true_2d, cmap='inferno', vmin=0, vmax=vmax_chi)
        ax[0].set_xlim(x0, x1)
        ax[0].set_ylim(y1, y0)
        ax[0].set_title('True Contrast')
        ax[0].set_aspect('equal')
        plt.colorbar(im0, ax=ax[0])

        im1 = ax[1].pcolormesh(self.X, self.Y, chi_recon_2d, cmap='inferno', vmin=0, vmax=vmax_chi)
        ax[1].set_xlim(x0, x1)
        ax[1].set_ylim(y1, y0)
        ax[1].set_title(f'Reconstructed (error={error:.3f})')
        ax[1].set_aspect('equal')
        plt.colorbar(im1, ax=ax[1])

        diff = chi_recon_2d - chi_true_2d
        vmax_d = max(abs(diff.min()), abs(diff.max()), 1e-10)
        im2 = ax[2].pcolormesh(self.X, self.Y, diff, cmap='RdBu_r', vmin=-vmax_d, vmax=vmax_d)
        ax[2].set_xlim(x0, x1)
        ax[2].set_ylim(y1, y0)
        ax[2].set_title('Difference')
        ax[2].set_aspect('equal')
        plt.colorbar(im2, ax=ax[2])

        if shape_result is not None:
            shape_str = shape_result['shape']
            if shape_str in ['ellipse', 'rectangle']:
                shape_str += f"\n(aspect={shape_result.get('aspect', 1):.2f}, angle={shape_result.get('angle', 0)}°)"

            im3 = ax[3].pcolormesh(self.X, self.Y, shape_result['chi_fitted'],
                                   cmap='inferno', vmin=0, vmax=vmax_chi)
            ax[3].set_xlim(x0, x1);
            ax[3].set_ylim(y1, y0)
            ax[3].set_title(f"Matched: {shape_str}\nconf={shape_result['confidence']:.2f}")
            ax[3].set_aspect('equal')
            plt.colorbar(im3, ax=ax[3])

        plt.tight_layout()
        plt.show()

    def _get_tick_spacing(self, domain_size, target_ticks=5):
        fractions = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        target = domain_size / target_ticks
        major_tick = min([f * self.mu for f in fractions],
                         key=lambda s: abs(s - target))
        return major_tick
