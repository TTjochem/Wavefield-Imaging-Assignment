import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import hankel1, hankel2
from hankel import HankelTransform
from matplotlib.ticker import FuncFormatter
from numpy import pi,pow,real,imag, abs
from numpy.linalg import norm, matrix_rank, matrix_power
from matplotlib.ticker import MultipleLocator
from Scene import *

# Some functions to define simple geometry rho_2D = [x, y], rho_3D = [x, y, z]
# ============================================================
# Shape functions — each returns (Chi, patch_list)
# ============================================================

def make_triangle(X, Y, rho_0, size, intensity):
    """Equilateral triangle contrast."""
    x0, y0 = rho_0
    h = size * np.sqrt(3) / 2

    v1 = np.array([x0, y0 + 2 * h / 3])
    v2 = np.array([x0 - size / 2, y0 - h / 3])
    v3 = np.array([x0 + size / 2, y0 - h / 3])

    def _sign(px, py, x1, y1, x2, y2):
        return (px - x2) * (y1 - y2) - (x1 - x2) * (py - y2)

    d1 = _sign(X, Y, v1[0], v1[1], v2[0], v2[1])
    d2 = _sign(X, Y, v2[0], v2[1], v3[0], v3[1])
    d3 = _sign(X, Y, v3[0], v3[1], v1[0], v1[1])

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    mask = ~(has_neg & has_pos)

    Chi = np.where(mask, intensity, 0.0)

    # Create patch for plotting
    triangle_verts = np.array([v1, v2, v3])
    patch = patches.Polygon(triangle_verts, closed=True,
                            linewidth=2, edgecolor='green',
                            facecolor='green', alpha=0.3)

    return Chi, [patch]

def make_star(X, Y, rho_0, size, intensity, points=5):
    """Star-shaped contrast."""
    x0, y0 = rho_0
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    inner_radius = size * 0.4
    outer_radius = size

    # Build star vertices for both mask and patch
    star_verts = []
    for i in range(2 * points):
        angle = i * np.pi / points - np.pi / 2
        radius = outer_radius if i % 2 == 0 else inner_radius
        star_verts.append([x0 + radius * np.cos(angle),
                           y0 + radius * np.sin(angle)])
    star_verts = np.array(star_verts)

    # Point-in-polygon test using matplotlib
    from matplotlib.path import Path
    star_path = Path(star_verts)
    points_flat = np.column_stack([X.ravel(), Y.ravel()])
    mask_flat = star_path.contains_points(points_flat)
    mask = mask_flat.reshape(X.shape)

    Chi = np.where(mask, intensity, 0.0)

    patch = patches.Polygon(star_verts, closed=True,
                            linewidth=2, edgecolor='gold',
                            facecolor='gold', alpha=0.3)

    return Chi, [patch]

def make_circle(X, Y, rho_0, size, intensity):
    """
    Create a circular contrast.

    Parameters
    ----------
    X, Y : 2D arrays
        Meshgrid coordinates.
    rho_0 : array-like (2,)
        Center position (x0, y0).
    size : float
        Radius of the circle.
    intensity : float
        Contrast value inside the circle.

    Returns
    -------
    Chi : 2D array
        Contrast function.
    patches_list : list
        Matplotlib patch for plotting.
    """
    x0, y0 = rho_0
    R = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    mask = (R <= size)
    Chi = np.where(mask, intensity, 0.0)

    # Create patch for plotting
    patch = patches.Circle((x0, y0), radius=size,
                           linewidth=2, edgecolor='green',
                           facecolor='green', alpha=0.3)

    return Chi, [patch]

def make_ellipse(X, Y, rho_0, size, intensity, aspect_ratio=0.6, angle=0):
    """
    Create an elliptical contrast.

    Parameters
    ----------
    X, Y : 2D arrays
    rho_0 : array-like (2,) — center (x0, y0)
    size : float — semi-major axis length
    intensity : float
    aspect_ratio : float — semi-minor / semi-major (default 0.6)
    angle : float — rotation angle in degrees
    """
    x0, y0 = rho_0
    theta = np.radians(angle)

    # Rotate coordinates
    X_rot = (X - x0) * np.cos(theta) + (Y - y0) * np.sin(theta)
    Y_rot = -(X - x0) * np.sin(theta) + (Y - y0) * np.cos(theta)

    semi_major = size
    semi_minor = size * aspect_ratio

    R_ellipse = (X_rot / semi_major) ** 2 + (Y_rot / semi_minor) ** 2
    mask = R_ellipse <= 1
    Chi = np.where(mask, intensity, 0.0)

    # Patch
    patch = patches.Ellipse((x0, y0), width=2 * semi_major, height=2 * semi_minor,
                            angle=angle, linewidth=2, edgecolor='green',
                            facecolor='green', alpha=0.3)

    return Chi, [patch]

def make_letter(X, Y, rho_0, size, intensity, letter='A'):
    """Letter-shaped contrast using bitmap approach."""
    # (Same implementation as before, returning Chi and an empty patch list
    #  since letters are complex shapes — or approximate with polygon)
    # ... [your letter implementation here]
    Chi = make_letter_mask(X, Y, rho_0, size, intensity, letter)  # your existing function

    # For letters, we can approximate with a bounding box patch
    x0, y0 = rho_0
    patch = patches.Rectangle((x0 - size / 2, y0 - size / 2), size, size,
                              linewidth=2, edgecolor='purple',
                              facecolor='purple', alpha=0.3,
                              label=f'Letter {letter}')

    return Chi, [patch]

def make_rectangle(X, Y, rho_0, size, intensity, width=None, height=None):
    """Rectangle contrast (default shape)."""
    x0, y0 = rho_0
    w = width if width is not None else size
    h = height if height is not None else size

    mask = ((X >= x0 - w / 2) & (X <= x0 + w / 2) &
            (Y >= y0 - h / 2) & (Y <= y0 + h / 2))

    Chi = np.where(mask, intensity, 0.0)

    patch = patches.Rectangle((x0 - w / 2, y0 - h / 2), w, h,
                              linewidth=2, edgecolor='red',
                              facecolor='red', alpha=0.3)

    return Chi, [patch]

def make_triangle_mask(X, Y, rho_0, size, intensity):
    """
    Creates an equilateral triangle contrast centered at rho_0.

    Parameters
    ----------
    X, Y : 2D arrays
        Meshgrid coordinates.
    rho_0 : array-like (2,)
        Center of the triangle (x0, y0).
    size : float
        Side length of the triangle.
    intensity : float
        Contrast value inside the shape.

    Returns
    -------
    Chi : 2D array
        Contrast function on the grid.
    """
    x0, y0 = rho_0
    h = size * np.sqrt(3) / 2  # height of equilateral triangle

    # Vertices of equilateral triangle (pointing up)
    v1 = np.array([x0, y0 + 2 * h / 3])  # top
    v2 = np.array([x0 - size / 2, y0 - h / 3])  # bottom left
    v3 = np.array([x0 + size / 2, y0 - h / 3])  # bottom right

    # Barycentric coordinate check
    def point_in_triangle(px, py):
        d1 = _sign(px, py, v1[0], v1[1], v2[0], v2[1])
        d2 = _sign(px, py, v2[0], v2[1], v3[0], v3[1])
        d3 = _sign(px, py, v3[0], v3[1], v1[0], v1[1])
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_neg & has_pos)

    mask = point_in_triangle(X, Y)
    Chi = np.where(mask, intensity, 0.0)
    return Chi

def make_star_mask(X, Y, rho_0, size, intensity, points=5):
    """
    Creates a star-shaped contrast centered at rho_0.

    Parameters
    ----------
    X, Y : 2D arrays
        Meshgrid coordinates.
    rho_0 : array-like (2,)
        Center of the star (x0, y0).
    size : float
        Outer radius of the star.
    intensity : float
        Contrast value inside the shape.
    points : int
        Number of star points (default 5).

    Returns
    -------
    Chi : 2D array
        Contrast function on the grid.
    """
    x0, y0 = rho_0
    dx = X - x0
    dy = Y - y0
    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)

    # Star shape in polar coordinates: outer radius modulated
    inner_radius = size * 0.4
    outer_radius = size

    # Star boundary radius as function of angle
    angle_per_point = 2 * np.pi / points
    star_radius = np.zeros_like(theta)

    for i in range(2 * points):
        angle_start = i * angle_per_point / 2 - np.pi / 2
        angle_end = (i + 1) * angle_per_point / 2 - np.pi / 2
        if i % 2 == 0:
            radius_val = outer_radius
        else:
            radius_val = inner_radius

        mask_angle = (theta >= angle_start) & (theta < angle_end)
        star_radius[mask_angle] = radius_val

    # Handle wrap-around
    star_radius[theta < -np.pi + angle_per_point / 2] = outer_radius

    mask = r <= star_radius
    Chi = np.where(mask, intensity, 0.0)
    return Chi

def make_letter_mask(X, Y, rho_0, size, intensity, letter='A'):
    """
    Creates a letter-shaped contrast (simple bitmap letters).
    Supports: A, B, C, E, F, H, I, L, O, T (uppercase).

    Parameters
    ----------
    X, Y : 2D arrays
        Meshgrid coordinates.
    rho_0 : array-like (2,)
        Center of the letter (x0, y0).
    size : float
        Approximate height/width of the letter.
    intensity : float
        Contrast value inside the shape.
    letter : str
        Letter character (default 'A').

    Returns
    -------
    Chi : 2D array
        Contrast function on the grid.
    """
    x0, y0 = rho_0
    # Normalize coordinates to a unit grid centered at (0,0)
    x_norm = (X - x0) / size * 2  # maps to roughly [-1, 1]
    y_norm = (Y - y0) / size * 2

    # Define letters as boolean masks on normalized coordinates
    letter = letter.upper()

    if letter == 'A':
        # Two diagonal lines and a horizontal crossbar
        mask = (
                ((y_norm <= 1.0) & (y_norm >= -1.0)) &
                ((np.abs(x_norm - y_norm * 0.6) <= 0.15) |
                 (np.abs(x_norm + y_norm * 0.6) <= 0.15) |
                 ((np.abs(y_norm) <= 0.15) & (np.abs(x_norm) <= 0.6)))
        )
    elif letter == 'B':
        mask = (
                (np.abs(x_norm + 0.5) <= 0.15) |  # vertical bar
                ((np.abs(y_norm - 0.5) <= 0.15) & (x_norm >= -0.5) & (x_norm <= 0.3)) |
                ((np.abs(y_norm) <= 0.15) & (x_norm >= -0.5) & (x_norm <= 0.3)) |
                ((np.abs(y_norm + 0.5) <= 0.15) & (x_norm >= -0.5) & (x_norm <= 0.3)) |
                ((x_norm - 0.3) ** 2 + (y_norm - 0.75) ** 2 <= 0.25 ** 2) |
                ((x_norm - 0.3) ** 2 + (y_norm + 0.75) ** 2 <= 0.25 ** 2)
        )
    elif letter == 'C':
        mask = (
                ((x_norm + 0.5) ** 2 + y_norm ** 2 >= 0.7 ** 2) &
                ((x_norm + 0.5) ** 2 + y_norm ** 2 <= 1.0 ** 2) &
                (x_norm <= 0.5) & (np.abs(y_norm) <= 0.7)
        )
    elif letter == 'E':
        mask = (
                (np.abs(x_norm + 0.5) <= 0.12) |  # vertical bar
                ((np.abs(y_norm - 1.0) <= 0.1) & (x_norm >= -0.5) & (x_norm <= 0.3)) |
                ((np.abs(y_norm) <= 0.1) & (x_norm >= -0.5) & (x_norm <= 0.2)) |
                ((np.abs(y_norm + 1.0) <= 0.1) & (x_norm >= -0.5) & (x_norm <= 0.3))
        )
    elif letter == 'F':
        mask = (
                (np.abs(x_norm + 0.5) <= 0.12) |  # vertical bar
                ((np.abs(y_norm - 1.0) <= 0.1) & (x_norm >= -0.5) & (x_norm <= 0.3)) |
                ((np.abs(y_norm) <= 0.1) & (x_norm >= -0.5) & (x_norm <= 0.2))
        )
    elif letter == 'H':
        mask = (
                (np.abs(x_norm + 0.5) <= 0.12) |
                (np.abs(x_norm - 0.5) <= 0.12) |
                ((np.abs(y_norm) <= 0.12) & (np.abs(x_norm) <= 0.5))
        )
    elif letter == 'I':
        mask = (
                (np.abs(x_norm) <= 0.12) |
                ((np.abs(y_norm - 1.0) <= 0.1) & (np.abs(x_norm) <= 0.3)) |
                ((np.abs(y_norm + 1.0) <= 0.1) & (np.abs(x_norm) <= 0.3))
        )
    elif letter == 'L':
        mask = (
                (np.abs(x_norm + 0.5) <= 0.12) |
                ((np.abs(y_norm + 1.0) <= 0.1) & (x_norm >= -0.5) & (x_norm <= 0.3))
        )
    elif letter == 'O':
        r_norm = np.sqrt(x_norm ** 2 + y_norm ** 2)
        mask = (r_norm >= 0.6) & (r_norm <= 1.0)
    elif letter == 'T':
        mask = (
                (np.abs(x_norm) <= 0.12) |
                ((np.abs(y_norm - 1.0) <= 0.1) & (np.abs(x_norm) <= 0.5))
        )
    else:
        # Default: rectangle
        mask = (np.abs(x_norm) <= 0.5) & (np.abs(y_norm) <= 0.5)

    Chi = np.where(mask, intensity, 0.0)
    return Chi

def _sign(x1, y1, x2, y2, x3, y3):
    """Helper: sign of cross product for point-in-triangle test."""
    return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)


def circle(ax, rho, radius=0.25, label='source'):
    """Draw a source as a red dot with label offset outside."""
    ax.set_aspect('equal')

    # Red filled dot
    patch = patches.Circle((rho[0], rho[1]), radius=radius,
                           linewidth=1.5, edgecolor='darkred',
                           facecolor='red', alpha=0.8, zorder=5)
    ax.add_patch(patch)

    # Label offset in points (screen units, not data coords)
    ax.annotate(label, xy=(rho[0], rho[1]),
                xytext=(10, -10), textcoords='offset points',
                fontsize=9, fontweight='bold', color='darkred',
                ha='left', va='top',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='darkred', alpha=0.8),
                zorder=6)

    return patch


def square(ax, rho, width, height, label='domain', label_offset=0.4):
    """Draw a rectangle patch with label above it."""
    patch = patches.Rectangle((rho[0], rho[1]), width=width, height=height,
                              linewidth=2, edgecolor='blue', facecolor='none')
    ax.add_patch(patch)

    # Label centered ABOVE the top edge
    ax.annotate(label, xy=(rho[0] + width / 2, rho[1]),
                xytext=(0, -15), textcoords='offset points',
                fontsize=10, fontweight='bold', color='blue',
                ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='blue', alpha=0.85),
                zorder=6)

    return patch


def make_grid_2D(x_range, y_range, stepsize=1, mu=None):
    """
    Create a formatted 2D grid for scene plotting.

    Parameters
    ----------
    x_range : list [xmin, xmax]
        x-axis limits.
    y_range : list [ymin, ymax]
        y-axis limits.
    stepsize : float
        Grid step size (for reference, not enforced here).
    mu : float or None
        Wavelength. If provided, axis ticks are labeled in λ units.

    Returns
    -------
    fig, ax : matplotlib figure and axis
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # inverted y-axis (down is positive)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if mu is not None:
        formatter = FuncFormatter(lambda val, pos: f'{val / mu:.1f}λ')
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)

    return fig, ax


def make_legend(ax, handles):
    """Add legend with deduplicated labels."""
    seen = {}
    unique_handles = []
    for h in handles:
        label = h.get_label()
        if label not in seen:
            seen[label] = True
            unique_handles.append(h)

    if unique_handles:
        # Place legend outside to the right
        ax.legend(handles=unique_handles,
                  loc='center left',
                  bbox_to_anchor=(1.02, 0.5),
                  fontsize=8,
                  framealpha=0.9,
                  borderaxespad=0)

def axis_formatter(axes, mu):
    formatter = FuncFormatter(lambda val, pos: f'{val/mu:g}λ')
    for a in np.atleast_1d(axes).ravel():
        a.xaxis.set_major_formatter(formatter)
        a.yaxis.set_major_formatter(formatter)


def field_plot(U, X, Y, x_range, y_range, mu, title='Field', vmin=0, vmax=0.4):
    """
    Plot a complex field (3 panels: abs, real, imag).

    Parameters
    ----------
    U : 2D complex array
        Field to plot.
    X, Y : 2D arrays
        Meshgrid coordinates.
    x_range : list [xmin, xmax]
        x-axis limits.
    y_range : list [ymin, ymax]
        y-axis limits.
    mu : float
        Wavelength for tick spacing.
    title : str
        Suptitle for the figure.
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    # Dynamic major tick
    domain_size = min(X.max() - X.min(), Y.max() - Y.min())
    fractions = [0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    target = domain_size / 5
    major_tick = min([f * mu for f in fractions], key=lambda s: abs(s - target))

    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    operations = [np.abs, np.real, np.imag]
    titles = ['Absolute Value', 'Real Part', 'Imaginary Part']

    for a, operation, sub_title in zip(ax, operations, titles):
        a.set_aspect('equal')
        field_vals = operation(U)

        # Manual range for testing
        if vmin == 0 and vmax == 0:
            # Auto-scale (original behavior)
            im = a.pcolormesh(X, Y, field_vals, cmap='viridis')
        else:
            # Fixed range for testing
            im = a.pcolormesh(X, Y, field_vals, cmap='viridis', vmin=vmin, vmax=vmax)

        a.set_xlim(xmin, xmax)
        a.set_ylim(ymax, ymin)
        a.xaxis.set_minor_locator(MultipleLocator(mu))
        a.yaxis.set_minor_locator(MultipleLocator(mu))
        a.xaxis.set_major_locator(MultipleLocator(major_tick))
        a.yaxis.set_major_locator(MultipleLocator(major_tick))
        a.grid(True, which='minor', color='white', linewidth=0.4, alpha=0.3)
        a.grid(True, which='major', color='white', linewidth=0.6, alpha=0.6)
        a.set_title(sub_title)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)

    axis_formatter(ax, mu)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def make_mesh(stepsize, xlim, ylim):
    x = np.arange(0, xlim, stepsize)
    y = np.arange(0, ylim, stepsize)
    X, Y = np.meshgrid(x,y) # gridsize [x, y] ~ {.shape[0], .shape[1]}
    return X, Y

def get_grid_params(mu):
    rho_s = np.array([.5*mu, 10*mu])  # 2D source position vector/coordinate
    stepsize = mu/20
    xlim = 20*mu; ylim = 20*mu
    return rho_s, stepsize, xlim, ylim

def evaluate_SVD(A_list):
    for A in A_list:
        M = np.size(A,0)
        U, S, V = np.linalg.svd(A)
        plt.plot(S, label=f'M = {M}')

    plt.xlabel('Index')
    plt.ylabel('Log(σ)')
    plt.show()


def match_shape(chi_recon, X, Y, mask, intensity_threshold=0.3):
    """
    Match reconstructed contrast to known shape templates.
    Supports: circle, ellipse, square, rectangle, triangle, star, letters A-L-O-T.
    """

    # Create 2D reconstruction
    chi_recon_2d = np.zeros_like(X, dtype=float)
    chi_recon_2d[mask] = chi_recon

    # Binarize
    chi_max = chi_recon_2d.max()
    if chi_max <= 0:
        return {'shape': 'none', 'confidence': 0, 'error': 1.0}

    chi_binary = (chi_recon_2d >= intensity_threshold * chi_max).astype(float)

    # Centroid
    weights = chi_recon_2d.copy()
    total = weights.sum()
    if total > 0:
        x0_init = np.sum(X * weights) / total
        y0_init = np.sum(Y * weights) / total
    else:
        x0_init = (X[mask].min() + X[mask].max()) / 2
        y0_init = (Y[mask].min() + Y[mask].max()) / 2

    # Estimate size from area
    dx = X[0, 1] - X[0, 0]
    area = chi_binary.sum() * dx ** 2
    size_init = np.sqrt(area / np.pi)

    def make_template(shape_name, x0, y0, size, aspect=0.6, angle=0):
        """Generate a template using existing shape functions."""
        if shape_name == 'circle':
            return make_circle(X, Y, (x0, y0), size, 1.0)[0]
        elif shape_name == 'ellipse':
            return make_ellipse(X, Y, (x0, y0), size, 1.0, aspect_ratio=aspect, angle=angle)[0]
        elif shape_name == 'triangle':
            return make_triangle(X, Y, (x0, y0), size, 1.0)[0]
        elif shape_name == 'star':
            return make_star(X, Y, (x0, y0), size, 1.0)[0]
        elif shape_name == 'square':
            return make_rectangle(X, Y, (x0, y0), size, 1.0, width=size, height=size)[0]
        elif shape_name == 'rectangle':
            return make_rectangle(X, Y, (x0, y0), size, 1.0, width=size, height=size * 0.6)[0]
        elif shape_name == 'A':
            return make_letter(X, Y, (x0, y0), size, 1.0, letter='A')[0]
        elif shape_name == 'O':
            return make_letter(X, Y, (x0, y0), size, 1.0, letter='O')[0]
        elif shape_name == 'T':
            return make_letter(X, Y, (x0, y0), size, 1.0, letter='T')[0]
        elif shape_name == 'L':
            return make_letter(X, Y, (x0, y0), size, 1.0, letter='L')[0]
        else:
            return np.zeros_like(X)

    # Search
    shapes = ['circle', 'ellipse', 'square', 'rectangle', 'triangle', 'star',
              'A', 'O', 'T', 'L']
    aspects = [0.3, 0.5, 0.6, 0.7, 0.8]
    angles = [0, 30, 45, 60, 90, 120, 135, 150]

    best_score = -1
    best_params = None

    x_shifts = np.arange(-3, 4) * dx + x0_init
    y_shifts = np.arange(-3, 4) * dx + y0_init
    sizes = np.linspace(0.3, 2.5, 8) * size_init

    for shape_name in shapes:
        for x0 in x_shifts:
            for y0 in y_shifts:
                for size in sizes:
                    if size <= 0:
                        continue

                    # For shapes that benefit from aspect ratio
                    if shape_name in ['ellipse', 'rectangle']:
                        for aspect in aspects:
                            for angle in angles:
                                template = make_template(shape_name, x0, y0, size, aspect, angle)
                                score = _compute_score(template, chi_binary, mask)
                                if score > best_score:
                                    best_score = score
                                    best_params = {
                                        'shape': shape_name,
                                        'center': (x0, y0),
                                        'size': size,
                                        'aspect': aspect,
                                        'angle': angle,
                                        'template': template,
                                    }
                    elif shape_name in ['A', 'O', 'T', 'L']:
                        # Letters: just try different sizes
                        template = make_template(shape_name, x0, y0, size)
                        score = _compute_score(template, chi_binary, mask)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'shape': shape_name,
                                'center': (x0, y0),
                                'size': size,
                                'template': template,
                            }
                    else:
                        template = make_template(shape_name, x0, y0, size)
                        score = _compute_score(template, chi_binary, mask)
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'shape': shape_name,
                                'center': (x0, y0),
                                'size': size,
                                'template': template,
                            }

    if best_params is None:
        return {'shape': 'unknown', 'confidence': 0, 'error': 1.0}

    # Error
    recon_flat = chi_recon_2d[mask]
    fitted_flat = best_params['template'][mask] * chi_max
    error = np.linalg.norm(recon_flat - fitted_flat) / (np.linalg.norm(recon_flat) + 1e-10)

    return {
        'shape': best_params['shape'],
        'confidence': best_score,
        'center': best_params['center'],
        'size': best_params['size'],
        'angle': best_params.get('angle', 0),
        'aspect': best_params.get('aspect', 1.0),
        'chi_fitted': best_params['template'] * chi_max,
        'error': error,
    }


def _compute_score(template, chi_binary, mask):
    """Compute normalized cross-correlation within mask."""
    t_flat = template[mask]
    c_flat = chi_binary[mask]

    t_norm = t_flat - t_flat.mean()
    c_norm = c_flat - c_flat.mean()

    denom = np.linalg.norm(t_norm) * np.linalg.norm(c_norm)
    if denom > 1e-10:
        return np.dot(t_norm, c_norm) / denom
    return 0

def add_arc_line_sources(scene, base_name, rho_center, radius, num_sources=5,
                         arc_angle_deg=180, intensity=1.0, kb=None, source_type='point'):
    """
    Place line sources along an arc with 50% overlap.

    Each line source is a chord of the arc, positioned tangentially.
    """
    angles = np.linspace(0, np.radians(arc_angle_deg), num_sources + 1)

    for i in range(num_sources):
        # Center angle of this source
        angle_mid = (angles[i] + angles[i + 1]) / 2

        # Start and end angles (50% overlap = extend by 50% on each side)
        angle_width = (angles[i + 1] - angles[i]) * 1.5  # 50% overlap means 1.5x width

        angle_start = angle_mid - angle_width / 2
        angle_end = angle_mid + angle_width / 2

        # Position along arc (at the given radius)
        x_start = rho_center[0] + radius * np.cos(angle_start)
        y_start = rho_center[1] + radius * np.sin(angle_start)
        x_end = rho_center[0] + radius * np.cos(angle_end)
        y_end = rho_center[1] + radius * np.sin(angle_end)

        scene.add_source(f'{base_name}_{i}',
                         source_type=source_type,
                         rho=np.array([x_start, y_start]),
                         rho_end=np.array([x_end, y_end]),
                         intensity=intensity, kb=kb)

# Approximate plane waves by placing sources far away (15-20μ)
# Far-field → wavefronts are nearly planar at the object
def add_plane_wave_source(scene, name, mu, angle_deg, distance=15, intensity=1.0,
                          use_line=False, line_length=10):
    """
    Approximate a plane wave with sources far away.

    Parameters
    ----------
    scene : Scene
    name : str
    mu : float — wavelength
    angle_deg : float — plane wave direction in degrees (0° = +x, 90° = +y)
    distance : float — how far away in wavelengths
    intensity : float
    use_line : bool — if True, uses a line source (more accurate plane wave)
    line_length : float — length of line source in wavelengths (only if use_line=True)
    """
    angle = np.radians(angle_deg)

    if use_line:
        # Line source perpendicular to propagation direction
        # This creates a more accurate plane wave
        perp_angle = angle + np.pi / 2  # perpendicular to wave direction

        half_length = line_length * mu / 2

        # Line endpoints (perpendicular to propagation direction)
        center_x = -distance * mu * np.cos(angle) + 0.5 * mu
        center_y = -distance * mu * np.sin(angle) + 0.5 * mu

        rho_start = np.array([
            center_x - half_length * np.cos(perp_angle),
            center_y - half_length * np.sin(perp_angle)
        ])
        rho_end = np.array([
            center_x + half_length * np.cos(perp_angle),
            center_y + half_length * np.sin(perp_angle)
        ])

        scene.add_source(name, source_type='line',
                         rho=rho_start, rho_end=rho_end,
                         intensity=intensity, kb=scene.kb)
    else:
        # Point source far away (approximate plane wave)
        rho = np.array([
            -distance * mu * np.cos(angle) + 0.5 * mu,
            -distance * mu * np.sin(angle) + 0.5 * mu
        ])
        scene.add_source(name, rho=rho, intensity=intensity, kb=scene.kb)

# ESPRIT Algorithm
def esprit(X, d, element_spacing, wavelength):
    """
    ESPRIT with configurable element spacing.

    Parameters
    ----------
    X : M×N data matrix
    d : number of sources
    element_spacing : float — distance between adjacent ULA elements
    wavelength : float — operating wavelength (mu)

    Returns
    -------
    theta : angles in degrees from broadside
    """
    U, _, _ = np.linalg.svd(X)
    Uz = U[:, :d]
    Ux = Uz[:-1, :]
    Uy = Uz[1:, :]
    Uxy = np.linalg.pinv(Ux) @ Uy
    phi = np.linalg.eigvals(Uxy)

    # Correct for actual element spacing
    sin_theta = np.angle(phi) * wavelength / (2 * np.pi * element_spacing)

    # Clip to valid range
    sin_theta = np.clip(sin_theta, -1, 1)
    theta = np.degrees(np.arcsin(sin_theta))

    return theta

def mvdr_beamformer(u_sc, positions, mu, steer_angle_deg):
    """Compute MVDR weights for steering toward steer_angle_deg."""
    M = len(u_sc)
    angle = np.radians(steer_angle_deg)
    k = 2 * np.pi / mu

    steering = np.exp(-1j * k * (positions[:, 0] * np.cos(angle) +
                                 positions[:, 1] * np.sin(angle)))

    R = np.outer(u_sc, u_sc.conj())
    R_reg = R + 0.01 * np.eye(M) * np.trace(R) / M

    R_inv = np.linalg.inv(R_reg)
    w = R_inv @ steering / (steering.conj() @ R_inv @ steering)

    return w


def estimate_model_order(X, threshold=0.1):
    """
    Estimate number of sources from singular values.
    Plot them and look for the "elbow".
    """
    _, S, _ = np.linalg.svd(X)

    # Normalize
    S_norm = S / S[0]

    # Find where singular values drop below threshold
    d = np.sum(S_norm > threshold)

    # Plot for visual inspection
    plt.semilogy(S_norm, 'b.-')
    plt.axhline(threshold, color='r', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Normalized singular value')
    plt.title(f'Estimated model order: d={d}')
    plt.show()

    return d

def estimate_order_aic(X, max_d=10):
    """Akaike Information Criterion for model order selection."""
    M, N = X.shape
    _, S, _ = np.linalg.svd(X)

    aic = np.zeros(max_d)
    for d in range(1, max_d + 1):
        noise_power = np.mean(S[d:] ** 2) if d < len(S) else 1e-10
        aic[d - 1] = -2 * N * np.sum(np.log(S[:d])) + 2 * d * (2 * M - d)

    d_est = np.argmin(aic) + 1
    return d_est