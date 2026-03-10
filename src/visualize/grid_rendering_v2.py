from collections import abc

import jax
import jax.numpy as jnp


def downsample(img: jnp.ndarray, factor: int):
    """Downsample an image along both dimensions by some factor."""
    assert img.shape[0] % factor == 0
    assert img.shape[1] % factor == 0

    img = img.reshape([img.shape[0] // factor, factor, img.shape[1] // factor, factor, 3])
    img = img.mean(axis=3)
    img = img.mean(axis=1)

    # convert back to uint8
    return img.astype(jnp.uint8)


def fill_coords(img: jnp.ndarray, fn: abc.Callable[[float, float], bool], color: jnp.ndarray):
    """Fill pixels of an image with coordinates matching a filter function."""

    def _mask_fn(y: int, x: int):
        yf = (y + 0.5) / img.shape[0]
        xf = (x + 0.5) / img.shape[1]
        return fn(xf, yf)

    ys, xs = jnp.indices(img.shape[:2])
    mask = jax.vmap(jax.vmap(_mask_fn))(ys, xs)

    color_img = jnp.full_like(img, color)
    return jnp.where(mask[:, :, None], color_img, img)


def rotate_fn(fin: abc.Callable[[float, float], bool], cx: float, cy: float, theta: float):
    def fout(x: float, y: float):
        x = x - cx
        y = y - cy

        x2 = cx + x * jnp.cos(-theta) - y * jnp.sin(-theta)
        y2 = cy + y * jnp.cos(-theta) + x * jnp.sin(-theta)

        return fin(x2, y2)

    return fout


def point_in_line(x0: float, y0: float, x1: float, y1: float, r: float):
    p0 = jnp.array([x0, y0])
    p1 = jnp.array([x1, y1])
    vec = p1 - p0
    dist = jnp.linalg.norm(vec)
    vec = vec / dist

    xmin = min(x0, x1) - r
    xmax = max(x0, x1) + r
    ymin = min(y0, y1) - r
    ymax = max(y0, y1) + r

    def fn(x: float, y: float):
        # Fast, early escape test
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return False

        q = jnp.array([x, y])
        pq = q - p0

        # Closest point on line
        a = jnp.dot(pq, vec)
        a = jnp.clip(a, 0, dist)
        p = p0 + a * vec

        dist_to_line = jnp.linalg.norm(q - p)
        return dist_to_line <= r

    return fn


def point_in_circle(cx: float, cy: float, r: float):
    def fn(x: float, y: float):
        return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r

    return fn


def point_in_rect(xmin: float, xmax: float, ymin: float, ymax: float):
    def fn(x: float, y: float):
        return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    return fn


def point_in_triangle(a: tuple, b: tuple, c: tuple):
    a = jnp.array(a)
    b = jnp.array(b)
    c = jnp.array(c)

    def fn(x: float, y: float):
        v0 = c - a
        v1 = b - a
        v2 = jnp.array((x, y)) - a

        # Compute dot products
        dot00 = jnp.dot(v0, v0)
        dot01 = jnp.dot(v0, v1)
        dot02 = jnp.dot(v0, v2)
        dot11 = jnp.dot(v1, v1)
        dot12 = jnp.dot(v1, v2)

        # Compute barycentric coordinates
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        # Check if point is in triangle
        return (u >= 0) & (v >= 0) & (u + v < 1)

    return fn


def highlight_img(img: jnp.ndarray, color: tuple[int, int, int] = (255, 255, 255), alpha: float = 0.30):
    """Add highlighting to an image."""
    blend_img = img + alpha * (jnp.array(color, dtype=jnp.uint8) - img)
    return jnp.clip(blend_img, 0, 255).astype(jnp.uint8)
