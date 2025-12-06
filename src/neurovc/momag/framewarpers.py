"""Utilities for spatial warping and splatting in the motion magnifier."""

__author__ = "Cosmas Heiss, Philipp Flotho"

try:
    import moderngl

    HAS_MODERNGL = True
except ModuleNotFoundError:
    moderngl = None
    HAS_MODERNGL = False

try:
    import torch

    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    HAS_TORCH = False

import numpy as np
from scipy.interpolate import griddata
import cv2
from typing import Tuple


def warp_image_pc(img, flow):
    """Warp an RGB image using SciPy point-cloud interpolation.

    Parameters
    ----------
    img : ndarray
        Source image with shape ``(H, W, 3)``.
    flow : ndarray
        Forward flow with shape ``(H, W, 2)`` in pixel coordinates.

    Returns
    -------
    ndarray
        Warped RGB image with the same shape and dtype as ``img``.
    """
    m, n = img.shape[:2]
    xi, yi = np.meshgrid(np.arange(n).astype(float), np.arange(m).astype(float))

    tmp = np.empty(img.shape, img.dtype)
    for i in range(3):
        griddata_result = griddata(
            (
                xi.flatten() + flow[:, :, 0].flatten(),
                yi.flatten() + flow[:, :, 1].flatten(),
            ),
            img[:, :, i].flatten(),
            (xi.flatten(), yi.flatten()),
            method="linear",
            fill_value=0,
        )
        tmp[:, :, i] = np.reshape(griddata_result, (m, n))

    return tmp


def warp_image_pc_single(img, flow):
    """Warp a single-channel image using point-cloud interpolation.

    Parameters
    ----------
    img : ndarray
        Source image with shape ``(H, W)``.
    flow : ndarray
        Forward flow with shape ``(H, W, 2)`` in pixel coordinates.

    Returns
    -------
    ndarray
        Warped image with the same shape and dtype as ``img``.
    """
    m, n = img.shape[:2]
    xi, yi = np.meshgrid(np.arange(n).astype(float), np.arange(m).astype(float))

    tmp = np.empty(img.shape, img.dtype)

    griddata_result = griddata(
        (
            xi.flatten() + flow[:, :, 0].flatten(),
            yi.flatten() + flow[:, :, 1].flatten(),
        ),
        img[:, :].flatten(),
        (xi.flatten(), yi.flatten()),
        method="nearest",
        fill_value=0,
    )
    tmp[:, :] = np.reshape(griddata_result, (m, n))

    return tmp


def warp_image_backwards(img, flow):
    """Backward warp an image using OpenCV's remap interface.

    Parameters
    ----------
    img : ndarray
        Source image with shape ``(H, W, C)``.
    flow : ndarray
        Backward flow with shape ``(H, W, 2)`` mapping output pixels to source
        locations.

    Returns
    -------
    ndarray
        Backward-warped image with the same shape and dtype as ``img``.
    """
    h, w = flow.shape[:2]
    # flow = -flow
    tmp_flow = np.empty(flow.shape, np.array(flow).dtype)
    np.copyto(tmp_flow, flow)
    tmp_flow[:, :, 0] += np.arange(w)
    tmp_flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, tmp_flow, None, cv2.INTER_LINEAR)
    return res


# moderngl-based fast forward warping, author: Cosmas Heiß
class OnlineFrameWarper:
    """Moderngl-backed forward warper for dense meshes."""

    def __init__(self, image_size):
        """Initialise the warper and allocate moderngl buffers.

        Parameters
        ----------
        image_size : tuple of int
            Output image size as ``(height, width)``.

        Raises
        ------
        ImportError
            If ``moderngl`` is not installed.
        """
        if not HAS_MODERNGL:
            raise ImportError(
                "moderngl is required for OnlineFrameWarper. Install the gui extra: 'pip install neurovc[gui]'"
            )
        self.image_size = image_size
        self.strip_indices = self.generate_triangle_strip_index_array()
        self.ctx = moderngl.create_context(standalone=True, require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_vert;
                in vec3 in_color;
                in float in_depth;
                out vec3 v_color;


                void main() {
                    v_color = in_color;
                    gl_Position = vec4(in_vert, in_depth, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 v_color;

                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            """,
        )
        dummy_vertices = self.get_dummy_vertices()
        self.vertex_buffer = self.ctx.buffer(dummy_vertices)
        self.vertex_array = self.ctx.vertex_array(
            self.prog, self.vertex_buffer, "in_vert", "in_color", "in_depth"
        )
        self.frame_buffer = self.ctx.simple_framebuffer(image_size[::-1])

    def get_dummy_vertices(self):
        """Return a dummy vertex buffer covering the full image plane.

        Returns
        -------
        bytes
            Vertex buffer encoded as 32-bit floats arranged in
            ``(x, y, r, g, b, depth)`` order.
        """
        image = np.zeros((*self.image_size, 3))
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, self.image_size[1], endpoint=True),
            np.linspace(-1, 1, self.image_size[0], endpoint=True),
        )
        displacements = np.stack((xx, yy), axis=2)
        depth = np.zeros(self.image_size)
        return self.get_into_vertex_buffer_shape(image, displacements, depth)

    def vertices_astype(self, vertices):
        """Convert vertex arrays to 32-bit floating point byte buffers.

        Parameters
        ----------
        vertices : ndarray
            Vertex array to serialise.

        Returns
        -------
        bytes
            Serialised vertex buffer.
        """
        return vertices.astype("f4").tobytes()

    def get_into_vertex_buffer_shape(self, image, displacements, depth):
        """Pack image, displacement, and depth channels into vertex layout.

        Parameters
        ----------
        image : ndarray
            RGB image normalised to ``[0, 1]`` with shape ``(H, W, 3)``.
        displacements : ndarray
            Flow field with shape ``(H, W, 2)``.
        depth : ndarray
            Depth buffer with shape ``(H, W)``.

        Returns
        -------
        bytes
            Serialised vertex buffer for the triangle strip.
        """
        vertices = np.concatenate((displacements, image, depth[:, :, None]), axis=2)[
            self.strip_indices[:, 0], self.strip_indices[:, 1]
        ]
        return self.vertices_astype(vertices)
        # return self.get_into_vb_fast(image, displacements, depth)

    # def get_into_vb_fast(self, image, displacements, depth):
    #    image = cp.asarray(image)
    #    displacements = cp.asarray(displacements)
    #    depth = cp.asarray(depth)
    #    strip_indices = cp.asarray(self.strip_indices)
    #    vertices = cp.concatenate((displacements, image, depth[:, :, None]), axis=2)[strip_indices[:, 0], strip_indices[:, 1]]
    #    return self.vertices_astype(cp.asnumpy(vertices))

    def generate_triangle_strip_index_array(self):
        """Generate a scan-line triangle strip index buffer.

        Returns
        -------
        ndarray
            Array of indices with shape ``(N, 2)`` describing the strip order.
        """
        out_indices_x = []
        out_indices_y = []
        for i in range(self.image_size[1] - 1):
            is_reversed = int((i % 2) * (-2) + 1)
            out_indices_x.append(np.arange(self.image_size[0]).repeat(2)[::is_reversed])
            out_indices_y.append(np.tile(np.array([i, i + 1]), self.image_size[0]))

        out_indices_x = np.concatenate(out_indices_x).astype(int)
        out_indices_y = np.concatenate(out_indices_y).astype(int)
        return np.stack((out_indices_x, out_indices_y), axis=1)

    def read_frame_buffer(self):
        """Return the framebuffer contents as ``uint8`` pixels.

        Returns
        -------
        ndarray
            Array containing the framebuffer data with shape ``(H, W, 4)``.
        """
        return np.frombuffer(self.frame_buffer.read(), "uint8").reshape(
            self.image_size[0], self.image_size[1], -1
        )

    def pixel_to_screenspace_coords(self, displacements):
        """Map pixel displacements to OpenGL clip-space coordinates.

        Parameters
        ----------
        displacements : ndarray
            Flow field with shape ``(H, W, 2)`` in pixel units.

        Returns
        -------
        ndarray
            Displacements mapped to clip space with the same shape as input.
        """
        out = np.zeros_like(displacements)
        out[:, :, 1] = displacements[:, :, 0] * (2.0 / (self.image_size[0] - 1)) - 1.0
        out[:, :, 0] = displacements[:, :, 1] * (2.0 / (self.image_size[1] - 1)) - 1.0
        return out

    def warp_image(self, image, displacements, depth):
        """Render an image under forward displacements and depth ordering.

        Parameters
        ----------
        image : ndarray
            Normalised RGB image in ``[0, 1]`` with shape ``(H, W, 3)``.
        displacements : ndarray
            Flow field with shape ``(H, W, 2)`` in pixel units.
        depth : ndarray
            Depth buffer with shape ``(H, W)``.

        Returns
        -------
        ndarray
            Rendered RGB image with shape ``(H, W, 3)`` and dtype ``float64``.
        """
        assert image.dtype == displacements.dtype == depth.dtype == float
        assert np.all(np.logical_and(image <= 1.0, image >= 0.0))
        assert (
            image.shape[:2] == displacements.shape[:2] == depth.shape == self.image_size
        )

        displacements = self.pixel_to_screenspace_coords(displacements)
        if depth.max() != depth.min():
            depth = -0.99 * (depth - depth.min()) / (depth.max() - depth.min())

        self.frame_buffer.use()
        self.frame_buffer.clear(0.0, 0.0, 0.0, 0.0)

        self.vertex_buffer.write(
            self.get_into_vertex_buffer_shape(image, displacements, depth)
        )

        self.vertex_array.render(moderngl.TRIANGLE_STRIP)

        return self.read_frame_buffer()[:, :, :3]

    def warp_image_uv(self, image, uv, depth=None):
        """Warp an image using UV flow (``u`` = x, ``v`` = y) and optional depth.

        Parameters
        ----------
        image : ndarray
            Source RGB image with shape ``(H, W, 3)``.
        uv : ndarray
            Flow field with shape ``(H, W, 2)`` where ``uv[..., 0]`` and
            ``uv[..., 1]`` represent ``u`` and ``v`` displacements.
        depth : ndarray, optional
            Depth buffer with shape ``(H, W)``. If omitted, zeros are used.

        Returns
        -------
        ndarray
            Rendered RGB image with shape ``(H, W, 3)``.
        """
        m, n = uv.shape[0:2]

        x, y = np.meshgrid(np.arange(n).astype(float), np.arange(m).astype(float))

        tmp_flow = np.empty(uv.shape, float)
        tmp_flow[:, :, 0] = y + uv[:, :, 1]
        tmp_flow[:, :, 1] = x + uv[:, :, 0]

        if depth is None:
            depth = np.zeros((m, n), float)

        return self.warp_image(np.array(image).astype(float) / 255.0, tmp_flow, depth)


def _to_bchw(image: "torch.Tensor") -> Tuple["torch.Tensor", str]:
    """Convert an image tensor to ``(B, C, H, W)`` layout.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor with 2–4 dimensions arranged as either
        ``(H, W)``, ``(C, H, W)``, ``(H, W, C)``, or their batched variants.

    Returns
    -------
    torch.Tensor
        Tensor reshaped to ``(B, C, H, W)``.
    str
        Identifier describing the original layout so it can be restored later.

    Raises
    ------
    ValueError
        If the layout cannot be inferred unambiguously.
    """
    tensor = image
    if tensor.ndim == 4:
        if tensor.shape[1] <= 4:
            layout = "bchw"
        elif tensor.shape[-1] <= 4:
            tensor = tensor.permute(0, 3, 1, 2)
            layout = "bhwc"
        else:
            raise ValueError("Expected channels-first or channels-last 4D tensor.")
    elif tensor.ndim == 3:
        if tensor.shape[0] <= 4 and tensor.shape[0] != tensor.shape[2]:
            tensor = tensor.unsqueeze(0)
            layout = "chw"
        elif tensor.shape[-1] <= 4:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            layout = "hwc"
        else:
            raise ValueError(
                "Ambiguous 3D tensor layout; cannot infer channel dimension."
            )
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        layout = "hw"
    else:
        raise ValueError("Expected tensor with 2 to 4 dimensions.")
    return tensor, layout


def _from_bchw(tensor: "torch.Tensor", layout: str):
    """Restore a tensor to its original layout after :func:`_to_bchw`.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor in ``(B, C, H, W)`` layout.
    layout : str
        Layout identifier returned by :func:`_to_bchw`.

    Returns
    -------
    torch.Tensor
        Tensor reshaped back to its original layout.

    Raises
    ------
    ValueError
        If ``layout`` is unknown.
    """
    if layout == "bchw":
        return tensor
    if layout == "bhwc":
        return tensor.permute(0, 2, 3, 1)
    if layout == "chw":
        return tensor.squeeze(0)
    if layout == "hwc":
        return tensor.squeeze(0).permute(1, 2, 0)
    if layout == "hw":
        return tensor.squeeze(0).squeeze(0)
    raise ValueError(f"Unknown layout identifier '{layout}'.")


def _to_b2hw(flow: "torch.Tensor") -> Tuple["torch.Tensor", str]:
    """Convert a flow tensor to ``(B, 2, H, W)`` layout.

    Parameters
    ----------
    flow : torch.Tensor
        Flow tensor with 3–4 dimensions arranged as either
        ``(2, H, W)``, ``(H, W, 2)``, or their batched variants.

    Returns
    -------
    torch.Tensor
        Tensor reshaped to ``(B, 2, H, W)``.
    str
        Identifier describing the original layout.

    Raises
    ------
    ValueError
        If the layout cannot be inferred.
    """
    tensor = flow
    if tensor.ndim == 4:
        if tensor.shape[1] == 2:
            layout = "b2hw"
        elif tensor.shape[-1] == 2:
            tensor = tensor.permute(0, 3, 1, 2)
            layout = "bhw2"
        else:
            raise ValueError("Expected flow tensor with 2 channels.")
    elif tensor.ndim == 3:
        if tensor.shape[0] == 2:
            tensor = tensor.unsqueeze(0)
            layout = "2hw"
        elif tensor.shape[-1] == 2:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            layout = "hw2"
        else:
            raise ValueError("Expected flow tensor with 2 components.")
    else:
        raise ValueError("Flow tensor must have 3 or 4 dimensions.")
    return tensor, layout


def _prepare_spatial_map(
    value,
    batch: int,
    height: int,
    width: int,
    device: "torch.device",
    dtype: "torch.dtype",
):
    """Broadcast a spatial map to ``(B, 1, H, W)``.

    Parameters
    ----------
    value : array-like or torch.Tensor or None
        Spatial weighting map. ``None`` results in a tensor of ones.
    batch : int
        Target batch size.
    height : int
        Target image height.
    width : int
        Target image width.
    device : torch.device
        Target device.
    dtype : torch.dtype
        Target dtype.

    Returns
    -------
    torch.Tensor
        Broadcasted tensor with shape ``(B, 1, H, W)``.

    Raises
    ------
    ValueError
        If ``value`` cannot be broadcast to the requested shape.
    """
    if value is None:
        return torch.ones((batch, 1, height, width), device=device, dtype=dtype)

    tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
    tensor = tensor.to(device=device, dtype=dtype)

    if tensor.ndim == 4:
        if tensor.shape[0] not in (1, batch):
            raise ValueError("Spatial map batch dimension does not match image batch.")
        if tensor.shape[0] == 1 and batch != 1:
            tensor = tensor.expand(batch, *tensor.shape[1:])
        if tensor.shape[1] == 1:
            return tensor.contiguous()
        if tensor.shape[-1] == 1:
            return tensor.permute(0, 3, 1, 2).contiguous()
    elif tensor.ndim == 3:
        if tensor.shape[0] == batch:
            return tensor.unsqueeze(1).contiguous()
        if tensor.shape[0] == 1 and batch != 1:
            tensor = tensor.expand(batch, -1, -1)
            return tensor.unsqueeze(1).contiguous()
        if tensor.shape == (height, width):
            tensor = tensor.unsqueeze(0).expand(batch, -1, -1)
            return tensor.unsqueeze(1).contiguous()
    elif tensor.ndim == 2 and tensor.shape == (height, width):
        tensor = tensor.unsqueeze(0).unsqueeze(0).expand(batch, 1, -1, -1)
        return tensor.contiguous()

    raise ValueError("Spatial map must broadcast to shape (B, 1, H, W).")


def _forward_splat_impl(
    image: "torch.Tensor",
    flow: "torch.Tensor",
    weight_map: "torch.Tensor",
) -> "torch.Tensor":
    """Perform weighted forward splatting in ``(B, C, H, W)`` layout.

    Parameters
    ----------
    image : torch.Tensor
        Image tensor in ``(B, C, H, W)`` layout.
    flow : torch.Tensor
        Flow tensor in ``(B, 2, H, W)`` layout.
    weight_map : torch.Tensor
        Per-pixel weights with shape ``(B, 1, H, W)``.

    Returns
    -------
    torch.Tensor
        Forward-splatted image in ``(B, C, H, W)`` layout.
    """
    batch, channels, height, width = image.shape
    image_flat = image.view(batch, channels, height * width)

    device = image.device
    dtype = image.dtype

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1)
    yy = yy.unsqueeze(0).expand(batch, -1, -1)

    flow_x = flow[:, 0, :, :]
    flow_y = flow[:, 1, :, :]

    target_x = xx + flow_x
    target_y = yy + flow_y

    x0 = torch.floor(target_x)
    y0 = torch.floor(target_y)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = target_x - x0
    fy = target_y - y0

    def _mask(x_coord, y_coord):
        return (
            (x_coord >= 0)
            & (x_coord <= width - 1)
            & (y_coord >= 0)
            & (y_coord <= height - 1)
        ).to(dtype)

    base_weight = weight_map.squeeze(1)

    mask00 = _mask(x0, y0)
    mask10 = _mask(x1, y0)
    mask01 = _mask(x0, y1)
    mask11 = _mask(x1, y1)

    w00 = (1 - fx) * (1 - fy) * base_weight * mask00
    w10 = fx * (1 - fy) * base_weight * mask10
    w01 = (1 - fx) * fy * base_weight * mask01
    w11 = fx * fy * base_weight * mask11

    x0i = x0.clamp(0, width - 1).long()
    x1i = x1.clamp(0, width - 1).long()
    y0i = y0.clamp(0, height - 1).long()
    y1i = y1.clamp(0, height - 1).long()

    out = torch.zeros((batch, channels, height * width), device=device, dtype=dtype)
    den = torch.zeros((batch, 1, height * width), device=device, dtype=dtype)

    def _splat(weights, x_idx, y_idx):
        if torch.count_nonzero(weights) == 0:
            return
        linear_idx = (y_idx * width + x_idx).view(batch, 1, -1)
        weights_flat = weights.view(batch, 1, -1)
        expanded_idx = linear_idx.expand(-1, channels, -1)

        out.scatter_add_(2, expanded_idx, image_flat * weights_flat)
        den.scatter_add_(2, linear_idx, weights_flat)

    for weights, x_idx, y_idx in (
        (w00, x0i, y0i),
        (w10, x1i, y0i),
        (w01, x0i, y1i),
        (w11, x1i, y1i),
    ):
        _splat(weights, x_idx, y_idx)

    den = torch.where(den == 0, torch.ones_like(den), den)
    return (out / den).view(batch, channels, height, width)


def forward_splat(image, flow, weight=None):
    """Forward warp an image using bilinear splatting.

    Parameters
    ----------
    image : array-like or torch.Tensor
        Image in any supported layout, e.g. ``(B, C, H, W)``, ``(C, H, W)``,
        ``(H, W, C)`` or ``(H, W)``.
    flow : array-like or torch.Tensor
        Flow field with layout ``(B, 2, H, W)`` or ``(H, W, 2)`` describing the
        forward displacement per pixel.
    weight : array-like or torch.Tensor, optional
        Additional per-pixel weights broadcastable to ``(B, 1, H, W)``. When
        omitted, uniform weights are used.

    Returns
    -------
    Same type as ``image``
        Forward warped image converted back to the original layout and dtype.

    Raises
    ------
    ImportError
        If ``torch`` is not available.
    ValueError
        If image and flow batch or spatial dimensions do not align.
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for forward splatting.")

    input_is_tensor = torch.is_tensor(image)
    input_is_numpy = isinstance(image, np.ndarray)
    image_tensor = image if input_is_tensor else torch.as_tensor(image)
    flow_tensor = flow if torch.is_tensor(flow) else torch.as_tensor(flow)

    orig_dtype = image_tensor.dtype
    orig_device = image_tensor.device if input_is_tensor else None
    orig_numpy_dtype = image.dtype if input_is_numpy else None

    common_device = flow_tensor.device
    if image_tensor.device != common_device:
        if common_device.type == "cpu":
            common_device = image_tensor.device
        else:
            image_tensor = image_tensor.to(device=common_device)
    else:
        common_device = image_tensor.device

    compute_dtype = torch.promote_types(image_tensor.dtype, torch.float32)
    image_tensor = image_tensor.to(device=common_device, dtype=compute_dtype)
    flow_tensor = flow_tensor.to(device=common_device, dtype=compute_dtype)

    image_bchw, image_layout = _to_bchw(image_tensor)
    flow_b2hw, _ = _to_b2hw(flow_tensor)

    if image_bchw.shape[0] != flow_b2hw.shape[0]:
        if image_bchw.shape[0] == 1:
            image_bchw = image_bchw.expand(flow_b2hw.shape[0], -1, -1, -1)
        elif flow_b2hw.shape[0] == 1:
            flow_b2hw = flow_b2hw.expand(image_bchw.shape[0], -1, -1, -1)
        else:
            raise ValueError("Image and flow batch sizes must match.")

    if image_bchw.shape[2:] != flow_b2hw.shape[2:]:
        raise ValueError("Image and flow spatial dimensions must align.")

    weight_map = _prepare_spatial_map(
        weight,
        image_bchw.shape[0],
        image_bchw.shape[2],
        image_bchw.shape[3],
        image_bchw.device,
        image_bchw.dtype,
    )

    result = _forward_splat_impl(image_bchw, flow_b2hw, weight_map)
    result = _from_bchw(result, image_layout)

    if input_is_tensor:
        result = result.to(dtype=orig_dtype)
        if orig_device is not None:
            result = result.to(device=orig_device)
        return result
    if input_is_numpy:
        np_result = result.cpu().numpy()
        if orig_numpy_dtype is not None:
            np_result = np_result.astype(orig_numpy_dtype, copy=False)
        return np_result
    return result


def softmax_splat(image, flow, importance, temperature=1.0, clamp=50.0):
    """Forward warp an image using softmax splatting (Niklaus et al., 2020).

    Parameters
    ----------
    image : array-like or torch.Tensor
        Image tensor in any layout supported by :func:`forward_splat`.
    flow : array-like or torch.Tensor
        Flow field with layout ``(B, 2, H, W)`` or ``(H, W, 2)`` describing the
        forward displacement per pixel.
    importance : array-like or torch.Tensor
        Importance or occlusion map broadcastable to ``(B, 1, H, W)`` that
        controls blending weights.
    temperature : float, optional
        Softmax temperature. Higher values sharpen the aggregation and lower
        values smooth the blend. Default is ``1.0``.
    clamp : float, optional
        Absolute value used to clamp the exponent for numerical stability.
        Default is ``50.0``.

    Returns
    -------
    Same type as ``image``
        Forward warped image aggregated with softmax-normalised weights.

    Raises
    ------
    ImportError
        If ``torch`` is not available.
    ValueError
        If image and flow batch or spatial dimensions do not align.
    """
    if not HAS_TORCH:
        raise ImportError("torch is required for softmax splatting.")

    input_is_tensor = torch.is_tensor(image)
    input_is_numpy = isinstance(image, np.ndarray)
    image_tensor = image if input_is_tensor else torch.as_tensor(image)
    flow_tensor = flow if torch.is_tensor(flow) else torch.as_tensor(flow)

    orig_dtype = image_tensor.dtype
    orig_device = image_tensor.device if input_is_tensor else None
    orig_numpy_dtype = image.dtype if input_is_numpy else None

    common_device = flow_tensor.device
    if image_tensor.device != common_device:
        if common_device.type == "cpu":
            common_device = image_tensor.device
        else:
            image_tensor = image_tensor.to(device=common_device)
    else:
        common_device = image_tensor.device

    compute_dtype = torch.promote_types(image_tensor.dtype, torch.float32)
    image_tensor = image_tensor.to(device=common_device, dtype=compute_dtype)
    flow_tensor = flow_tensor.to(device=common_device, dtype=compute_dtype)

    image_bchw, image_layout = _to_bchw(image_tensor)
    flow_b2hw, _ = _to_b2hw(flow_tensor)

    if image_bchw.shape[0] != flow_b2hw.shape[0]:
        if image_bchw.shape[0] == 1:
            image_bchw = image_bchw.expand(flow_b2hw.shape[0], -1, -1, -1)
        elif flow_b2hw.shape[0] == 1:
            flow_b2hw = flow_b2hw.expand(image_bchw.shape[0], -1, -1, -1)
        else:
            raise ValueError("Image and flow batch sizes must match.")

    if image_bchw.shape[2:] != flow_b2hw.shape[2:]:
        raise ValueError("Image and flow spatial dimensions must align.")

    importance_map = _prepare_spatial_map(
        importance,
        image_bchw.shape[0],
        image_bchw.shape[2],
        image_bchw.shape[3],
        image_bchw.device,
        image_bchw.dtype,
    )

    scaled = torch.clamp(importance_map * temperature, min=-clamp, max=clamp)
    weight_map = torch.exp(scaled)

    result = _forward_splat_impl(image_bchw, flow_b2hw, weight_map)
    result = _from_bchw(result, image_layout)

    if input_is_tensor:
        result = result.to(dtype=orig_dtype)
        if orig_device is not None:
            result = result.to(device=orig_device)
        return result
    if input_is_numpy:
        np_result = result.cpu().numpy()
        if orig_numpy_dtype is not None:
            np_result = np_result.astype(orig_numpy_dtype, copy=False)
        return np_result
    return result


class TorchForwardSplatWarper:
    """Torch implementation of forward and softmax splatting."""

    def __init__(
        self, mode: str = "average", temperature: float = 1.0, clamp: float = 50.0
    ):
        if not HAS_TORCH:
            raise ImportError("torch is required for TorchForwardSplatWarper.")
        if mode not in {"average", "softmax"}:
            raise ValueError("mode must be either 'average' or 'softmax'.")
        self.mode = mode
        self.temperature = temperature
        self.clamp = clamp

    def __call__(self, image, flow, *, importance=None, weight=None):
        if self.mode == "average":
            return forward_splat(image, flow, weight=weight)
        if importance is None:
            raise ValueError("importance map must be provided for softmax splatting.")
        return softmax_splat(
            image,
            flow,
            importance=importance,
            temperature=self.temperature,
            clamp=self.clamp,
        )


__all__ = [
    "HAS_MODERNGL",
    "HAS_TORCH",
    "warp_image_pc",
    "warp_image_pc_single",
    "warp_image_backwards",
    "OnlineFrameWarper",
    "forward_splat",
    "softmax_splat",
    "TorchForwardSplatWarper",
]
