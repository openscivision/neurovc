import numpy as np
import pytest

from neurovc.momag import flow_processing as fp


def test_get_motion_magnitude_returns_expected_norm():
    flow = np.array([[[3.0, 4.0]]], dtype=np.float32)
    magnitudes = fp.get_motion_magnitude(flow)
    assert magnitudes.shape == (1, 1)
    assert np.allclose(magnitudes[0, 0], 5.0)


def test_compressive_function_thresh_scales_below_threshold():
    g_mag = np.array([[0.5, 1.5]], dtype=np.float32)
    result = fp.compressive_function_thresh(g_mag.copy(), alpha=2.0, threshold=1.0)
    expected = np.array([[1.0, 1.5]], dtype=np.float32)
    assert np.allclose(result, expected)


def test_const_compressor_scales_flow():
    flow = np.ones((1, 1, 2), dtype=np.float32)
    compressor = fp.ConstCompressor(alpha=2.0)
    out = compressor(flow.copy())
    g = np.sqrt(2.0)
    expected_value = (2.0 * g) / (g + 0.0001)
    assert np.allclose(out, expected_value)


def test_thresh_compressor_applies_threshold():
    flow = np.ones((1, 1, 2), dtype=np.float32)
    compressor = fp.ThreshCompressor(alpha=3.0, threshold=2.0)
    out = compressor(flow.copy())
    g = np.sqrt(2.0)
    expected_value = (3.0 * g) / (g + 0.0001)
    assert np.allclose(out, expected_value)


def test_flow_decomposer_creates_mask_and_decomposes_zero_flow():
    landmarks = np.array(
        [
            [8.0, 8.0, -0.2],
            [24.0, 8.0, -0.1],
            [24.0, 24.0, 0.1],
            [8.0, 24.0, 0.3],
        ],
        dtype=np.float32,
    )
    decomposer = fp.FlowDecomposer(landmarks, (32, 32), [0, 1, 2, 3])
    assert decomposer.mask.shape == (32, 32)
    assert np.any(decomposer.mask > 0)

    flow = np.zeros((32, 32, 2), dtype=np.float32)
    global_flow, local_flow = decomposer.decompose(flow)
    assert np.allclose(global_flow, 0.0)
    assert np.allclose(local_flow, 0.0)


def test_alpha_looper_cycles_values():
    looper = fp.AlphaLooper(alpha=(0.0, 2.0), step=0.5)
    values = [looper() for _ in range(6)]
    assert values == [0.0, 0.5, 1.0, 1.5, 1.5, 1.0]
    looper.reset()
    assert looper() == 0.0


def test_extract_landmarks_success():
    class _DummyLM:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

    class _DummyResults:
        def __init__(self):
            self.multi_face_landmarks = [
                type(
                    "LMList",
                    (),
                    {"landmark": [_DummyLM(0.1, 0.2, 0.3), _DummyLM(0.4, 0.5, 0.6)]},
                )
            ]

    result = fp._extract_landmarks(_DummyResults())
    expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    assert np.allclose(result, expected)


def test_extract_landmarks_raises_without_faces():
    with pytest.raises(ValueError):
        fp._extract_landmarks(None)


class _DummyWarper:
    def __init__(self):
        self.calls = []

    def warp_image_uv(self, base, uv, depth):
        self.calls.append((base, uv, depth))
        return {"base": base, "uv": uv, "depth": depth}


class _DummyFlowStrategy:
    def __init__(self, flow_global, flow_local):
        self.flow_global = flow_global
        self.flow_local = flow_local
        self.updated_refs = []
        self.calls = []

    def update_reference(self, ref, landmarks=None):
        self.updated_refs.append((ref, landmarks))

    def __call__(self, frame, landmarks=None):
        self.calls.append((frame, landmarks))
        return self.flow_global, self.flow_local


def test_online_motion_magnifier_initializes_and_warps_with_default_alpha():
    frame = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    flow_global = np.zeros((2, 2, 2), dtype=np.float32)
    flow_local = np.ones((2, 2, 2), dtype=np.float32)

    strategy = _DummyFlowStrategy(flow_global, flow_local)
    warper = _DummyWarper()
    warper_factory_calls = []

    def warper_factory(hw):
        warper_factory_calls.append(hw)
        return warper

    magnifier = fp.OnlineMotionMagnifier(
        strategy,
        alpha=2.0,
        warper_factory=warper_factory,
    )

    result = magnifier(frame, landmarks="lm")

    expected_base = fp.warp_image_backwards(frame, flow_global + flow_local)
    expected_uv = flow_global + 2.0 * flow_local

    assert strategy.updated_refs == [(frame, "lm")]
    assert strategy.calls == [(frame, "lm")]
    assert warper_factory_calls == [frame.shape[:2]]
    assert len(warper.calls) == 1
    assert np.allclose(result["base"], expected_base)
    assert np.allclose(result["uv"], expected_uv)
    assert result["depth"] is None


def test_online_motion_magnifier_uses_depth_provider_and_alpha_override():
    frame = np.ones((2, 2, 3), dtype=np.float32)
    flow_global = np.zeros((2, 2, 2), dtype=np.float32)
    flow_local = 0.5 * np.ones((2, 2, 2), dtype=np.float32)
    depth_map = np.full(frame.shape[:2], 0.3, dtype=np.float32)

    strategy = _DummyFlowStrategy(flow_global, flow_local)
    warper = _DummyWarper()
    depth_calls = []

    def warper_factory(hw):
        return warper

    def depth_provider():
        depth_calls.append(True)
        return depth_map

    magnifier = fp.OnlineMotionMagnifier(
        strategy,
        alpha=10.0,
        warper_factory=warper_factory,
    )
    magnifier.set_depth_provider(depth_provider)

    result = magnifier(frame, alpha=0.25)

    expected_uv = flow_global + 0.25 * flow_local

    assert depth_calls == [True]
    assert np.allclose(result["uv"], expected_uv)
    assert np.allclose(result["depth"], depth_map)
    assert len(warper.calls) == 1
