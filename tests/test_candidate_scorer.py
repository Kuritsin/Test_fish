import numpy as np
import cv2

from config import BotConfig
from fishing_bot.automatic_bobber_finder import build_background, novelty_mask
from fishing_bot.candidate_scorer import score_candidate


def test_composite_bobber_scores_above_plain_water_patch():
    frame = np.full((80, 100, 3), (85, 70, 55), dtype=np.uint8)
    frame[40:52, 45:58] = (65, 145, 220)  # cork body
    frame[35:43, 35:48] = (15, 25, 220)   # red feather
    frame[31:37, 47:53] = (230, 230, 230)  # bright fitting
    bobber = score_candidate(frame, (35, 31, 23, 21))
    water = score_candidate(frame, (5, 5, 23, 21))
    assert bobber.total > water.total
    assert bobber.body > 0 and bobber.colour > 0 and bobber.diamond > 0


def test_multi_frame_background_suppresses_preexisting_water_motion():
    config = BotConfig(auto_find_pixel_difference=15)
    frames = []
    for value in (20, 45, 25, 50, 30):
        frame = np.zeros((40, 50, 3), dtype=np.uint8)
        frame[10:20, 10:20] = value
        frames.append(frame)
    after = frames[-1].copy()
    after[25:31, 30:36] = (0, 0, 240)
    model = build_background(frames)
    mask = novelty_mask(model, after, config)
    assert np.count_nonzero(mask[24:33, 29:38]) > 0


def test_template_variant_adds_multiscale_evidence():
    template = np.zeros((14, 18, 3), dtype=np.uint8)
    template[5:12, 8:15] = (55, 140, 220)
    template[2:7, 2:10] = (10, 20, 230)
    frame = np.zeros((70, 90, 3), dtype=np.uint8)
    enlarged = cv2.resize(template, (27, 21))
    frame[25:46, 30:57] = enlarged
    score = score_candidate(frame, (30, 25, 27, 21), templates=[template], scales=(1.5,))
    assert score.template > 0.7
