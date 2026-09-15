"""
Why a zone stays silent when it should speak.

A 6-minute session on the Pi logged 136 tracker resets - one per dropout where
MediaPipe briefly lost the hand, 97 of them under 0.1 s - and 116
announcements suppressed as `same_name` against 76 that played. In the 0.8-2s
window right after a reset the ratio was 39 suppressed to 10 played. Both
halves are covered here: the main loop no longer treats a momentary dropout as
the hand leaving, and a double-tap no longer loses to the change check.
"""

import time

import simple_camio
from src.audio.audio import ZoneAudioPlayer
from src.config import InteractionConfig

GRACE = InteractionConfig.HAND_LOSS_GRACE_SECONDS


class FakeAudioWorker:
    """Records commands instead of playing them."""

    def __init__(self):
        self.commands = []

    def enqueue_command(self, command):
        self.commands.append(command.command_type)


def _hand_state(missing_for=None):
    """A tracked hand, optionally already missing for that many seconds."""
    return {
        'was_detected': True,
        'description_played': True,
        'first_detected_ts': 100.0,
        'missing_since': 0.0 if missing_for is None else time.time() - missing_for,
    }


# --- one dropped frame is not the hand leaving ----------------------------


def test_a_momentary_dropout_keeps_the_hand():
    """
    The pose worker republishes ~14 times a second while the loop polls at ~30,
    so a dropped detection is routine - the median one lasted 3 ms. Acting on
    it restarted the whole "hand appeared" scenario and marked the zone under
    the finger as already spoken.
    """
    state = _hand_state()
    worker = FakeAudioWorker()

    simple_camio.mark_hand_missing(state, worker)
    simple_camio.mark_hand_missing(state, worker)

    assert state['was_detected'] is True
    assert state['first_detected_ts'] == 100.0
    assert worker.commands == []


def test_the_first_missing_frame_starts_the_clock():
    """Without a start time the window could never expire."""
    state = _hand_state()

    simple_camio.mark_hand_missing(state, FakeAudioWorker())

    assert state['missing_since'] > 0


def test_the_hand_still_leaves_once_the_grace_runs_out():
    """The other side of the gate, so the tests above cannot pass vacuously."""
    state = _hand_state(missing_for=GRACE + 0.1)
    worker = FakeAudioWorker()

    simple_camio.mark_hand_missing(state, worker)

    assert state['was_detected'] is False
    assert state['first_detected_ts'] == 0.0
    assert worker.commands == ['heartbeat_pause', 'crickets_play']


def test_a_recovered_frame_clears_the_clock():
    """
    Flicker is a near-miss, a hit, then another near-miss. Without the reset
    the second one would inherit the first one's start time and tear down a
    hand that never actually left.
    """
    state = _hand_state(missing_for=GRACE - 0.05)
    worker = FakeAudioWorker()

    state['missing_since'] = 0.0  # what a valid gesture does
    simple_camio.mark_hand_missing(state, worker)

    assert state['was_detected'] is True
    assert worker.commands == []


def test_crickets_start_once_not_every_frame_the_hand_is_gone():
    """A hand off the map for a minute must not queue a command per frame."""
    state = _hand_state(missing_for=GRACE + 0.1)
    worker = FakeAudioWorker()

    for _ in range(30):
        simple_camio.mark_hand_missing(state, worker)

    assert worker.commands == ['heartbeat_pause', 'crickets_play']


# --- double-tap is an explicit repeat --------------------------------------


def _player():
    """A player with the zone bookkeeping convey() touches, and nothing else."""
    player = ZoneAudioPlayer.__new__(ZoneAudioPlayer)
    player.hotspots = {7: {'textDescription': 'Бухгалтерія'}}
    player.prev_zone_name = 'Бухгалтерія'
    player.played = []
    player._play_zone_audio = player.played.append
    return player


def test_a_double_tap_repeats_the_zone_it_just_announced():
    """
    Of 6 double-taps in the Pi session exactly one made a sound; the rest lost
    to this check or landed on the background.
    """
    player = _player()

    player.convey(7, 'double_tap')

    assert player.played == [7]


def test_pointing_at_the_same_zone_still_says_nothing():
    """Otherwise a resting finger would repeat its zone forever."""
    player = _player()

    player.convey(7, 'pointing')

    assert player.played == []


def test_a_double_tap_on_the_background_still_says_nothing():
    """The zone has to exist; 4 of the 6 logged double-taps hit the backdrop."""
    player = _player()

    player.convey(16777215, 'double_tap')

    assert player.played == []
    assert player.prev_zone_name is None
