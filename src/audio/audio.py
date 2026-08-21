"""
Audio playback components for Simple CamIO.

This module handles all audio-related functionality including ambient sounds,
sound effects, and zone-based audio descriptions.

HEADLESS MODE SUPPORT:
For Raspberry Pi or other headless systems without X11, this module falls back
to pygame, which needs no display for audio.

pyglet is preferred when available, but pyglet.media cannot be imported without
a display: it pulls in pyglet.gl, which creates a shadow window. So the display
is probed first and pyglet.media is left unimported when there is none -
importing it and catching the failure also works, but leaves behind an atexit
hook that raises the same error again during interpreter shutdown.
"""

import os
import sys
import logging

from src.config import TTSConfig

logger = logging.getLogger(__name__)

# Backend selection: pyglet when a display is reachable, else pygame.
AUDIO_BACKEND = None
USE_PYGLET = False
USE_PYGAME = False


def _display_reachable():
    """
    Whether pyglet can open a display, which its media stack requires.

    Probed via pyglet.display, which is cheap and creates no window, so
    pyglet.media stays unimported when the answer is no. DISPLAY is read, never
    written: setting it here would change the whole process's environment as a
    side effect of importing this module.
    """
    try:
        import pyglet.display
        pyglet.display.get_display()
        return True
    except Exception as e:
        logger.info(f"No display available for pyglet ({e})")
        return False


def _init_pygame():
    """Initialize the pygame mixer. Returns True on success."""
    global USE_PYGAME, AUDIO_BACKEND
    try:
        import pygame
        # Audio only - no display, no video subsystem.
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.init()
        USE_PYGAME = True
        AUDIO_BACKEND = 'pygame'
        logger.info("Audio backend: pygame (headless compatible)")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize pygame: {e}")
        return False


if _display_reachable():
    try:
        import pyglet.media
        USE_PYGLET = True
        AUDIO_BACKEND = 'pyglet'
        logger.info("Audio backend: pyglet")
    except Exception as e:
        logger.warning(f"Display is reachable but pyglet.media failed to load: {e}")
        _init_pygame()
else:
    _init_pygame()

if AUDIO_BACKEND is None:
    logger.warning("=" * 60)
    logger.warning("WARNING: No audio backend available! Audio will not work.")
    logger.warning("Install pygame for headless audio: pip install pygame")
    logger.warning("Or provide a display: xvfb-run python simple_camio.py --headless")
    logger.warning("=" * 60)


class AmbientSoundPlayer:
    """
    Player for looping ambient background sounds.

    This class manages a single looping audio track with volume control,
    typically used for ambient sounds like crickets or heartbeat.
    Supports both pyglet and pygame backends.
    """

    def __init__(self, soundfile):
        """
        Initialize the ambient sound player.

        Args:
            soundfile (str): Path to the audio file to play
        """
        self.soundfile = soundfile
        self.volume = 1.0
        
        if USE_PYGLET:
            import pyglet.media
            self.sound = pyglet.media.load(soundfile, streaming=False)
            self.player = pyglet.media.Player()
            self.player.queue(self.sound)
            self.player.eos_action = 'loop'
            self.player.loop = True
            logger.debug(f"Initialized pyglet ambient sound player with {soundfile}")
        elif USE_PYGAME:
            import pygame
            self.sound = pygame.mixer.Sound(soundfile)
            self.player = None
            self._playing = False
            logger.debug(f"Initialized pygame ambient sound player with {soundfile}")
        else:
            logger.warning(f"No audio backend - ambient sound not loaded: {soundfile}")
            self.sound = None
            self.player = None

    def set_volume(self, volume):
        """
        Set the playback volume.

        Args:
            volume (float): Volume level between 0.0 and 1.0
        """
        if 0 <= volume <= 1:
            self.volume = volume
            if USE_PYGLET and self.player:
                self.player.volume = volume
            elif USE_PYGAME and self.sound:
                self.sound.set_volume(volume)
            logger.debug(f"Set volume to {volume}")

    def play_sound(self):
        """Start playing the ambient sound if not already playing."""
        if USE_PYGLET and self.player:
            if not self.player.playing:
                self.player.play()
                logger.debug("Started pyglet ambient sound playback")
        elif USE_PYGAME and self.sound:
            if not self._playing:
                # Set volume before playing (pygame requirement)
                self.sound.set_volume(self.volume)
                self.sound.play(loops=-1)  # -1 means loop forever
                self._playing = True
                logger.debug(f"Started pygame ambient sound playback at volume {self.volume}")

    def pause_sound(self):
        """Pause the ambient sound if currently playing."""
        if USE_PYGLET and self.player:
            if self.player.playing:
                self.player.pause()
                logger.debug("Paused pyglet ambient sound playback")
        elif USE_PYGAME and self.sound:
            if self._playing:
                self.sound.stop()
                self._playing = False
                logger.debug("Stopped pygame ambient sound playback")


class ZoneAudioPlayer:
    """
    Manages audio playback for interactive zones on the map.

    This class handles playing audio descriptions when users interact with
    different zones, including blip sounds for zone transitions and
    descriptive audio for each hotspot.
    Supports both pyglet and pygame backends.
    """

    def __init__(self, model):
        """
        Initialize the zone audio player.

        Args:
            model (dict): Map model configuration containing audio file paths
        """
        self.model = model
        self.prev_zone_name = ''
        self.prev_zone_moving = -1
        self.curr_zone_moving = -1
        self.sound_files = {}
        self.hotspots = {}
        self.enable_blips = False

        if USE_PYGLET:
            import pyglet.media
            self.player = pyglet.media.Player()
        else:
            self.player = None
        self.welcome_player = None
        self.goodbye_player = None
        self.current_channel = None

        self._synthesize_missing()

        self.blip_sound = self._load_sound(self.model.get('blipsound'), 'blip sound')
        self.map_description = self._load_sound(
            self.model.get('map_description'), 'map description')
        self.have_played_description = self.map_description is None
        self.welcome_message = self._load_sound(
            self.model.get('welcome_message'), 'welcome message')
        self.goodbye_message = self._load_sound(
            self.model.get('goodbye_message'), 'goodbye message')

        # Load audio files for each hotspot
        self._load_hotspot_audio()

        logger.info(f"Initialized zone audio player ({AUDIO_BACKEND}) with {len(self.hotspots)} hotspots")

    def _synthesize_missing(self):
        """
        Fill in narration audio that is absent, before anything is loaded.

        Runs once at construction rather than on demand, so a zone never waits for
        synthesis while a finger is resting on it. Writes only audio files - a
        device in the field does not rewrite its own configuration - at whatever
        destination model_audio.output_path() names, the same module the offline
        generator CLI uses. For a model whose audioDescription/map_description
        already point inside model_audio's generated subdirectory (as every model
        that has already been through the CLI does), that destination is the
        literal path the JSON names, so nothing changes for them; a model that
        still names an un-generated path elsewhere (e.g. Audio/X.mp3) gets its
        clip written under Audio/tts/X.wav instead of as WAV bytes under the
        literal .mp3 name.

        model_audio.pending() also owns the missing-file test, so this and the
        CLI cannot drift on what counts as "already generated".

        Every failure here is logged and tolerated. The player continues with
        whatever files do exist.
        """
        if not TTSConfig.RUNTIME_FALLBACK:
            return

        # Lazy: Piper may not be installed, and that must not stop the app starting.
        from src.tts import engine, model_audio

        todo = model_audio.pending(model_audio.narration_entries(self.model))
        if not todo:
            return

        logger.info(f"{len(todo)} narration clip(s) missing - synthesizing")
        jobs = [
            engine.SynthesisJob(text=entry.text, output_path=model_audio.output_path(entry))
            for entry in todo
        ]

        def report(done, total, job):
            logger.info(f"synthesizing {done}/{total}: {job.output_path.name}")

        try:
            _, failed = engine.synthesize(jobs, progress=report)
        except engine.TTSUnavailable as e:
            logger.warning(f"Cannot synthesize the missing narration: {e}")
            return
        except Exception as e:
            logger.error(f"Synthesis failed unexpectedly: {e}", exc_info=True)
            return

        for job in failed:
            logger.warning(
                f"No audio for {job.output_path.name}; that zone will stay silent")

    def _load_sound(self, path, label=None):
        """
        Load one audio file with whichever backend is active.

        Args:
            path (str): Path to the audio file.
            label (str, optional): Name to use in log messages.

        Returns:
            The backend's sound object, or None when the path is empty, the file is
            absent, or no backend is available. Callers must tolerate None: one
            silent clip is survivable where an exception would end the session.
        """
        if not path:
            return None

        label = label or path
        if not os.path.exists(path):
            logger.warning(f"Audio file not found: {path}")
            return None

        try:
            if USE_PYGLET:
                import pyglet.media
                return pyglet.media.load(path, streaming=False)
            if USE_PYGAME:
                import pygame
                return pygame.mixer.Sound(path)
        except Exception as e:
            logger.error(f"Could not load {label}: {e}")
        return None

    def _load_hotspot_audio(self):
        """Load audio files for all hotspots defined in the model."""
        for hotspot in self.model['hotspots']:
            # Create unique key from color
            key = (hotspot['color'][2] +
                   hotspot['color'][1] * 256 +
                   hotspot['color'][0] * 256 * 256)

            self.hotspots[key] = hotspot

            sound = self._load_sound(
                hotspot.get('audioDescription'), hotspot.get('textDescription'))
            if sound is not None:
                self.sound_files[key] = sound

    def set_zone_volume(self, volume):
        """
        Set the volume for zone audio playback (descriptions, welcome, goodbye).
        
        Args:
            volume (float): Volume level between 0.0 and 1.0
        """
        if not 0 <= volume <= 1:
            logger.warning(f"Invalid volume {volume}, must be between 0.0 and 1.0")
            return
            
        # Note: For pygame, volumes are set per-sound when playing
        # For pyglet, we'll store the volume to apply when playing
        self.zone_volume = volume
        
        # Set volume for all loaded zone sounds
        if USE_PYGAME:
            if self.blip_sound:
                self.blip_sound.set_volume(volume * 0.3)  # Blips quieter than descriptions
            if hasattr(self, 'map_description') and self.map_description:
                self.map_description.set_volume(volume)
            if self.welcome_message:
                self.welcome_message.set_volume(volume)
            if self.goodbye_message:
                self.goodbye_message.set_volume(volume)
            for sound in self.sound_files.values():
                sound.set_volume(volume)
        
        logger.debug(f"Set zone audio volume to {volume}")

    def play_description(self):
        """Play the map description audio (only once)."""
        if not self.have_played_description:
            if USE_PYGLET:
                self.player = self.map_description.play()
            elif USE_PYGAME:
                self.map_description.play()
            self.have_played_description = True
            logger.info("Playing map description")

    def play_welcome(self):
        """Play the welcome message."""
        if USE_PYGLET:
            # Stop previous welcome if still playing
            if self.welcome_player and self.welcome_player.playing:
                self.welcome_player.pause()
                self.welcome_player.delete()
            self.welcome_player = self.welcome_message.play()
        elif USE_PYGAME:
            self.welcome_message.play()
        logger.info("Playing welcome message")

    def play_goodbye(self, blocking=False):
        """
        Play the goodbye message.
        
        Args:
            blocking (bool): If True, returns player object for caller to manage.
                           If False (default), plays asynchronously.
        
        Returns:
            pyglet.media.Player or None: Player object if blocking=True, else None
        """
        if USE_PYGLET:
            # Stop previous goodbye if still playing
            if self.goodbye_player and self.goodbye_player.playing:
                logger.info("Stopping previous goodbye player")
                self.goodbye_player.pause()
                self.goodbye_player.delete()
            
            try:
                player = self.goodbye_message.play()
                logger.info("Playing goodbye message")
                if blocking:
                    return player
                else:
                    self.goodbye_player = player
                    return None
            except Exception as e:
                logger.error(f"Error starting goodbye player: {e}", exc_info=True)
                raise
        elif USE_PYGAME:
            self.goodbye_message.play()
            logger.info("Playing goodbye message (pygame)")
            return None
        
        return None
    
    def stop_all(self):
        """
        Stop all currently playing audio.
        
        Stops zone audio, welcome, goodbye, description, and blips.
        """
        logger.info("Stopping all ZoneAudioPlayer sounds...")
        
        if USE_PYGLET:
            # Stop main zone player
            try:
                if self.player and self.player.playing:
                    self.player.pause()
                    self.player.delete()
            except Exception as e:
                logger.debug(f"Error stopping main player: {e}")
            
            # Stop welcome player
            try:
                if self.welcome_player and self.welcome_player.playing:
                    self.welcome_player.pause()
                    self.welcome_player.delete()
            except Exception as e:
                logger.debug(f"Error stopping welcome player: {e}")
            
            # Stop goodbye player
            try:
                if self.goodbye_player and self.goodbye_player.playing:
                    self.goodbye_player.pause()
                    self.goodbye_player.delete()
            except Exception as e:
                logger.debug(f"Error stopping goodbye player: {e}")
        
        elif USE_PYGAME:
            import pygame
            pygame.mixer.stop()  # Stop all channels
            logger.debug("Stopped all pygame mixer channels")

    def convey(self, zone, status):
        """
        Play audio based on zone interaction.

        Args:
            zone (int): Zone ID that the user is interacting with
            status (str): Interaction status ('moving', 'still', 'double_tap', etc.)
        """
        # Handle moving status with blip sounds
        if status == "moving":
            self._handle_moving_zone(zone)
            return

        # Ignore invalid zones
        if zone not in self.hotspots:
            self.prev_zone_name = None
            return

        # Get zone name and play audio if zone changed
        zone_name = self.hotspots[zone]['textDescription']
        if self.prev_zone_name != zone_name:
            self._play_zone_audio(zone)
            self.prev_zone_name = zone_name

    def _handle_moving_zone(self, zone):
        """
        Handle audio for moving through zones (play blip sound).

        Args:
            zone (int): Current zone ID
        """
        if (self.curr_zone_moving != zone and
            self.prev_zone_moving == zone and
            self.enable_blips):

            if USE_PYGLET:
                if self.player and self.player.playing:
                    self.player.delete()
                try:
                    self.player = self.blip_sound.play()
                    logger.debug(f"Playing blip for zone {zone}")
                except Exception as e:
                    logger.error(f"Cannot play blip sound: {e}")
            elif USE_PYGAME:
                self.blip_sound.play()
                logger.debug(f"Playing blip for zone {zone}")

            self.curr_zone_moving = zone

        self.prev_zone_moving = zone

    def _play_zone_audio(self, zone):
        """
        Play the audio description for a specific zone.

        Args:
            zone (int): Zone ID to play audio for
        """
        if USE_PYGLET:
            # Stop current audio
            if self.player:
                self.player.pause()
                self.player.delete()

            # Play new audio if available
            if zone in self.sound_files:
                sound = self.sound_files[zone]
                try:
                    self.player = sound.play()
                    logger.debug(f"Playing audio for zone {zone}")
                except Exception as e:
                    logger.error(f"Cannot play zone audio: {e}")
        
        elif USE_PYGAME:
            # Stop current audio
            import pygame
            pygame.mixer.stop()
            
            # Play new audio if available
            if zone in self.sound_files:
                sound = self.sound_files[zone]
                try:
                    sound.play()
                    logger.debug(f"Playing audio for zone {zone}")
                except Exception as e:
                    logger.error(f"Cannot play zone audio: {e}")

