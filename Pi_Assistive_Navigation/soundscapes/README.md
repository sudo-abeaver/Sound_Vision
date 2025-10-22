# Soundscapes Audio Files

This directory contains ambient audio files that play automatically based on detected scenes.

## Directory Structure

Each folder should contain one or more audio files (MP3, WAV, OGG):

- `forest/` - Forest ambience, birds chirping, rustling leaves
- `park/` - Park sounds, distant children, light activity
- `beach/` - Ocean waves, seagulls, wind
- `city/` - Urban traffic, city buzz, horns
- `residential/` - Quiet neighborhood, distant sounds
- `indoor/` - Indoor ambient, HVAC, echo
- `museum/` - Classical music pieces (switches on art detection)
- `plaza/` - Public square, fountains, crowds
- `parking/` - Parking lot, car engines, footsteps

## File Requirements

- **Format**: MP3 or WAV (MP3 recommended for smaller size)
- **Length**: 2-10 minutes (will loop)
- **Quality**: 128-192 kbps MP3 or 44.1kHz WAV
- **Naming**: Descriptive names (e.g., `forest_birds_01.mp3`)

## Free Audio Sources

### Sound Effects
- [Freesound.org](https://freesound.org) - Community sound library
- [BBC Sound Effects](https://sound-effects.bbcrewind.co.uk) - 16,000+ free effects
- [SoundBible](https://soundbible.com) - Free sound clips

### Music (for museum/)
- [Free Music Archive](https://freemusicarchive.org)
- [Incompetech](https://incompetech.com) - Royalty-free music
- [Musopen](https://musopen.org) - Public domain classical

### Ambiences
- [Ambient Mixer](https://www.ambient-mixer.com)
- [MyNoise.net](https://mynoise.net)
- YouTube (with proper licensing)

## Tips

1. **Seamless Loops**: Use audio that loops naturally
2. **Consistent Volume**: Normalize all files to similar levels
3. **Variety**: Add 2-3 variations per folder for variety
4. **File Size**: Keep under 10MB each for faster loading
5. **Licensing**: Ensure you have rights to use the audio

## Example Setup

```bash
# Download free ocean waves
cd beach/
wget https://example.com/ocean_waves.mp3

# Convert MP3 to normalize volume
ffmpeg -i ocean_waves.mp3 -af "volume=0.8" ocean_normalized.mp3

# Create seamless loop
ffmpeg -stream_loop 5 -i ocean_normalized.mp3 -c copy ocean_loop.mp3
```

## Testing

Test audio playback:

```bash
# Using pygame (what the system uses)
python3 -c "import pygame; pygame.mixer.init(); pygame.mixer.music.load('forest/forest_ambient.mp3'); pygame.mixer.music.play(-1); import time; time.sleep(10)"

# Using system player
aplay forest/forest_ambient.wav
mpg123 city/city_traffic.mp3
```

## Current Status

Place your audio files in the appropriate folders. The system will:
- Randomly select a file when entering a scene
- Loop the file continuously
- Crossfade (3.5s) when scene changes
- Duck volume when speaking or beeping

**Note**: This repo does not include audio files. You must provide your own.

