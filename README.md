# YouTube Code and Content Extractor

This project consists of Python scripts for extracting and analyzing content from YouTube videos, with a focus on recognizing code snippets in programming tutorials.

## Features

- Extract transcripts from YouTube videos
- Download audio and video content from YouTube
- Analyze video transcripts using OpenAI GPT models
- Extract code snippets from video frames
- Generate comprehensive analysis of video content
- Save results in both Markdown and JSON formats

## Requirements

### Dependencies

- Python 3.8+
- OpenAI API key (for GPT-based analysis)
- Libraries listed in the scripts (see Installation)

### Optional

- Tesseract OCR (for text recognition from images)
- Whisper (for speech-to-text when YouTube transcripts aren't available)

## Installation

1. Clone the repository
2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install the required packages:
```bash
pip install pytubefix whisper openai youtube-transcript-api python-dotenv opencv-python numpy pytesseract pillow requests
```
4. Create a `.env` file in the root directory with your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Transcript Extraction and Analysis (`get_transcript.py`)

This script extracts the transcript from a YouTube video and analyzes it using OpenAI.

```bash
python get_transcript.py <youtube_url>
```

Example:
```bash
python get_transcript.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

The script will:
1. Extract the transcript from the YouTube video
2. Send the transcript to OpenAI for analysis
3. Save the transcript and analysis as both Markdown and JSON files in the `results` directory

### Comprehensive Video Analysis (`youtube.py`)

This script performs more advanced analysis including code extraction from video frames.

```bash
python youtube.py <youtube_url> [options]
```

Options:
- `--api-key`: OpenAI API key (if not specified, it's taken from OPENAI_API_KEY environment variable)
- `--api-url`: API URL for LLM (default: OpenAI API URL)
- `--whisper-model`: Whisper model size for transcript creation (choices: tiny, base, small, medium, large, default: tiny)
- `--output`: Directory for output files (default: results)
- `--temp`: Directory for temporary files (default: temp)
- `--extract-code`: Flag to extract code frames from video (must be specified to enable code extraction)
- `--video-quality`: Quality of downloaded video (choices: lowest, highest, default: highest)
- `--interval`: Interval in seconds for frame extraction (default: 1)

Example:
```bash
python youtube.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --whisper-model base --extract-code --interval 2
```

The script will:
1. Download the video and audio
2. Extract the transcript (from YouTube or using Whisper)
3. Extract frames from the video at specified intervals (if --extract-code is set)
4. Analyze frames to detect code snippets (if --extract-code is set)
5. Use GPT to extract and identify code from frames (if --extract-code is set)
6. Generate comprehensive analysis of the video content
7. Save all results in the output directory

## Extracting Code from Videos

To extract code from video frames, you must use the `--extract-code` flag with the `youtube.py` script:

```bash
python youtube.py https://www.youtube.com/watch?v=your_video_id --extract-code
```

### How Code Extraction Works

1. The script first downloads the video from YouTube using the PyTubeFix library
2. It then extracts frames at regular intervals (configurable with the `--interval` parameter)
3. Each frame is analyzed to detect if it contains code
4. Frames with potential code are processed using OpenAI's vision capabilities to extract the code
5. The extracted code is identified by language and saved to the results directory

### Best Practices for Code Extraction

- Use videos with high resolution and clear code displays
- Decrease the interval between frames to capture more code (e.g., `--interval 0.5`)
- Use the highest video quality for better text recognition (default: `--video-quality highest`)
- If the code isn't recognized correctly, try adjusting the contrast threshold in the script

### Example Command for Optimized Code Extraction

```bash
python youtube.py https://www.youtube.com/watch?v=your_video_id --extract-code --interval 0.5 --video-quality highest --whisper-model base
```

## Creating Transcripts

There are two methods for creating transcripts:

### 1. Using YouTube's Built-in Transcripts

The simplest approach is to extract existing transcripts from YouTube:

```bash
python get_transcript.py https://www.youtube.com/watch?v=your_video_id
```

This method is fast but relies on the availability of transcripts on YouTube. Many videos, especially technical tutorials, include auto-generated or manually created captions.

### 2. Creating Transcripts with Whisper

When YouTube transcripts are unavailable or of poor quality, you can generate transcripts using OpenAI's Whisper model:

```bash
python youtube.py https://www.youtube.com/watch?v=your_video_id --whisper-model medium
```

The `--whisper-model` parameter specifies the Whisper model size:
- `tiny`: Fastest but least accurate
- `base`: Good balance of speed and accuracy
- `small`: Better accuracy, slower
- `medium`: High accuracy, slower
- `large`: Best accuracy, slowest

### Processing and Formatting Transcripts

Both methods produce transcripts that are:
1. Saved as plain text and included in Markdown analysis files
2. Post-processed to match the timing with extracted frames (when using `youtube.py`)
3. Stored in the output directory (default: `results`)

## Output

All results are saved in the output directory (default: `results`):
- Transcript and analysis in Markdown format (`<video_id>_transcript.md`)
- JSON file with complete analysis results (`<video_id>_analysis.json`)
- Extracted code snippets from frames (when the `--extract-code` flag is used)

## Tips

- For better code extraction, use videos with clear, high-resolution code displays
- The higher the quality of the downloaded video, the better the code recognition
- If you encounter issues with YouTube transcript extraction, the script will fall back to Whisper for transcription
- To save disk space, you can use `--video-quality lowest` when not extracting code

## Troubleshooting

- If you see errors related to PyTube, try reinstalling the pytubefix package
- Ensure your OpenAI API key is correctly set in the `.env` file
- For OCR issues, make sure Tesseract is properly installed on your system

## License

This project is available for personal and educational use. 